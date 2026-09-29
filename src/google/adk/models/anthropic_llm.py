# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic integration for Claude models."""

from __future__ import annotations

import base64
import copy
import dataclasses
from functools import cached_property
import json
import logging
import os
import re
from typing import Any
from typing import AsyncGenerator
from typing import Iterable
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from anthropic import AsyncAnthropic
from anthropic import AsyncAnthropicVertex
from anthropic import NOT_GIVEN
from anthropic import NotGiven
from anthropic import types as anthropic_types
from google.genai import types
from pydantic import BaseModel
from typing_extensions import override

from . import _prompt_cache
from ..utils import _json_utils
from ..utils import streaming_utils
from ..utils._google_client_headers import get_tracking_headers
from .base_llm import BaseLlm
from .llm_response import LlmResponse

if TYPE_CHECKING:
    from .llm_request import LlmRequest

__all__ = ["AnthropicLlm", "Claude"]

logger = logging.getLogger("google_adk." + __name__)


@dataclasses.dataclass
class _ToolUseAccumulator:
    """Accumulates streamed tool_use content block data."""

  id: str
  name: str
  args_json: str
  tracker: streaming_utils._JsonPathTracker | None = None


@dataclasses.dataclass
class _ThinkingAccumulator:
  """Accumulates streamed thinking content block data."""

  thinking: str
  signature: str


def _build_anthropic_thinking_param(
    config: Optional[types.GenerateContentConfig],
) -> Union[
    anthropic_types.ThinkingConfigEnabledParam,
    anthropic_types.ThinkingConfigDisabledParam,
    anthropic_types.ThinkingConfigAdaptiveParam,
    NotGiven,
]:
  """Maps genai ThinkingConfig to Anthropic's thinking parameter.

  Per ``google.genai.types.ThinkingConfig``, ``thinking_budget`` semantics are:
    * ``None``: not specified; the genai default is model-dependent. Anthropic
      requires an explicit choice whenever thinking is configured, so we
      surface this as a ``ValueError`` to keep the developer's intent
      explicit (mirroring the Anthropic API).
    * ``0``: thinking is DISABLED (``thinking.type: "disabled"``).
    * negative (e.g. ``-1`` AUTOMATIC): maps to Anthropic's adaptive thinking
      (``thinking.type: "adaptive"``, ``thinking.display: "summarized"``). The
      model picks the depth itself (controlled by the separate
      ``output_config.effort`` parameter when set) and returns its reasoning
      as summarized thoughts. REQUIRED for Claude Opus 4.7 and later models
      that reject ``"enabled"`` with a 400 error; also recommended for Opus
      4.6 and Sonnet 4.6 where ``"enabled"`` is deprecated.
    * positive int: budget in tokens for legacy manual mode
      (``thinking.type: "enabled"``; Anthropic requires ``>= 1024`` and
      ``< max_tokens``; validation is delegated to the Anthropic API so the
      caller gets the canonical error message). Rejected by Claude Opus 4.7
      -- callers targeting 4.7+ must use a negative value (adaptive) or
      ``0`` (disabled).

  Args:
    config: Optional GenerateContentConfig object.

  Returns:
    Mapped thinking parameter or NotGiven.
  """
  if not config or not config.thinking_config:
    return NOT_GIVEN

  thinking_budget = config.thinking_config.thinking_budget

  if thinking_budget is None:
    raise ValueError(
        "thinking_budget must be set explicitly when ThinkingConfig is"
        " provided for Anthropic models. Use 0 to disable thinking, -1 for"
        " adaptive (model-chosen depth), or a positive integer (>= 1024)"
        " for manual budgeting."
    )

  if thinking_budget == 0:
    return anthropic_types.ThinkingConfigDisabledParam(type="disabled")

  if thinking_budget < 0:
    # genai AUTOMATIC (-1) and any other negative value map to Anthropic
    # adaptive thinking. Required for Claude Opus 4.7 (which returns a 400
    # error for ``"enabled"``) and recommended for Opus 4.6 / Sonnet 4.6
    # where ``"enabled"`` is deprecated. Adaptive does not accept a budget;
    # depth is controlled by the model itself (or by the separate
    # ``output_config.effort`` parameter when set).
    # Without ``display``, Claude redacts the reasoning it just billed for.
    return anthropic_types.ThinkingConfigAdaptiveParam(
        type="adaptive",
        display="summarized",
    )

  return anthropic_types.ThinkingConfigEnabledParam(
      type="enabled",
      budget_tokens=thinking_budget,
  )


class AnthropicGenerateContentConfig(types.GenerateContentConfig):
  """Configuration options for Anthropic Claude content generation.

  This specialized configuration class is the recommended way to
  configure reasoning and extended thinking for newer Claude models.

  Attributes:
    effort: The reasoning effort level for adaptive extended thinking. Set
      directly to guide the reasoning depth ("low", "medium", "high", "xhigh",
      "max"). This is the preferred alternative to the deprecated manual
      `thinking_budget` on newer Claude models.
  """

  effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = Field(
      default=None,
      description=(
          "Configures the Claude-specific reasoning effort level for adaptive"
          " extended thinking. This is the recommended, future-proof way to"
          " control reasoning depth on newer Claude models."
      ),
  )

  @model_validator(mode="after")
  def validate_no_thinking_level(self) -> "AnthropicGenerateContentConfig":
    """Ensures thinking_level is not configured on Anthropic-specific config."""

    if self.thinking_config and self.thinking_config.thinking_level is not None:
      raise ValueError(
          "thinking_level is not supported in AnthropicGenerateContentConfig. "
          "Use the `effort` field directly to configure reasoning effort."
      )
    return self


def _build_effort_param(
    config: Optional[types.GenerateContentConfig],
) -> Optional[str]:
  """Extracts Anthropic's effort parameter from the configuration.

  To configure a specific reasoning effort level for Anthropic models,
  callers must use ``google.adk.models.AnthropicGenerateContentConfig`` and
  set the ``effort`` field directly.
  Using the standard ``thinking_config.thinking_level`` is explicitly
  unsupported because the standard `ThinkingLevel` enum (4 levels) cannot map
  consistently to Anthropic's 5 effort levels
  ("low", "medium", "high", "xhigh", "max").

  Any attempt to set `thinking_level` will not be passed to the model and will
  log a warning.

  If `effort` is not set, we return `None`.
  If `effort` and `thinking_level` are both set, `effort` takes precedence.

  Args:
    config: Optional GenerateContentConfig object.

  Returns:
    The effort level string (e.g., "xhigh") if specified via
    AnthropicGenerateContentConfig, or None.
  """
  if not config:
    return None

  if isinstance(config, AnthropicGenerateContentConfig) and config.effort:
    return config.effort

  # If effort is not set, but thinking_level is, log a warning and ignore it.
  if config.thinking_config and config.thinking_config.thinking_level:
    warnings.warn(
        "Standard thinking_config.thinking_level is not supported for Anthropic"
        " models and will be ignored. Use AnthropicGenerateContentConfig and"
        " set the `effort` field directly to configure reasoning effort.",
        category=UserWarning,
        stacklevel=4,
    )

  return None


class ClaudeRequest(BaseModel):
    system_instruction: str
    messages: Iterable[anthropic_types.MessageParam]
    tools: list[anthropic_types.ToolParam]


def to_claude_role(role: Optional[str]) -> Literal["user", "assistant"]:
    if role in ["model", "assistant"]:
        return "assistant"
    return "user"


def to_google_genai_finish_reason(
    anthropic_stop_reason: Optional[str],
) -> types.FinishReason:
    if anthropic_stop_reason in ["end_turn", "stop_sequence", "tool_use"]:
        return "STOP"
    if anthropic_stop_reason == "max_tokens":
        return "MAX_TOKENS"
    return "FINISH_REASON_UNSPECIFIED"


def _is_image_part(part: types.Part) -> bool:
    return (
        part.inline_data
        and part.inline_data.mime_type
        and part.inline_data.mime_type.startswith("image")
    )


def _is_pdf_part(part: types.Part) -> bool:
  inline_data = part.inline_data
  return bool(
      inline_data is not None
      and inline_data.mime_type is not None
      and inline_data.mime_type.split(";", 1)[0].strip() == "application/pdf"
  )


# The fields that mean a part carries something to send. The rest of a Part is
# annotation -- `part_metadata`, `video_metadata`, `media_resolution` and the
# like -- and ADK's own A2A converter sets some of them, so asking "is anything
# else set?" would leave the part in place and the session wedged.
_PART_CONTENT_FIELDS = (
    "audio_transcription",
    "code_execution_result",
    "executable_code",
    "file_data",
    "function_call",
    "function_response",
    "inline_data",
    "tool_call",
    "tool_response",
)


def _is_content_free_signature(part: types.Part) -> bool:
  """Whether a part is a thought signature with nothing to send beside it.

  A part that still has its `thought` flag is redacted thinking Claude issued
  itself, and the branches in `_part_to_message_block` handle it.
  """
  if not part.thought_signature or part.thought or part.text:
    return False
  return not any(getattr(part, f, None) for f in _PART_CONTENT_FIELDS)


def _normalize_image_media_type(mime_type: str) -> _ImageMediaType:
  normalized = mime_type.split(";", 1)[0].strip().lower()
  if normalized not in _ANTHROPIC_IMAGE_MEDIA_TYPES:
    raise ValueError(f"Unsupported Anthropic image MIME type: {mime_type}")
  return cast(_ImageMediaType, normalized)


def _function_response_media_blocks(
    function_response: types.FunctionResponse,
) -> list[_ToolResultContentBlockParam]:
  """Converts media a tool attached to its response into tool result blocks.

  Media Claude cannot carry in a tool result is dropped with a warning rather
  than raised on, because the tool that produced it is often third-party code
  the caller cannot change, and losing one image is better than losing the
  conversation.
  """
  blocks: list[_ToolResultContentBlockParam] = []
  for response_part in function_response.parts or []:
    blob = response_part.inline_data
    if blob is None or blob.data is None or not blob.mime_type:
      continue
    media_type = blob.mime_type.split(";", 1)[0].strip().lower()
    data = base64.b64encode(blob.data).decode()
    if media_type in _ANTHROPIC_IMAGE_MEDIA_TYPES:
      blocks.append(
          anthropic_types.ImageBlockParam(
              type="image",
              source=anthropic_types.Base64ImageSourceParam(
                  type="base64",
                  # Narrowed by the membership test above.
                  media_type=cast(_ImageMediaType, media_type),
                  data=data,
              ),
          )
      )
    elif media_type == "application/pdf":
      blocks.append(
          anthropic_types.DocumentBlockParam(
              type="document",
              source=anthropic_types.Base64PDFSourceParam(
                  type="base64",
                  media_type="application/pdf",
                  data=data,
              ),
          )
      )
    else:
      logger.warning(
          "Dropping tool result media of type %s, which Claude cannot receive"
          " in a tool result.",
          media_type,
      )
  return blocks


class _ToolUseIdSanitizer:
  """Maps invalid or empty tool_use IDs to deterministic unique fallbacks.

  Reuse one instance per conversation so a tool_use and its paired
  tool_result get matching unique outputs without collision across turns.
  """

  def __init__(self) -> None:
    self._mapping: dict[str, str] = {}
    self._unpaired_calls_by_name: dict[str, list[str]] = {}
    self._unpaired_calls_list: list[str] = []
    self._next_fallback: int = 0

  def sanitize(
      self,
      tool_id: str | None,
      tool_name: str | None = None,
      is_call: bool = False,
  ) -> str:
    if tool_id and re.fullmatch(r"[a-zA-Z0-9_-]+", tool_id):
      return tool_id
    if tool_id and tool_id in self._mapping:
      return self._mapping[tool_id]

    if is_call or (tool_id and tool_id not in self._mapping):
      assigned_id = f"toolu_fallback_{self._next_fallback}"
      self._next_fallback += 1
      if tool_id:
        self._mapping[tool_id] = assigned_id
      else:
        if tool_name:
          self._unpaired_calls_by_name.setdefault(tool_name, []).append(
              assigned_id
          )
        self._unpaired_calls_list.append(assigned_id)
      return assigned_id
    else:
      # Response with empty/None tool_id: pair with oldest pending call
      if tool_name and self._unpaired_calls_by_name.get(tool_name):
        assigned_id = self._unpaired_calls_by_name[tool_name].pop(0)
        if assigned_id in self._unpaired_calls_list:
          self._unpaired_calls_list.remove(assigned_id)
        return assigned_id
      elif self._unpaired_calls_list:
        assigned_id = self._unpaired_calls_list.pop(0)
        for ids in self._unpaired_calls_by_name.values():
          if assigned_id in ids:
            ids.remove(assigned_id)
            break
        return assigned_id
      else:
        assigned_id = f"toolu_fallback_{self._next_fallback}"
        self._next_fallback += 1
        return assigned_id


def _part_to_message_block(
    part: types.Part,
    sanitizer: _ToolUseIdSanitizer,
) -> _MessageBlockParam:
  if part.thought and part.text:
    signature = ""
    if part.thought_signature:
      signature = part.thought_signature.decode("utf-8")
    return anthropic_types.ThinkingBlockParam(
        type="thinking",
        thinking=part.text,
        signature=signature,
    )
  if part.thought and part.thought_signature:
    # Redacted thinking: no plaintext, only the encrypted blob produced by
    # content_block_to_part for round-tripping back to Claude.
    return anthropic_types.RedactedThinkingBlockParam(
        type="redacted_thinking",
        data=part.thought_signature.decode("utf-8"),
    )
  if part.text:
    return anthropic_types.TextBlockParam(text=part.text, type="text")
  elif part.function_call:
    function_call = part.function_call
    assert function_call.name
    tool_input: dict[str, object] = dict(function_call.args or {})

    return anthropic_types.ToolUseBlockParam(
        id=sanitizer.sanitize(
            function_call.id, tool_name=function_call.name, is_call=True
        ),
        name=function_call.name,
        input=tool_input,
        type="tool_use",
    )
  elif part.function_response:
    function_response = part.function_response
    content = ""
    response_data = function_response.response or {}

    if (
        "content" in response_data
        and isinstance(response_data["content"], list)
        and response_data["content"]
    ):
      content_items = []
      for item in response_data["content"]:
        if isinstance(item, dict):
          if item.get("type") == "text" and "text" in item:
            content_items.append(item["text"])
          else:
            content_items.append(str(item))
        else:
          content_items.append(str(item))
      content = "\n".join(content_items) if content_items else ""
    elif (
        "content" in response_data
        and isinstance(response_data["content"], str)
        and response_data["content"]
    ):
      content = response_data["content"]
    # We serialize to str here
    # SDK ref: anthropic.types.tool_result_block_param
    # https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/tool_result_block_param.py
    # Exactly {"result": value} is ADK's wrapper for a non-dict tool return.
    elif (
        response_data.keys() == {"result"}
        and response_data["result"] is not None
    ):
      result = response_data["result"]
      if isinstance(result, (dict, list)):
        content = json.dumps(result)
      else:
        content = str(result)
    elif response_data:
      # Fallback: serialize the entire response dict as JSON so that tools
      # returning arbitrary key structures (e.g. load_skill returning
      # {"skill_name", "instructions", "frontmatter"}) are not silently
      # dropped.
      content = json.dumps(response_data)

    # A tool can attach media alongside the serializable part of its result.
    # It travels in a dedicated field, so it has to be mapped over explicitly
    # or the model never sees it.
    media_blocks = _function_response_media_blocks(function_response)
    tool_result_content: Union[str, list[_ToolResultContentBlockParam]]
    if media_blocks:
      leading_text: list[_ToolResultContentBlockParam] = (
          [anthropic_types.TextBlockParam(type="text", text=content)]
          if content
          else []
      )
      tool_result_content = leading_text + media_blocks
    else:
      tool_result_content = content

    return anthropic_types.ToolResultBlockParam(
        tool_use_id=sanitizer.sanitize(
            function_response.id,
            tool_name=function_response.name,
            is_call=False,
        ),
        type="tool_result",
        content=tool_result_content,
        is_error=False,
    )
  elif _is_image_part(part):
    inline_data = part.inline_data
    if (
        inline_data is None
        or inline_data.data is None
        or inline_data.mime_type is None
    ):
      raise ValueError("Anthropic image parts require MIME type and data")
    data = base64.b64encode(inline_data.data).decode()
    image_source = anthropic_types.Base64ImageSourceParam(
        type="base64",
        media_type=_normalize_image_media_type(inline_data.mime_type),
        data=data,
    )
    return anthropic_types.ImageBlockParam(
        type="image",
        source=image_source,
    )
  elif _is_pdf_part(part):
    inline_data = part.inline_data
    if inline_data is None or inline_data.data is None:
      raise ValueError("Anthropic PDF parts require data")
    data = base64.b64encode(inline_data.data).decode()
    pdf_source = anthropic_types.Base64PDFSourceParam(
        type="base64",
        media_type="application/pdf",
        data=data,
    )
    return anthropic_types.DocumentBlockParam(
        type="document",
        source=pdf_source,
    )
  elif part.executable_code:
    return anthropic_types.TextBlockParam(
        type="text",
        text="Code:```python\n" + (part.executable_code.code or "") + "\n```",
    )
  elif part.code_execution_result:
    return anthropic_types.TextBlockParam(
        text="Execution Result:```code_output\n"
        + (part.code_execution_result.output or "")
        + "\n```",
        type="text",
    )

  raise NotImplementedError(f"Not supported yet: {part}")


def _content_to_message_param(
    content: types.Content,
    sanitizer: _ToolUseIdSanitizer,
) -> anthropic_types.MessageParam:
  message_block = []
  for part in content.parts or []:
    # Image data is not supported in Claude for assistant turns.
    if content.role != "user" and _is_image_part(part):
      logger.warning(
          "Image data is not supported in Claude for assistant turns."
      )
      continue

    # PDF data is not supported in Claude for assistant turns.
    if content.role != "user" and _is_pdf_part(part):
      logger.warning("PDF data is not supported in Claude for assistant turns.")
      continue

    # A signature with nothing to send beside it: there is no block to build
    # from it, and it used to raise and wedge the session for good.
    if _is_content_free_signature(part):
      logger.warning("Dropping a thought signature from another model.")
      continue

    message_block.append(_part_to_message_block(part, sanitizer))

  return {
      "role": to_claude_role(content.role),
      "content": message_block,
  }


def part_to_message_block(
    part: types.Part,
) -> Union[
    anthropic_types.TextBlockParam,
    anthropic_types.ImageBlockParam,
    anthropic_types.DocumentBlockParam,
    anthropic_types.ToolUseBlockParam,
    anthropic_types.ToolResultBlockParam,
]:
    if part.text:
        return anthropic_types.TextBlockParam(text=part.text, type="text")
    elif part.function_call:
        assert part.function_call.name

        return anthropic_types.ToolUseBlockParam(
            id=part.function_call.id or "",
            name=part.function_call.name,
            input=part.function_call.args,
            type="tool_use",
        )
    elif part.function_response:
        content = ""
        response_data = part.function_response.response

        # Handle response with content array
        if "content" in response_data and response_data["content"]:
            content_items = []
            for item in response_data["content"]:
                if isinstance(item, dict):
                    # Handle text content blocks
                    if item.get("type") == "text" and "text" in item:
                        content_items.append(item["text"])
                    else:
                        # Handle other structured content
                        content_items.append(str(item))
                else:
                    content_items.append(str(item))
            content = "\n".join(content_items) if content_items else ""
        # We serialize to str here
        # SDK ref: anthropic.types.tool_result_block_param
        # https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/tool_result_block_param.py
        elif "result" in response_data and response_data["result"] is not None:
            result = response_data["result"]
            if isinstance(result, (dict, list)):
                content = json.dumps(result)
            else:
                content = str(result)
        elif response_data:
            # Fallback: serialize the entire response dict as JSON so that tools
            # returning arbitrary key structures (e.g. load_skill returning
            # {"skill_name", "instructions", "frontmatter"}) are not silently
            # dropped.
            content = json.dumps(response_data)

        return anthropic_types.ToolResultBlockParam(
            tool_use_id=part.function_response.id or "",
            type="tool_result",
            content=content,
            is_error=False,
        )
    elif _is_image_part(part):
        data = base64.b64encode(part.inline_data.data).decode()
        return anthropic_types.ImageBlockParam(
            type="image",
            source=dict(
                type="base64", media_type=part.inline_data.mime_type, data=data
            ),
        )
    elif _is_pdf_part(part):
        data = base64.b64encode(part.inline_data.data).decode()
        return anthropic_types.DocumentBlockParam(
            type="document",
            source=dict(
                type="base64", media_type=part.inline_data.mime_type, data=data
            ),
        )
    elif part.executable_code:
        return anthropic_types.TextBlockParam(
            type="text",
            text="Code:```python\n" + part.executable_code.code + "\n```",
        )
    elif part.code_execution_result:
        return anthropic_types.TextBlockParam(
            text="Execution Result:```code_output\n"
            + part.code_execution_result.output
            + "\n```",
            type="text",
        )

    raise NotImplementedError(f"Not supported yet: {part}")


def content_to_message_param(
    content: types.Content,
) -> anthropic_types.MessageParam:
    message_block = []
    for part in content.parts or []:
        # Image data is not supported in Claude for assistant turns.
        if content.role != "user" and _is_image_part(part):
            logger.warning(
                "Image data is not supported in Claude for assistant turns."
            )
            continue

        # PDF data is not supported in Claude for assistant turns.
        if content.role != "user" and _is_pdf_part(part):
            logger.warning(
                "PDF data is not supported in Claude for assistant turns."
            )
            continue

        message_block.append(part_to_message_block(part))

    return {
        "role": to_claude_role(content.role),
        "content": message_block,
    }


def content_block_to_part(
    content_block: anthropic_types.ContentBlock,
) -> types.Part:
    if isinstance(content_block, anthropic_types.TextBlock):
        return types.Part.from_text(text=content_block.text)
    if isinstance(content_block, anthropic_types.ToolUseBlock):
        assert isinstance(content_block.input, dict)
        part = types.Part.from_function_call(
            name=content_block.name, args=content_block.input
        )
        part.function_call.id = content_block.id
        return part
    raise NotImplementedError("Not supported yet.")


def message_to_generate_content_response(
    message: anthropic_types.Message,
) -> LlmResponse:
    logger.info("Received response from Claude.")
    logger.debug(
        "Claude response: %s",
        message.model_dump_json(indent=2, exclude_none=True),
    )

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[content_block_to_part(cb) for cb in message.content],
        ),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=message.usage.input_tokens,
            candidates_token_count=message.usage.output_tokens,
            total_token_count=(
                message.usage.input_tokens + message.usage.output_tokens
            ),
        ),
        # TODO: Deal with these later.
        # finish_reason=to_google_genai_finish_reason(message.stop_reason),
    )


def _update_type_string(value: Any):
    """Lowercases nested JSON schema type strings for Anthropic compatibility."""
    if isinstance(value, list):
        for item in value:
            _update_type_string(item)
        return

    if not isinstance(value, dict):
        return

    schema_type = value.get("type")
    if isinstance(schema_type, str):
        value["type"] = schema_type.lower()

    for dict_key in (
        "$defs",
        "defs",
        "dependentSchemas",
        "patternProperties",
        "properties",
    ):
        child_dict = value.get(dict_key)
        if isinstance(child_dict, dict):
            for child_value in child_dict.values():
                _update_type_string(child_value)

    for single_key in (
        "additionalProperties",
        "additional_properties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedProperties",
    ):
        child_value = value.get(single_key)
        if isinstance(child_value, (dict, list)):
            _update_type_string(child_value)

    for list_key in (
        "allOf",
        "all_of",
        "anyOf",
        "any_of",
        "oneOf",
        "one_of",
        "prefixItems",
    ):
        child_list = value.get(list_key)
        if isinstance(child_list, list):
            _update_type_string(child_list)


def function_declaration_to_tool_param(
    function_declaration: types.FunctionDeclaration,
) -> anthropic_types.ToolParam:
    """Converts a function declaration to an Anthropic tool param."""
    assert function_declaration.name

    # Use parameters_json_schema if available, otherwise convert from parameters
    if function_declaration.parameters_json_schema:
        input_schema = copy.deepcopy(
            function_declaration.parameters_json_schema
        )
        _update_type_string(input_schema)
    else:
        properties = {}
        required_params = []
        if function_declaration.parameters:
            if function_declaration.parameters.properties:
                for (
                    key,
                    value,
                ) in function_declaration.parameters.properties.items():
                    properties[key] = value.model_dump(
                        by_alias=True, exclude_none=True
                    )
            if function_declaration.parameters.required:
                required_params = function_declaration.parameters.required

        input_schema = {
            "type": "object",
            "properties": properties,
        }
        if required_params:
            input_schema["required"] = required_params
        _update_type_string(input_schema)

    return anthropic_types.ToolParam(
        name=function_declaration.name,
        description=function_declaration.description or "",
        input_schema=input_schema,
    )


class AnthropicLlm(BaseLlm):
    """Integration with Claude models via the Anthropic API.

    Attributes:
      model: The name of the Claude model.
      max_tokens: The maximum number of tokens to generate.
    """

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        return [r"claude-3-.*", r"claude-.*-4.*"]

      elif event.type == "content_block_start":
        block = event.content_block
        if isinstance(block, anthropic_types.ThinkingBlock):
          thinking_blocks[event.index] = _ThinkingAccumulator(
              thinking=block.thinking,
              signature=block.signature,
          )
        elif isinstance(block, anthropic_types.RedactedThinkingBlock):
          # Redacted blocks arrive fully formed at start; no deltas follow.
          redacted_thinking_blocks[event.index] = block.data
        elif isinstance(block, anthropic_types.TextBlock):
          text_blocks[event.index] = block.text
        elif isinstance(block, anthropic_types.ToolUseBlock):
          tool_use_blocks[event.index] = _ToolUseAccumulator(
              id=block.id,
              name=block.name,
              args_json="",
          )
          yield LlmResponse(
              partial=True,
              content=types.Content(
                  role="model",
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id=block.id,
                              name=block.name,
                              will_continue=True,
                          )
                      )
                  ],
              ),
              model_version=llm_request.model or self.model,
          )

      elif event.type == "content_block_delta":
        delta = event.delta
        if isinstance(delta, anthropic_types.ThinkingDelta):
          thinking_blocks.setdefault(
              event.index,
              _ThinkingAccumulator(thinking="", signature=""),
          )
          thinking_blocks[event.index].thinking += delta.thinking
          yield LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[types.Part(text=delta.thinking, thought=True)],
              ),
              model_version=model_version,
              partial=True,
          )
        elif isinstance(delta, anthropic_types.SignatureDelta):
          # Claude streams the thinking block's cryptographic signature as a
          # separate delta near the end of the block. Accumulate it so the
          # aggregated thinking Part below carries ``thought_signature``.
          # Without it the reasoning block cannot round-trip back to Claude on
          # the next request -- extended thinking + tool use requires echoing
          # the signed thinking blocks, and re-serializing history for the
          # follow-up call would otherwise fail. Not surfaced as a partial (the
          # signature is opaque, not user-visible text).
          thinking_blocks.setdefault(
              event.index,
              _ThinkingAccumulator(thinking="", signature=""),
          )
          thinking_blocks[event.index].signature += delta.signature
        elif isinstance(delta, anthropic_types.TextDelta):
          text_blocks.setdefault(event.index, "")
          text_blocks[event.index] += delta.text
          yield LlmResponse(
              content=types.Content(
                  role="model",
                  parts=[types.Part.from_text(text=delta.text)],
              ),
              model_version=model_version,
              partial=True,
          )
        elif isinstance(delta, anthropic_types.InputJSONDelta):
          if event.index in tool_use_blocks:
            tool_use_blocks[event.index].args_json += delta.partial_json
            accumulator = tool_use_blocks[event.index]
            partial_args = None
            if delta.partial_json:
              if accumulator.tracker is None:
                accumulator.tracker = streaming_utils._JsonPathTracker()
              partial_args = accumulator.tracker.handle_chunk(
                  delta.partial_json
              )
            yield LlmResponse(
                partial=True,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id=accumulator.id,
                                name=accumulator.name,
                                partial_args=partial_args or None,
                                will_continue=True,
                            )
                        )
                    ],
                ),
                model_version=llm_request.model or self.model,
            )

      elif event.type == "message_delta":
        # ``message_delta`` carries the authoritative cumulative counts, so the
        # thinking detail is refreshed alongside the total it is nested in.
        output_tokens = event.usage.output_tokens
        thinking_tokens = _extract_thinking_token_count(event.usage)
        if event.delta and event.delta.stop_reason:
          stop_reason = event.delta.stop_reason

    # Build the final aggregated response with all content.
    all_parts: list[types.Part] = []
    all_indices = sorted(
        set(
            list(thinking_blocks.keys())
            + list(redacted_thinking_blocks.keys())
            + list(text_blocks.keys())
            + list(tool_use_blocks.keys())
        )
    )
    for idx in all_indices:
      if idx in thinking_blocks:
        thinking_acc = thinking_blocks[idx]
        part = types.Part(text=thinking_acc.thinking, thought=True)
        if thinking_acc.signature:
          part.thought_signature = thinking_acc.signature.encode("utf-8")
        all_parts.append(part)
      if idx in redacted_thinking_blocks:
        all_parts.append(
            types.Part(
                thought=True,
                thought_signature=redacted_thinking_blocks[idx].encode("utf-8"),
            )
            if match:
                return match.group(1)
        return model

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        model_to_use = self._resolve_model_name(llm_request.model)
        messages = [
            content_to_message_param(content)
            for content in llm_request.contents or []
        ]
        tools = NOT_GIVEN
        if (
            llm_request.config
            and llm_request.config.tools
            and llm_request.config.tools[0].function_declarations
        ):
            tools = [
                function_declaration_to_tool_param(tool)
                for tool in llm_request.config.tools[0].function_declarations
            ]
        tool_choice = (
            anthropic_types.ToolChoiceAutoParam(type="auto")
            if llm_request.tools_dict
            else NOT_GIVEN
        )

        if not stream:
            message = await self._anthropic_client.messages.create(
                model=model_to_use,
                system=llm_request.config.system_instruction,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=self.max_tokens,
            )
            yield message_to_generate_content_response(message)
        else:
            async for response in self._generate_content_streaming(
                llm_request, messages, tools, tool_choice
            ):
                yield response

    async def _generate_content_streaming(
        self,
        llm_request: LlmRequest,
        messages: list[anthropic_types.MessageParam],
        tools: Union[Iterable[anthropic_types.ToolUnionParam], NotGiven],
        tool_choice: Union[anthropic_types.ToolChoiceParam, NotGiven],
    ) -> AsyncGenerator[LlmResponse, None]:
        """Handles streaming responses from Anthropic models.

        Yields partial LlmResponse objects as content arrives, followed by
        a final aggregated LlmResponse with all content.
        """
        model_to_use = self._resolve_model_name(llm_request.model)
        raw_stream = await self._anthropic_client.messages.create(
            model=model_to_use,
            system=llm_request.config.system_instruction,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=self.max_tokens,
            stream=True,
        )

        # Track content blocks being built during streaming.
        # Each entry maps a block index to its accumulated state.
        text_blocks: dict[int, str] = {}
        tool_use_blocks: dict[int, _ToolUseAccumulator] = {}
        input_tokens = 0
        output_tokens = 0

        async for event in raw_stream:
            if event.type == "message_start":
                input_tokens = event.message.usage.input_tokens
                output_tokens = event.message.usage.output_tokens

            elif event.type == "content_block_start":
                block = event.content_block
                if isinstance(block, anthropic_types.TextBlock):
                    text_blocks[event.index] = block.text
                elif isinstance(block, anthropic_types.ToolUseBlock):
                    tool_use_blocks[event.index] = _ToolUseAccumulator(
                        id=block.id,
                        name=block.name,
                        args_json="",
                    )

            elif event.type == "content_block_delta":
                delta = event.delta
                if isinstance(delta, anthropic_types.TextDelta):
                    text_blocks.setdefault(event.index, "")
                    text_blocks[event.index] += delta.text
                    yield LlmResponse(
                        content=types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=delta.text)],
                        ),
                        partial=True,
                    )
                elif isinstance(delta, anthropic_types.InputJSONDelta):
                    if event.index in tool_use_blocks:
                        tool_use_blocks[
                            event.index
                        ].args_json += delta.partial_json

            elif event.type == "message_delta":
                output_tokens = event.usage.output_tokens

        # Build the final aggregated response with all content.
        all_parts: list[types.Part] = []
        all_indices = sorted(
            set(list(text_blocks.keys()) + list(tool_use_blocks.keys()))
        )
        for idx in all_indices:
            if idx in text_blocks:
                all_parts.append(types.Part.from_text(text=text_blocks[idx]))
            if idx in tool_use_blocks:
                acc = tool_use_blocks[idx]
                args = json.loads(acc.args_json) if acc.args_json else {}
                part = types.Part.from_function_call(name=acc.name, args=args)
                part.function_call.id = acc.id
                all_parts.append(part)

        yield LlmResponse(
            content=types.Content(role="model", parts=all_parts),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=input_tokens,
                candidates_token_count=output_tokens,
                total_token_count=input_tokens + output_tokens,
            ),
            partial=False,
        )

    @cached_property
    def _anthropic_client(self) -> AsyncAnthropic:
        return AsyncAnthropic()


class Claude(AnthropicLlm):
    """Integration with Claude models served from Vertex AI.

    Attributes:
      model: The name of the Claude model.
      max_tokens: The maximum number of tokens to generate.
    """

    model: str = "claude-3-5-sonnet-v2@20241022"

    @cached_property
    @override
    def _anthropic_client(self) -> AsyncAnthropicVertex:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")

        if self.model.startswith("projects/"):
            match = re.search(
                r"projects/([^/]+)/locations/([^/]+)/",
                self.model,
            )
            if match:
                project_id = match.group(1)
                location = match.group(2)

        if not project_id or not location:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set for using"
                " Anthropic on Vertex."
            )

        return AsyncAnthropicVertex(
            project_id=project_id,
            region=location,
            default_headers=get_tracking_headers(),
        )
