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

import asyncio
import json
import logging
from pathlib import Path
import signal
import tempfile
from typing import Any
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.adk.cli import api_server as api_server_module
from google.adk.cli import fast_api as fast_api_module
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.errors.input_validation_error import InputValidationError
from google.adk.errors.session_not_found_error import SessionNotFoundError
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_result import EvalSetResult
from google.adk.evaluation.in_memory_eval_sets_manager import (
    InMemoryEvalSetsManager,
)
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.api_core.exceptions import GoogleAPICallError
from google.api_core.exceptions import InvalidArgument
from google.genai import types
from pydantic import BaseModel
import pytest
from starlette.applications import Starlette
from starlette.routing import Mount

# Configure logging to help diagnose server startup issues
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("google_adk." + __name__)


# Here we create a dummy agent module that get_fast_api_app expects
class DummyAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name=name)
        self.sub_agents = []


root_agent = DummyAgent(name="dummy_agent")


# Create sample events that our mocked runner will return
def _event_1():
    return Event(
        author="dummy agent",
        invocation_id="invocation_id",
        content=types.Content(
            role="model",
            parts=[types.Part(text="LLM reply", inline_data=None)],
        ),
    )


def _event_2():
    return Event(
        author="dummy agent",
        invocation_id="invocation_id",
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    text=None,
                    inline_data=types.Blob(
                        mime_type="audio/pcm;rate=24000", data=b"\x00\xff"
                    ),
                )
            ],
        ),
    )


def _event_3():
    return Event(
        author="dummy agent", invocation_id="invocation_id", interrupted=True
    )


def _event_state_delta(state_delta: dict[str, Any]):
    return Event(
        author="dummy agent",
        invocation_id="invocation_id",
        actions=EventActions(state_delta=state_delta),
    )


# Define mocked async generator functions for the Runner
async def dummy_run_live(self, session, live_request_queue):
    yield _event_1()
    await asyncio.sleep(0)

    yield _event_2()
    await asyncio.sleep(0)

    yield _event_3()


async def dummy_run_async(
    self,
    user_id,
    session_id,
    new_message,
    state_delta=None,
    run_config: Optional[RunConfig] = None,
    invocation_id: Optional[str] = None,
):
    run_config = run_config or RunConfig()
    yield _event_1()
    await asyncio.sleep(0)

    yield _event_2()
    await asyncio.sleep(0)

    yield _event_3()
    await asyncio.sleep(0)

    if state_delta is not None:
        yield _event_state_delta(state_delta)


# Define a local mock for EvalCaseResult specific to fast_api tests
class _MockEvalCaseResult(BaseModel):
    eval_set_id: str
    eval_id: str
    final_eval_status: Any
    user_id: str
    session_id: str
    eval_set_file: str
    eval_metric_results: list = {}
    overall_eval_metric_results: list = ({},)
    eval_metric_result_per_invocation: list = {}


#################################################
# Test Fixtures
#################################################


@pytest.fixture(autouse=True)
def patch_runner(monkeypatch):
    """Patch the Runner methods to use our dummy implementations."""
    monkeypatch.setattr(Runner, "run_live", dummy_run_live)
    monkeypatch.setattr(Runner, "run_async", dummy_run_async)


@pytest.fixture
def test_session_info():
    """Return test user and session IDs for testing."""
    return {
        "app_name": "test_app",
        "user_id": "test_user",
        "session_id": "test_session",
    }


@pytest.fixture
def mock_agent_loader():

    class MockAgentLoader:
        def __init__(self, agents_dir: str):
            pass

        def load_agent(self, app_name):
            return root_agent

        def list_agents(self):
            return ["test_app"]

        def list_agents_detailed(self):
            return [
                {
                    "name": "test_app",
                    "root_agent_name": "test_agent",
                    "description": "A test agent for unit testing",
                    "language": "python",
                    "is_computer_use": False,
                }
            ]

    return MockAgentLoader(".")


@pytest.fixture
def mock_session_service():
    """Create an in-memory session service instance for testing."""
    return InMemorySessionService()


@pytest.fixture
def mock_artifact_service():
    """Create a mock artifact service."""

    artifacts: dict[str, list[dict[str, Any]]] = {}

    def _artifact_key(
        app_name: str, user_id: str, session_id: Optional[str], filename: str
    ) -> str:
        if session_id is None:
            return f"{app_name}:{user_id}:user:{filename}"
        return f"{app_name}:{user_id}:{session_id}:{filename}"

    def _canonical_uri(
        app_name: str,
        user_id: str,
        session_id: Optional[str],
        filename: str,
        version: int,
    ) -> str:
        if session_id is None:
            return (
                f"artifact://apps/{app_name}/users/{user_id}/artifacts/"
                f"{filename}/versions/{version}"
            )
        return (
            f"artifact://apps/{app_name}/users/{user_id}/sessions/{session_id}/"
            f"artifacts/{filename}/versions/{version}"
        )

    class MockArtifactService:
        def __init__(self):
            self._artifacts = artifacts
            self.save_artifact_side_effect: Optional[BaseException] = None

        async def save_artifact(
            self,
            *,
            app_name: str,
            user_id: str,
            filename: str,
            artifact: types.Part,
            session_id: Optional[str] = None,
            custom_metadata: Optional[dict[str, Any]] = None,
        ) -> int:
            if self.save_artifact_side_effect is not None:
                effect = self.save_artifact_side_effect
                if isinstance(effect, BaseException):
                    raise effect
                raise TypeError(
                    "save_artifact_side_effect must be an exception instance."
                )
            key = _artifact_key(app_name, user_id, session_id, filename)
            entries = artifacts.setdefault(key, [])
            version = len(entries)
            artifact_version = ArtifactVersion(
                version=version,
                canonical_uri=_canonical_uri(
                    app_name, user_id, session_id, filename, version
                ),
                custom_metadata=custom_metadata or {},
            )
            if artifact.inline_data is not None:
                artifact_version.mime_type = artifact.inline_data.mime_type
            elif artifact.text is not None:
                artifact_version.mime_type = "text/plain"
            elif artifact.file_data is not None:
                artifact_version.mime_type = artifact.file_data.mime_type

            entries.append(
                {
                    "version": version,
                    "artifact": artifact,
                    "metadata": artifact_version,
                }
            )
            return version

        def add_artifact(
            self,
            *,
            app_name: str,
            user_id: str,
            session_id: str,
            filename: str,
            artifact: types.Part,
            custom_metadata: Optional[dict[str, Any]] = None,
            canonical_uri: Optional[str] = None,
            mime_type: Optional[str] = None,
        ) -> int:
            """Synchronous helper for tests to add artifacts."""
            key = _artifact_key(app_name, user_id, session_id, filename)
            entries = artifacts.setdefault(key, [])
            version = len(entries)
            artifact_version = ArtifactVersion(
                version=version,
                canonical_uri=(
                    canonical_uri
                    or _canonical_uri(
                        app_name, user_id, session_id, filename, version
                    )
                ),
                custom_metadata=custom_metadata or {},
            )
            if mime_type:
                artifact_version.mime_type = mime_type
            elif artifact.inline_data is not None:
                artifact_version.mime_type = artifact.inline_data.mime_type
            elif artifact.text is not None:
                artifact_version.mime_type = "text/plain"
            elif artifact.file_data is not None:
                artifact_version.mime_type = artifact.file_data.mime_type

            entries.append(
                {
                    "version": version,
                    "artifact": artifact,
                    "metadata": artifact_version,
                }
            )
            return version

        async def load_artifact(
            self, app_name, user_id, session_id, filename, version=None
        ):
            """Load an artifact by filename."""
            key = _artifact_key(app_name, user_id, session_id, filename)
            if key not in artifacts:
                return None

            if version is not None:
                for entry in artifacts[key]:
                    if entry["version"] == version:
                        return entry["artifact"]
                return None

            return artifacts[key][-1]["artifact"]

        async def list_artifact_keys(self, app_name, user_id, session_id):
            """List artifact names for a session."""
            prefix = f"{app_name}:{user_id}:{session_id}:"
            return [
                key.split(":")[-1]
                for key in artifacts.keys()
                if key.startswith(prefix)
            ]

        async def list_versions(self, app_name, user_id, session_id, filename):
            """List versions of an artifact."""
            key = _artifact_key(app_name, user_id, session_id, filename)
            if key not in artifacts:
                return []
            return [entry["version"] for entry in artifacts[key]]

        async def list_artifact_versions(
            self, app_name, user_id, session_id, filename
        ):
            """List all artifact versions with metadata."""
            key = _artifact_key(app_name, user_id, session_id, filename)
            if key not in artifacts:
                return []
            return [entry["metadata"] for entry in artifacts[key]]

        async def delete_artifact(
            self, app_name, user_id, session_id, filename
        ):
            """Delete an artifact."""
            key = _artifact_key(app_name, user_id, session_id, filename)
            artifacts.pop(key, None)

        async def get_artifact_version(
            self,
            *,
            app_name: str,
            user_id: str,
            filename: str,
            session_id: Optional[str] = None,
            version: Optional[int] = None,
        ) -> Optional[ArtifactVersion]:
            key = _artifact_key(app_name, user_id, session_id, filename)
            entries = artifacts.get(key)
            if not entries:
                return None
            if version is None:
                return entries[-1]["metadata"]
            for entry in entries:
                if entry["version"] == version:
                    return entry["metadata"]
            return None

    return MockArtifactService()


@pytest.fixture
def mock_memory_service():
    """Create a mock memory service."""
    return AsyncMock()


@pytest.fixture
def mock_eval_sets_manager():
    """Create a mock eval sets manager."""
    return InMemoryEvalSetsManager()


@pytest.fixture
def mock_eval_set_results_manager():
    """Create a mock local eval set results manager."""

    # Storage for eval set results.
    eval_set_results = {}

    class MockEvalSetResultsManager:
        """Mock eval set results manager."""

        def save_eval_set_result(
            self, app_name, eval_set_id, eval_case_results
        ):
            if app_name not in eval_set_results:
                eval_set_results[app_name] = {}
            eval_set_result_id = f"{app_name}_{eval_set_id}_eval_result"
            eval_set_result = EvalSetResult(
                eval_set_result_id=eval_set_result_id,
                eval_set_result_name=eval_set_result_id,
                eval_set_id=eval_set_id,
                eval_case_results=eval_case_results,
            )
            if eval_set_result_id not in eval_set_results[app_name]:
                eval_set_results[app_name][
                    eval_set_result_id
                ] = eval_set_result
            else:
                eval_set_results[app_name][eval_set_result_id].append(
                    eval_set_result
                )

        def get_eval_set_result(self, app_name, eval_set_result_id):
            if app_name not in eval_set_results:
                raise ValueError(f"App {app_name} not found.")
            if eval_set_result_id not in eval_set_results[app_name]:
                raise ValueError(
                    f"Eval set result {eval_set_result_id} not found in app {app_name}."
                )
            return eval_set_results[app_name][eval_set_result_id]

        def list_eval_set_results(self, app_name):
            """List eval set results."""
            if app_name not in eval_set_results:
                raise ValueError(f"App {app_name} not found.")
            return list(eval_set_results[app_name].keys())

    return MockEvalSetResultsManager()


def _create_test_client(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    **app_kwargs,
):
    """Helper to create a TestClient with the given get_fast_api_app overrides."""
    defaults = dict(
        agents_dir=".",
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=["*"],
        a2a=False,
        host="127.0.0.1",
        port=8000,
    )
    defaults.update(app_kwargs)
    with (
        patch.object(signal, "signal", autospec=True, return_value=None),
        patch.object(
            fast_api_module,
            "create_session_service_from_options",
            autospec=True,
            return_value=mock_session_service,
        ),
        patch.object(
            fast_api_module,
            "create_artifact_service_from_options",
            autospec=True,
            return_value=mock_artifact_service,
        ),
        patch.object(
            fast_api_module,
            "create_memory_service_from_options",
            autospec=True,
            return_value=mock_memory_service,
        ),
        patch.object(
            fast_api_module,
            "AgentLoader",
            autospec=True,
            return_value=mock_agent_loader,
        ),
        patch.object(
            fast_api_module,
            "LocalEvalSetsManager",
            autospec=True,
            return_value=mock_eval_sets_manager,
        ),
        patch.object(
            fast_api_module,
            "LocalEvalSetResultsManager",
            autospec=True,
            return_value=mock_eval_set_results_manager,
        ),
    ):
        app = get_fast_api_app(**defaults)
        return TestClient(app)


@pytest.fixture
def test_app(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
    """Create a TestClient for the FastAPI app without starting a server."""
    return _create_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        mock_eval_sets_manager,
        mock_eval_set_results_manager,
    )


@pytest.fixture
def builder_test_client(
    tmp_path,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
    """Return a TestClient rooted in a temporary agents directory."""
    with (
        patch.object(signal, "signal", autospec=True, return_value=None),
        patch.object(
            fast_api_module,
            "create_session_service_from_options",
            autospec=True,
            return_value=mock_session_service,
        ),
        patch.object(
            fast_api_module,
            "create_artifact_service_from_options",
            autospec=True,
            return_value=mock_artifact_service,
        ),
        patch.object(
            fast_api_module,
            "create_memory_service_from_options",
            autospec=True,
            return_value=mock_memory_service,
        ),
        patch.object(
            fast_api_module,
            "AgentLoader",
            autospec=True,
            return_value=mock_agent_loader,
        ),
        patch.object(
            fast_api_module,
            "LocalEvalSetsManager",
            autospec=True,
            return_value=mock_eval_sets_manager,
        ),
        patch.object(
            fast_api_module,
            "LocalEvalSetResultsManager",
            autospec=True,
            return_value=mock_eval_set_results_manager,
        ),
    ):
        app = get_fast_api_app(
            agents_dir=str(tmp_path),
            web=True,
            session_service_uri="",
            artifact_service_uri="",
            memory_service_uri="",
            allow_origins=None,
            a2a=False,
            host="127.0.0.1",
            port=8000,
        )
        return TestClient(app)


@pytest.fixture
async def create_test_session(
    test_app, test_session_info, mock_session_service
):
    """Create a test session using the mocked session service."""

    # Create the session directly through the mock service
    session = await mock_session_service.create_session(
        app_name=test_session_info["app_name"],
        user_id=test_session_info["user_id"],
        session_id=test_session_info["session_id"],
        state={},
    )

    logger.info(f"Created test session: {session.id}")
    return test_session_info


@pytest.fixture
async def create_test_eval_set(
    test_app, test_session_info, mock_eval_sets_manager
):
    """Create a test eval set using the mocked eval sets manager."""
    _ = mock_eval_sets_manager.create_eval_set(
        app_name=test_session_info["app_name"],
        eval_set_id="test_eval_set_id",
    )
    test_eval_case = EvalCase(
        eval_id="test_eval_case_id",
        conversation=[
            Invocation(
                invocation_id="test_invocation_id",
                user_content=types.Content(
                    parts=[types.Part(text="test_user_content")],
                    role="user",
                ),
            )
        ],
    )
    _ = mock_eval_sets_manager.add_eval_case(
        app_name=test_session_info["app_name"],
        eval_set_id="test_eval_set_id",
        eval_case=test_eval_case,
    )
    return test_session_info


@pytest.fixture
def temp_agents_dir_with_a2a():
    """Create a temporary agents directory with A2A agent configurations for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test agent directory
        agent_dir = Path(temp_dir) / "test_a2a_agent"
        agent_dir.mkdir()

        # Create agent.json file
        agent_card = {
            "name": "test_a2a_agent",
            "description": "Test A2A agent",
            "version": "1.0.0",
            "author": "test",
            "capabilities": ["text"],
        }

        with open(agent_dir / "agent.json", "w") as f:
            json.dump(agent_card, f)

        # Create a simple agent.py file
        agent_py_content = """
from google.adk.agents.base_agent import BaseAgent

class TestA2AAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="test_a2a_agent")
"""

        with open(agent_dir / "agent.py", "w") as f:
            f.write(agent_py_content)

        yield temp_dir


@pytest.fixture
def test_app_with_a2a(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    temp_agents_dir_with_a2a,
    monkeypatch,
):
    """Create a TestClient for the FastAPI app with A2A enabled."""
    # Mock A2A related classes
    with (
        patch("signal.signal", return_value=None),
        patch(
            "google.adk.cli.fast_api.create_session_service_from_options",
            return_value=mock_session_service,
        ),
        patch(
            "google.adk.cli.fast_api.create_artifact_service_from_options",
            return_value=mock_artifact_service,
        ),
        patch(
            "google.adk.cli.fast_api.create_memory_service_from_options",
            return_value=mock_memory_service,
        ),
        patch(
            "google.adk.cli.fast_api.AgentLoader",
            return_value=mock_agent_loader,
        ),
        patch(
            "google.adk.cli.fast_api.LocalEvalSetsManager",
            return_value=mock_eval_sets_manager,
        ),
        patch(
            "google.adk.cli.fast_api.LocalEvalSetResultsManager",
            return_value=mock_eval_set_results_manager,
        ),
        patch("a2a.server.tasks.InMemoryTaskStore") as mock_task_store,
        patch(
            "google.adk.a2a.executor.a2a_agent_executor.A2aAgentExecutor"
        ) as mock_executor,
        patch(
            "a2a.server.request_handlers.DefaultRequestHandler"
        ) as mock_handler,
        patch("a2a.server.apps.A2AStarletteApplication") as mock_a2a_app,
    ):
        # Configure mocks
        mock_task_store.return_value = MagicMock()
        mock_executor.return_value = MagicMock()
        mock_handler.return_value = MagicMock()

        # Mock A2AStarletteApplication
        mock_app_instance = MagicMock()
        mock_app_instance.routes.return_value = (
            []
        )  # Return empty routes for testing
        mock_a2a_app.return_value = mock_app_instance

        # Change to temp directory
        monkeypatch.chdir(temp_agents_dir_with_a2a)

        app = get_fast_api_app(
            agents_dir=".",
            web=True,
            session_service_uri="",
            artifact_service_uri="",
            memory_service_uri="",
            allow_origins=["*"],
            a2a=True,
            host="127.0.0.1",
            port=8000,
        )

        client = TestClient(app)
        yield client


#################################################
# Test Cases
#################################################


def test_list_apps(test_app):
    """Test listing available applications."""
    # Use the TestClient to make a request
    response = test_app.get("/list-apps")

    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    logger.info(f"Listed apps: {data}")


def test_list_apps_detailed(test_app):
    """Test listing available applications with detailed metadata."""
    response = test_app.get("/list-apps?detailed=true")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "apps" in data
    assert isinstance(data["apps"], list)

    for app in data["apps"]:
        assert "name" in app
        assert "rootAgentName" in app
        assert "description" in app
        assert "language" in app
        assert app["language"] in ["yaml", "python"]
        assert "isComputerUse" in app
        assert not app["isComputerUse"]

    logger.info(f"Listed apps: {data}")


def test_create_session_with_id(test_app, test_session_info):
    """Test creating a session with a specific ID."""
    new_session_id = "new_session_id"
    url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions/{new_session_id}"
    response = test_app.post(url, json={"state": {}})

    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == new_session_id
    assert data["appName"] == test_session_info["app_name"]
    assert data["userId"] == test_session_info["user_id"]
    logger.info(f"Created session with ID: {data['id']}")


def test_create_session_with_id_already_exists(test_app, test_session_info):
    """Test creating a session with an ID that already exists."""
    session_id = "existing_session_id"
    url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions/{session_id}"

    # Create the session for the first time
    response = test_app.post(url, json={"state": {}})
    assert response.status_code == 200

    # Attempt to create it again
    response = test_app.post(url, json={"state": {}})
    assert response.status_code == 409
    assert "Session already exists" in response.json()["detail"]
    logger.info("Verified 409 on duplicate session creation.")


def test_create_session_without_id(test_app, test_session_info):
    """Test creating a session with a generated ID."""
    url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
    response = test_app.post(url, json={"state": {}})

    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["appName"] == test_session_info["app_name"]
    assert data["userId"] == test_session_info["user_id"]
    logger.info(f"Created session with generated ID: {data['id']}")


class _OptionsRecordingSessionService(InMemorySessionService):
  """In-memory session service that accepts and records arbitrary kwargs."""

  def __init__(self):
    super().__init__()
    self.recorded_kwargs: dict[str, Any] = {}

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
      **kwargs: Any,
  ) -> Session:
    self.recorded_kwargs = dict(kwargs)
    return await super().create_session(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )


class _ExplicitOptionsSessionService(InMemorySessionService):
  """In-memory session service that explicitly accepts custom parameters."""

  def __init__(self):
    super().__init__()
    self.recorded_kwargs: dict[str, Any] = {}

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
      custom_option: Optional[str] = None,
  ) -> Session:
    self.recorded_kwargs = {"custom_option": custom_option}
    return await super().create_session(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )


class _ValidatingOptionsSessionService(InMemorySessionService):
  """In-memory session service that validates options and raises ValueError."""

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
      **kwargs: Any,
  ) -> Session:
    if kwargs.get("ttl") is not None and kwargs.get("expire_time") is not None:
      raise ValueError(
          "Cannot specify both 'ttl' and 'expire_time' simultaneously."
      )
    return await super().create_session(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )


@pytest.fixture
def explicit_options_session_service():
  """Create a session service with explicit custom parameters."""
  return _ExplicitOptionsSessionService()


@pytest.fixture
def explicit_options_test_app(
    explicit_options_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Create a TestClient backed by an explicit options session service."""
  return _create_test_client(
      explicit_options_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )


@pytest.fixture
def options_session_service():
  """Create a session service whose create_session accepts kwargs."""
  return _OptionsRecordingSessionService()


@pytest.fixture
def options_test_app(
    options_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Create a TestClient backed by a kwargs-capable session service."""
  return _create_test_client(
      options_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )


@pytest.fixture
def validating_options_session_service():
  """Create a session service that validates kwargs."""
  return _ValidatingOptionsSessionService()


@pytest.fixture
def validating_options_test_app(
    validating_options_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Create a TestClient backed by a validating session service."""
  return _create_test_client(
      validating_options_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )


def test_create_session_forwards_options(
    options_test_app, options_session_service, test_session_info
):
  """Test that options dict is forwarded to a supporting session service."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = options_test_app.post(
      url,
      json={
          "options": {
              "ttl": "7200s",
              "expire_time": "2026-08-01T00:00:00Z",
              "custom_param": "foo",
          }
      },
  )

  assert response.status_code == 200
  assert options_session_service.recorded_kwargs == {
      "ttl": "7200s",
      "expire_time": "2026-08-01T00:00:00Z",
      "custom_param": "foo",
  }


def test_create_session_explicit_options_forwards_kwargs(
    explicit_options_test_app,
    explicit_options_session_service,
    test_session_info,
):
  """Test that options are forwarded to service with explicit parameter."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = explicit_options_test_app.post(
      url, json={"options": {"custom_option": "bar"}}
  )

  assert response.status_code == 200
  assert (
      explicit_options_session_service.recorded_kwargs.get("custom_option")
      == "bar"
  )


def test_create_session_options_with_unsupported_service(
    test_app, test_session_info
):
  """Test 400 when options are requested but the service cannot honor them."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = test_app.post(url, json={"options": {"ttl": "7200s"}})

  assert response.status_code == 400
  assert "not supported" in response.json()["detail"]


def test_create_session_options_validation_error_returns_400(
    validating_options_test_app, test_session_info
):
  """Test 400 when session service raises ValueError on invalid options."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = validating_options_test_app.post(
      url,
      json={
          "options": {
              "ttl": "7200s",
              "expire_time": "2026-08-01T00:00:00Z",
          }
      },
  )

  assert response.status_code == 400
  assert (
      "Cannot specify both 'ttl' and 'expire_time'" in response.json()["detail"]
  )


def test_create_session_options_conflicting_key_returns_400(
    test_app, test_session_info
):
  """Test 400 when options contains a key already bound by the endpoint."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = test_app.post(url, json={"options": {"app_name": "other_app"}})

  assert response.status_code == 400


def test_accepts_kwargs_rejects_var_positional_parameter():
  """_accepts_kwargs should return False for variadic positional parameters."""
  from google.adk.cli.api_server import _accepts_kwargs

  def func(*args: Any) -> None:
    pass

  assert not _accepts_kwargs(func, {"args": "value"})


def test_get_session(test_app, create_test_session):
    """Test retrieving a session by ID."""
    info = create_test_session
    url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/{info['session_id']}"
    response = test_app.get(url)

    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == info["session_id"]
    assert data["appName"] == info["app_name"]
    assert data["userId"] == info["user_id"]
    logger.info(f"Retrieved session: {data['id']}")


def test_list_sessions(test_app, create_test_session):
    """Test listing all sessions for a user."""
    info = create_test_session
    url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions"
    response = test_app.get(url)

    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # At least our test session should be present
    assert any(session["id"] == info["session_id"] for session in data)
    logger.info(f"Listed {len(data)} sessions")


def test_delete_session(test_app, create_test_session):
    """Test deleting a session."""
    info = create_test_session
    url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/{info['session_id']}"
    response = test_app.delete(url)

    # Verify the response
    assert response.status_code == 200

    # Verify the session is deleted
    response = test_app.get(url)
    assert response.status_code == 404
    logger.info("Session deleted successfully")


def test_update_session(test_app, create_test_session):
    """Test patching a session state."""
    info = create_test_session
    url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/{info['session_id']}"

    # Get the original session
    response = test_app.get(url)
    assert response.status_code == 200
    original_session = response.json()
    original_state = original_session.get("state", {})

    # Prepare state delta
    state_delta = {"test_key": "test_value", "counter": 42}

    # Patch the session
    response = test_app.patch(url, json={"state_delta": state_delta})
    assert response.status_code == 200

    # Verify the response
    patched_session = response.json()
    assert patched_session["id"] == info["session_id"]

    # Verify state was updated correctly
    expected_state = {**original_state, **state_delta}
    assert patched_session["state"] == expected_state

    # Verify the session was actually updated in storage
    response = test_app.get(url)
    assert response.status_code == 200
    retrieved_session = response.json()
    assert retrieved_session["state"] == expected_state

    # Verify an event was created for the state change
    events = retrieved_session.get("events", [])
    assert len(events) > len(original_session.get("events", []))

    # Find the state patch event (looking for "p-" prefix pattern)
    state_patch_events = [
        event
        for event in events
        if event.get("invocationId", "").startswith("p-")
    ]

    assert len(state_patch_events) == 1, (
        f"Expected 1 state_patch event, found {len(state_patch_events)}. Events:"
        f" {events}"
    )
    state_patch_event = state_patch_events[0]
    assert state_patch_event["author"] == "user"

    # Check for actions in both camelCase and snake_case
    actions = state_patch_event.get("actions")
    assert (
        actions is not None
    ), f"No actions found in event: {state_patch_event}"
    state_delta_in_event = actions.get("stateDelta")
    assert state_delta_in_event == state_delta

    logger.info("Session state patched successfully")


def test_patch_session_not_found(test_app, test_session_info):
    """Test patching a nonexistent session."""
    info = test_session_info
    url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/nonexistent"

    state_delta = {"test_key": "test_value"}
    response = test_app.patch(url, json={"state_delta": state_delta})

    assert response.status_code == 404
    assert "Session not found" in response.json()["detail"]
    logger.info("Patch session not found test passed")


def test_agent_run(test_app, create_test_session):
    """Test running an agent with a message."""
    info = create_test_session
    url = "/run"
    payload = {
        "app_name": info["app_name"],
        "user_id": info["user_id"],
        "session_id": info["session_id"],
        "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
        "streaming": False,
    }

    response = test_app.post(url, json=payload)

    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3  # We expect 3 events from our dummy_run_async

    # Verify we got the expected events
    assert data[0]["author"] == "dummy agent"
    assert data[0]["content"]["parts"][0]["text"] == "LLM reply"

    # Second event should have binary data
    assert (
        data[1]["content"]["parts"][0]["inlineData"]["mimeType"]
        == "audio/pcm;rate=24000"
    )

    # Third event should have interrupted flag
    assert data[2]["interrupted"] is True

    logger.info("Agent run test completed successfully")


def test_agent_run_passes_state_delta(test_app, create_test_session):
    """Test /run forwards state_delta and surfaces it in events."""
    info = create_test_session
    payload = {
        "app_name": info["app_name"],
        "user_id": info["user_id"],
        "session_id": info["session_id"],
        "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
        "streaming": False,
        "state_delta": {"k": "v", "count": 1},
    }

    # Verify the response
    response = test_app.post("/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 4

    # Verify we got the expected event
    assert data[3]["actions"]["stateDelta"] == payload["state_delta"]


def test_agent_run_passes_invocation_id(
    test_app, create_test_session, monkeypatch
):
    """Test /run forwards invocation_id for resumable invocations."""
    info = create_test_session
    captured_invocation_id: dict[str, Optional[str]] = {"invocation_id": None}

    async def run_async_capture(
        self,
        *,
        user_id: str,
        session_id: str,
        invocation_id: Optional[str] = None,
        new_message: Optional[types.Content] = None,
        state_delta: Optional[dict[str, Any]] = None,
        run_config: Optional[RunConfig] = None,
    ):
        del self, user_id, session_id, new_message, state_delta, run_config
        captured_invocation_id["invocation_id"] = invocation_id
        yield _event_1()

    monkeypatch.setattr(Runner, "run_async", run_async_capture)

    payload = {
        "app_name": info["app_name"],
        "user_id": info["user_id"],
        "session_id": info["session_id"],
        "new_message": {"role": "user", "parts": [{"text": "Resume run"}]},
        "streaming": False,
        "invocation_id": "resume-invocation-id",
    }

    response = test_app.post("/run", json=payload)

    assert response.status_code == 200
    assert captured_invocation_id["invocation_id"] == payload["invocation_id"]


def test_agent_run_sse_splits_artifact_delta(
    test_app, create_test_session, monkeypatch
):
    """Test /run_sse splits artifact deltas to avoid double-rendering in web."""
    info = create_test_session

    async def run_async_with_artifact_delta(
        self,
        *,
        user_id: str,
        session_id: str,
        invocation_id: Optional[str] = None,
        new_message: Optional[types.Content] = None,
        state_delta: Optional[dict[str, Any]] = None,
        run_config: Optional[RunConfig] = None,
    ):
        del (
            user_id,
            session_id,
            invocation_id,
            new_message,
            state_delta,
            run_config,
        )
        yield Event(
            author="dummy agent",
            invocation_id="invocation_id",
            content=types.Content(
                role="model", parts=[types.Part(text="LLM reply")]
            ),
            actions=EventActions(artifact_delta={"artifact.txt": 0}),
        )

    monkeypatch.setattr(Runner, "run_async", run_async_with_artifact_delta)

    payload = {
        "app_name": info["app_name"],
        "user_id": info["user_id"],
        "session_id": info["session_id"],
        "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
        "streaming": True,
    }

    response = test_app.post("/run_sse", json=payload)
    assert response.status_code == 200

    sse_events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert len(sse_events) == 2

    # First event: content but artifactDelta cleared.
    assert sse_events[0]["content"]["parts"][0]["text"] == "LLM reply"
    assert sse_events[0]["actions"]["artifactDelta"] == {}

    # Second event: artifactDelta but no content.
    assert "content" not in sse_events[1]
    assert sse_events[1]["actions"]["artifactDelta"] == {"artifact.txt": 0}


def test_agent_run_sse_does_not_split_artifact_delta_for_function_resume(
    test_app, create_test_session, monkeypatch
):
    """Test /run_sse keeps artifactDelta with content for function resume flow."""
    info = create_test_session

    async def run_async_with_artifact_delta(
        self,
        *,
        user_id: str,
        session_id: str,
        invocation_id: Optional[str] = None,
        new_message: Optional[types.Content] = None,
        state_delta: Optional[dict[str, Any]] = None,
        run_config: Optional[RunConfig] = None,
    ):
        del (
            user_id,
            session_id,
            invocation_id,
            new_message,
            state_delta,
            run_config,
        )
        yield Event(
            author="dummy agent",
            invocation_id="invocation_id",
            content=types.Content(
                role="model", parts=[types.Part(text="LLM reply")]
            ),
            actions=EventActions(artifact_delta={"artifact.txt": 0}),
        )

    monkeypatch.setattr(Runner, "run_async", run_async_with_artifact_delta)

    payload = {
        "app_name": info["app_name"],
        "user_id": info["user_id"],
        "session_id": info["session_id"],
        "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
        "streaming": True,
        "functionCallEventId": "function-call-event-id",
    }

    response = test_app.post("/run_sse", json=payload)
    assert response.status_code == 200

    sse_events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]

    assert len(sse_events) == 1
    assert sse_events[0]["content"]["parts"][0]["text"] == "LLM reply"
    assert sse_events[0]["actions"]["artifactDelta"] == {"artifact.txt": 0}


def test_agent_run_sse_yields_error_object_on_exception(
    test_app, create_test_session, monkeypatch
):
    """Test /run_sse streams an error object if streaming raises."""
    info = create_test_session

    async def run_async_raises(self, **kwargs):
        raise ValueError("boom")
        yield  # make it an async generator  # pylint: disable=unreachable

    monkeypatch.setattr(Runner, "run_async", run_async_raises)

    payload = {
        "app_name": info["app_name"],
        "user_id": info["user_id"],
        "session_id": info["session_id"],
        "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
        "streaming": True,
    }

    response = test_app.post("/run_sse", json=payload)
    assert response.status_code == 200

    sse_events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert sse_events == [{"error": "boom"}]


def test_list_artifact_names(test_app, create_test_session):
    """Test listing artifact names for a session."""
    info = create_test_session
    url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/{info['session_id']}/artifacts"
    response = test_app.get(url)

    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    logger.info(f"Listed {len(data)} artifacts")


def test_save_artifact(test_app, create_test_session, mock_artifact_service):
    """Test saving an artifact through the FastAPI endpoint."""
    info = create_test_session
    url = (
        f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
        f"{info['session_id']}/artifacts"
    )
    artifact_part = types.Part(text="hello world")
    payload = {
        "filename": "greeting.txt",
        "artifact": artifact_part.model_dump(by_alias=True, exclude_none=True),
    }

    response = test_app.post(url, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == 0
    assert data["customMetadata"] == {}
    assert data["mimeType"] in (None, "text/plain")
    assert data["canonicalUri"].endswith(
        f"/sessions/{info['session_id']}/artifacts/{payload['filename']}/versions/0"
    )
    assert isinstance(data["createTime"], float)

    key = (
        f"{info['app_name']}:{info['user_id']}:{info['session_id']}:"
        f"{payload['filename']}"
    )
    stored = mock_artifact_service._artifacts[key][0]
    assert stored["artifact"].text == "hello world"


def test_save_artifact_returns_400_on_validation_error(
    test_app, create_test_session, mock_artifact_service
):
    """Test save artifact endpoint surfaces validation errors as HTTP 400."""
    info = create_test_session
    url = (
        f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
        f"{info['session_id']}/artifacts"
    )
    artifact_part = types.Part(text="bad data")
    payload = {
        "filename": "invalid.txt",
        "artifact": artifact_part.model_dump(by_alias=True, exclude_none=True),
    }

    mock_artifact_service.save_artifact_side_effect = InputValidationError(
        "invalid artifact"
    )

    response = test_app.post(url, json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "invalid artifact"


def test_save_artifact_returns_500_on_unexpected_error(
    test_app, create_test_session, mock_artifact_service
):
    """Test save artifact endpoint surfaces unexpected errors as HTTP 500."""
    info = create_test_session
    url = (
        f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
        f"{info['session_id']}/artifacts"
    )
    artifact_part = types.Part(text="bad data")
    payload = {
        "filename": "invalid.txt",
        "artifact": artifact_part.model_dump(by_alias=True, exclude_none=True),
    }

    mock_artifact_service.save_artifact_side_effect = RuntimeError(
        "unexpected failure"
    )

    response = test_app.post(url, json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "unexpected failure"


def test_get_artifact_version_metadata(
    test_app, create_test_session, mock_artifact_service
):
    """Test retrieving metadata for a specific artifact version."""
    info = create_test_session
    mock_artifact_service.add_artifact(
        app_name=info["app_name"],
        user_id=info["user_id"],
        session_id=info["session_id"],
        filename="report.txt",
        artifact=types.Part(text="hello"),
        custom_metadata={"foo": "bar"},
        mime_type="text/plain",
    )

    url = (
        f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
        f"{info['session_id']}/artifacts/report.txt/versions/0/metadata"
    )
    response = test_app.get(url)

    assert response.status_code == 200
    data = response.json()
    assert data["version"] == 0
    assert data["customMetadata"] == {"foo": "bar"}
    assert data["mimeType"] == "text/plain"


def test_list_artifact_versions_metadata(
    test_app, create_test_session, mock_artifact_service
):
    """Test listing metadata for all versions of an artifact."""
    info = create_test_session
    mock_artifact_service.add_artifact(
        app_name=info["app_name"],
        user_id=info["user_id"],
        session_id=info["session_id"],
        filename="report.txt",
        artifact=types.Part(text="v0"),
    )
    mock_artifact_service.add_artifact(
        app_name=info["app_name"],
        user_id=info["user_id"],
        session_id=info["session_id"],
        filename="report.txt",
        artifact=types.Part(text="v1"),
        custom_metadata={"foo": "bar"},
    )

    url = (
        f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
        f"{info['session_id']}/artifacts/report.txt/versions/metadata"
    )
    response = test_app.get(url)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[1]["version"] == 1
    assert data[1]["customMetadata"] == {"foo": "bar"}


def test_get_eval_set_result_not_found(test_app):
    """Test getting an eval set result that doesn't exist."""
    url = "/apps/test_app_name/eval_results/test_eval_result_id_not_found"
    response = test_app.get(url)
    assert response.status_code == 404


def test_list_metrics_info(test_app):
    """Test listing metrics info."""
    url = "/apps/test_app/metrics-info"
    response = test_app.get(url)

    # Verify the response
    assert response.status_code == 200
    data = response.json()
    metrics_info_key = "metricsInfo"
    assert metrics_info_key in data
    assert isinstance(data[metrics_info_key], list)
    # Add more assertions based on the expected metrics
    assert len(data[metrics_info_key]) > 0
    for metric in data[metrics_info_key]:
        assert "metricName" in metric
        assert "description" in metric
        assert "metricValueInfo" in metric


def test_list_metrics_info_omits_metrics_that_need_no_threshold(
    builder_test_client,
):
  """Always-on informational metrics are not offered for threshold selection.

  This surface asks the user to pick metrics and set a threshold for each, and
  bounds the threshold control by the metric's value interval. Metrics that
  need no threshold have neither, so listing them leaves consumers with nothing
  to render.
  """
  response = builder_test_client.get("/dev/apps/test_app/metrics-info")

  assert response.status_code == 200
  listed = [metric["metricName"] for metric in response.json()["metricsInfo"]]
  assert "tool_trajectory_avg_score" in listed
  for informational in (
      "tool_call_count_v1",
      "inference_call_count_v1",
      "token_usage_v1",
  ):
    assert informational not in listed
  # Everything that is listed can be rendered as a bounded threshold control.
  for metric in response.json()["metricsInfo"]:
    assert metric["metricValueInfo"]["interval"]


def test_debug_trace(test_app):
    """Test the debug trace endpoint."""
    # This test will likely return 404 since we haven't set up trace data,
    # but it tests that the endpoint exists and handles missing traces correctly.
    url = "/debug/trace/nonexistent-event"
    response = test_app.get(url)

    # Verify we get a 404 for a nonexistent trace
    assert response.status_code == 404
    logger.info("Debug trace test completed successfully")


def test_openapi_json_schema_accessible(test_app):
    """Test that the OpenAPI /openapi.json endpoint is accessible."""
    response = test_app.get("/openapi.json")
    assert response.status_code == 200
    logger.info("OpenAPI /openapi.json endpoint is accessible")


def test_get_event_graph_returns_dot_src_for_app_agent():
    """Ensure graph endpoint unwraps App instances before building the graph."""
    from google.adk.cli.adk_web_server import AdkWebServer

    root_agent = DummyAgent(name="dummy_agent")
    app_agent = App(name="test_app", root_agent=root_agent)

    class Loader:
        def load_agent(self, app_name):
            return app_agent

        def list_agents(self):
            return [app_agent.name]

    session_service = AsyncMock()
    session = Session(
        id="session_id",
        app_name="test_app",
        user_id="user",
        state={},
        events=[Event(author="dummy_agent")],
    )
    event_id = session.events[0].id
    session_service.get_session.return_value = session

    adk_web_server = AdkWebServer(
        agent_loader=Loader(),
        session_service=session_service,
        memory_service=MagicMock(),
        artifact_service=MagicMock(),
        credential_service=MagicMock(),
        eval_sets_manager=MagicMock(),
        eval_set_results_manager=MagicMock(),
        agents_dir=".",
    )

    fast_api_app = adk_web_server.get_fast_api_app(
        setup_observer=lambda _observer, _server: None,
        tear_down_observer=lambda _observer, _server: None,
    )

    client = TestClient(fast_api_app)
    response = client.get(
        f"/apps/test_app/users/user/sessions/session_id/events/{event_id}/graph"
    )
    assert response.status_code == 200
    assert "dotSrc" in response.json()


def test_a2a_agent_discovery(test_app_with_a2a):
    """Test that A2A agents are properly discovered and configured."""
    # This test mainly verifies that the A2A setup doesn't break the app
    response = test_app_with_a2a.get("/list-apps")
    assert response.status_code == 200
    logger.info("A2A agent discovery test passed")


def test_a2a_request_handler_uses_push_config_store(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    temp_agents_dir_with_a2a,
    monkeypatch,
):
    """Test A2A request handler gets push config store when supported."""
    with (
        patch("signal.signal", return_value=None),
        patch(
            "google.adk.cli.fast_api.create_session_service_from_options",
            return_value=mock_session_service,
        ),
        patch(
            "google.adk.cli.fast_api.create_artifact_service_from_options",
            return_value=mock_artifact_service,
        ),
        patch(
            "google.adk.cli.fast_api.create_memory_service_from_options",
            return_value=mock_memory_service,
        ),
        patch(
            "google.adk.cli.fast_api.AgentLoader",
            return_value=mock_agent_loader,
        ),
        patch(
            "google.adk.cli.fast_api.LocalEvalSetsManager",
            return_value=mock_eval_sets_manager,
        ),
        patch(
            "google.adk.cli.fast_api.LocalEvalSetResultsManager",
            return_value=mock_eval_set_results_manager,
        ),
        patch("a2a.server.tasks.InMemoryTaskStore") as mock_task_store,
        patch(
            "a2a.server.tasks.InMemoryPushNotificationConfigStore"
        ) as mock_push_config_store_class,
        patch(
            "google.adk.a2a.executor.a2a_agent_executor.A2aAgentExecutor"
        ) as mock_executor,
        patch(
            "a2a.server.request_handlers.DefaultRequestHandler"
        ) as mock_handler,
        patch("a2a.server.apps.A2AStarletteApplication") as mock_a2a_app,
    ):
        mock_task_store_instance = MagicMock()
        mock_task_store.return_value = mock_task_store_instance
        mock_push_config_store = MagicMock()
        mock_push_config_store_class.return_value = mock_push_config_store
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        mock_handler.return_value = MagicMock()
        mock_a2a_app_instance = MagicMock()
        mock_a2a_app_instance.routes.return_value = []
        mock_a2a_app.return_value = mock_a2a_app_instance

        monkeypatch.chdir(temp_agents_dir_with_a2a)
        _ = get_fast_api_app(
            agents_dir=".",
            web=True,
            session_service_uri="",
            artifact_service_uri="",
            memory_service_uri="",
            allow_origins=["*"],
            a2a=True,
            host="127.0.0.1",
            port=8000,
        )

        mock_handler.assert_called_once_with(
            agent_executor=mock_executor_instance,
            push_config_store=mock_push_config_store,
            task_store=mock_task_store_instance,
        )


def test_a2a_disabled_by_default(test_app):
    """Test that A2A functionality is disabled by default."""
    # The regular test_app fixture has a2a=False
    # This test ensures no A2A routes are added
    response = test_app.get("/list-apps")
    assert response.status_code == 200
    logger.info("A2A disabled by default test passed")


def test_patch_memory(test_app, create_test_session, mock_memory_service):
    """Test adding a session to memory."""
    info = create_test_session
    url = f"/apps/{info['app_name']}/users/{info['user_id']}/memory"
    payload = {"session_id": info["session_id"]}
    response = test_app.patch(url, json=payload)

    # Verify the response
    assert response.status_code == 200
    mock_memory_service.add_session_to_memory.assert_called_once()
    logger.info("Add session to memory test completed successfully")


def test_builder_final_save_preserves_files_and_cleans_tmp(
    builder_test_client, tmp_path
):
    files = [
        (
            "files",
            ("app/root_agent.yaml", b"name: app\n", "application/x-yaml"),
        ),
        (
            "files",
            ("app/sub_agent.yaml", b"name: sub\n", "application/x-yaml"),
        ),
    ]
    response = builder_test_client.post("/builder/save?tmp=true", files=files)
    assert response.status_code == 200
    assert response.json() is True

    response = builder_test_client.post(
        "/builder/save",
        files=[
            (
                "files",
                (
                    "app/root_agent.yaml",
                    b"name: app_updated\n",
                    "application/x-yaml",
                ),
            )
        ],
    )
    assert response.status_code == 200
    assert response.json() is True

    assert (tmp_path / "app" / "sub_agent.yaml").is_file()
    assert not (tmp_path / "app" / "tmp" / "app").exists()
    tmp_dir = tmp_path / "app" / "tmp"
    assert not tmp_dir.exists() or not any(tmp_dir.iterdir())


def test_builder_save_rejects_cross_origin_post(builder_test_client, tmp_path):
    response = builder_test_client.post(
        "/builder/save?tmp=true",
        headers={"origin": "https://evil.com"},
        files=[
            (
                "files",
                ("app/root_agent.yaml", b"name: app\n", "application/x-yaml"),
            )
        ],
    )

    assert response.status_code == 403
    assert response.text == "Forbidden: origin not allowed"
    assert not (tmp_path / "app" / "tmp" / "app").exists()


def test_builder_save_allows_same_origin_post(builder_test_client, tmp_path):
    response = builder_test_client.post(
        "/builder/save?tmp=true",
        headers={"origin": "http://testserver"},
        files=[
            (
                "files",
                ("app/root_agent.yaml", b"name: app\n", "application/x-yaml"),
            )
        ],
    )

    assert response.status_code == 200
    assert response.json() is True
    assert (tmp_path / "app" / "tmp" / "app" / "root_agent.yaml").is_file()


def test_builder_get_allows_cross_origin_get(builder_test_client):
    response = builder_test_client.get(
        "/builder/app/missing?tmp=true",
        headers={"origin": "https://evil.com"},
    )

    assert response.status_code == 200
    assert response.text == ""


def test_builder_cancel_deletes_tmp_idempotent(builder_test_client, tmp_path):
    tmp_agent_root = tmp_path / "app" / "tmp" / "app"
    tmp_agent_root.mkdir(parents=True, exist_ok=True)
    (tmp_agent_root / "root_agent.yaml").write_text("name: app\n")

    response = builder_test_client.post("/builder/app/app/cancel")
    assert response.status_code == 200
    assert response.json() is True
    assert not (tmp_path / "app" / "tmp").exists()

    response = builder_test_client.post("/builder/app/app/cancel")
    assert response.status_code == 200
    assert response.json() is True
    assert not (tmp_path / "app" / "tmp").exists()


def test_builder_get_tmp_true_recreates_tmp(builder_test_client, tmp_path):
    app_root = tmp_path / "app"
    app_root.mkdir(parents=True, exist_ok=True)
    (app_root / "root_agent.yaml").write_text("name: app\n")
    nested_dir = app_root / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (nested_dir / "nested.yaml").write_text("nested: true\n")

    assert not (app_root / "tmp").exists()
    response = builder_test_client.get("/builder/app/app?tmp=true")
    assert response.status_code == 200
    assert response.text == "name: app\n"

    tmp_agent_root = app_root / "tmp" / "app"
    assert (tmp_agent_root / "root_agent.yaml").is_file()
    assert (tmp_agent_root / "nested" / "nested.yaml").is_file()

    response = builder_test_client.get(
        "/builder/app/app?tmp=true&file_path=nested/nested.yaml"
    )
    assert response.status_code == 200
    assert response.text == "nested: true\n"


def test_builder_get_tmp_true_missing_app_returns_empty(
    builder_test_client, tmp_path
):
    response = builder_test_client.get("/builder/app/missing?tmp=true")
    assert response.status_code == 200
    assert response.text == ""
    assert not (tmp_path / "missing").exists()


def test_builder_save_rejects_traversal(builder_test_client, tmp_path):
    response = builder_test_client.post(
        "/builder/save?tmp=true",
        files=[
            (
                "files",
                ("app/../escape.yaml", b"nope\n", "application/x-yaml"),
            )
        ],
    )
    assert response.status_code == 400
    assert not (tmp_path / "escape.yaml").exists()
    assert not (tmp_path / "app" / "tmp" / "escape.yaml").exists()


def test_builder_save_rejects_py_files(builder_test_client, tmp_path):
    """Uploading .py files via /builder/save is rejected."""
    response = builder_test_client.post(
        "/builder/save?tmp=true",
        files=[
            (
                "files",
                (
                    "app/agent.py",
                    b"import os\nos.system('id')\n",
                    "text/plain",
                ),
            )
        ],
    )
    assert response.status_code == 400
    assert not (tmp_path / "app" / "tmp" / "app" / "agent.py").exists()


def test_builder_save_rejects_non_yaml_extensions(
    builder_test_client, tmp_path
):
    """Uploading non-YAML files (.json, .txt, .sh, etc.) is rejected."""
    for ext, content in [
        (".py", b"print('hi')"),
        (".json", b"{}"),
        (".txt", b"hello"),
        (".sh", b"#!/bin/bash"),
        (".pth", b"import os"),
    ]:
        response = builder_test_client.post(
            "/builder/save?tmp=true",
            files=[
                (
                    "files",
                    (f"app/file{ext}", content, "application/octet-stream"),
                )
            ],
        )
        assert response.status_code == 400, f"Expected 400 for {ext}"


def test_builder_save_allows_yaml_files(builder_test_client, tmp_path):
    """Uploading .yaml and .yml files is allowed."""
    response = builder_test_client.post(
        "/builder/save?tmp=true",
        files=[
            (
                "files",
                ("app/root_agent.yaml", b"name: app\n", "application/x-yaml"),
            )
        ],
    )
    assert response.status_code == 200
    assert response.json() is True

    response = builder_test_client.post(
        "/builder/save?tmp=true",
        files=[
            (
                "files",
                ("app/sub_agent.yml", b"name: sub\n", "application/x-yaml"),
            )
        ],
    )
    assert response.status_code == 200
    assert response.json() is True


def test_builder_save_rejects_args_key(builder_test_client, tmp_path):
    """Uploading YAML with an `args` key is rejected (RCE prevention)."""
    yaml_with_args = b"""\
name: my_tool
args:
  key: value
"""
    response = builder_test_client.post(
        "/builder/save?tmp=true",
        files=[
            (
                "files",
                ("app/root_agent.yaml", yaml_with_args, "application/x-yaml"),
            )
        ],
    )
    assert response.status_code == 400
    assert "args" in response.json()["detail"]
    assert not (tmp_path / "app" / "tmp" / "app" / "root_agent.yaml").exists()


def test_builder_save_rejects_nested_args_key(builder_test_client, tmp_path):
    """Uploading YAML with a nested `args` key is rejected."""
    yaml_with_nested_args = b"""\
tools:
  - name: some_tool
    args:
      param: value
"""
    response = builder_test_client.post(
        "/builder/save?tmp=true",
        files=[
            (
                "files",
                (
                    "app/root_agent.yaml",
                    yaml_with_nested_args,
                    "application/x-yaml",
                ),
            )
        ],
    )
    assert response.status_code == 400
    assert "args" in response.json()["detail"]


def test_builder_get_rejects_non_yaml_file_paths(
    builder_test_client, tmp_path
):
    """GET /builder/app/{app_name}?file_path=... rejects non-YAML extensions."""
    app_root = tmp_path / "app"
    app_root.mkdir(parents=True, exist_ok=True)
    (app_root / ".env").write_text("SECRET=supersecret\n")
    (app_root / "agent.py").write_text("root_agent = None\n")
    (app_root / "config.json").write_text("{}\n")

    for file_path in [".env", "agent.py", "config.json"]:
        response = builder_test_client.get(
            f"/builder/app/app?file_path={file_path}"
        )
        assert response.status_code == 200, f"Expected 200 for {file_path}"
        assert response.text == "", f"Expected empty response for {file_path}"


def test_builder_get_allows_yaml_file_paths(builder_test_client, tmp_path):
    """GET /builder/app/{app_name}?file_path=... allows YAML extensions."""
    app_root = tmp_path / "app"
    app_root.mkdir(parents=True, exist_ok=True)
    (app_root / "sub_agent.yaml").write_text("name: sub\n")
    (app_root / "tool.yml").write_text("name: tool\n")

    response = builder_test_client.get(
        "/builder/app/app?file_path=sub_agent.yaml"
    )
    assert response.status_code == 200
    assert response.text == "name: sub\n"

    response = builder_test_client.get("/builder/app/app?file_path=tool.yml")
    assert response.status_code == 200
    assert response.text == "name: tool\n"


def test_builder_endpoints_not_registered_without_web(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
    """Builder endpoints must not be registered when web=False (e.g. deploy)."""
    client = _create_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        mock_eval_sets_manager,
        mock_eval_set_results_manager,
        web=False,
    )
    # /builder/save should return 404/405, not 200
    response = client.post(
        "/builder/save",
        files=[
            (
                "files",
                ("app/agent.yaml", b"name: test\n", "application/x-yaml"),
            )
        ],
    )
    assert response.status_code in (404, 405)

    # /builder/app/{name}/cancel should also be absent
    response = client.post("/builder/app/app/cancel")
    assert response.status_code in (404, 405)

    # /builder/app/{name} GET should also be absent
    response = client.get("/builder/app/app")
    assert response.status_code in (404, 405)


def test_builder_endpoints_registered_with_web(builder_test_client):
    """Builder endpoints are available when web=True."""
    response = builder_test_client.post(
        "/builder/save?tmp=true",
        files=[
            (
                "files",
                ("app/agent.yaml", b"name: test\n", "application/x-yaml"),
            )
        ],
    )
    assert response.status_code == 200


def test_agent_run_resume_without_message_success(
    test_app, create_test_session
):
    """Test that /run allows resuming a session with only an invocation_id, without a new message."""
    info = create_test_session
    url = "/run"
    payload = {
        "app_name": info["app_name"],
        "user_id": info["user_id"],
        "session_id": info["session_id"],
        "invocation_id": "test_invocation_id",
        "streaming": False,
    }
    response = test_app.post(url, json=payload)
    assert response.status_code == 200


def test_health_endpoint(test_app):
    """Test the health endpoint."""
    response = test_app.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint(test_app):
    """Test the version endpoint."""
    response = test_app.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "language" in data
    assert data["language"] == "python"
    assert "language_version" in data


@pytest.fixture
def test_app_auto_session(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
    """Create a TestClient with auto_create_session=True."""
    return _create_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        mock_eval_sets_manager,
        mock_eval_set_results_manager,
        web=False,
        auto_create_session=True,
    )


@pytest.mark.parametrize("endpoint", ["/run", "/run_sse"])
def test_auto_creates_session(
    test_app_auto_session, test_session_info, endpoint
):
    """Test /run and /run_sse auto-create sessions when auto_create_session=True."""
    payload = {
        "app_name": test_session_info["app_name"],
        "user_id": test_session_info["user_id"],
        "session_id": "nonexistent_session",
        "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
    }

    response = test_app_auto_session.post(endpoint, json=payload)
    assert response.status_code == 200

    if endpoint == "/run":
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    else:
        sse_events = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert len(sse_events) > 0
        assert not any("error" in e for e in sse_events)


def test_finalize_agent_identity_credentials_invalid_base64(test_app):
  """Test error handling when user_id_validation_state is invalid Base64."""
  from google.cloud import iamconnectorcredentials_v1alpha

  with patch.object(
      iamconnectorcredentials_v1alpha,
      "IAMConnectorCredentialsServiceClient",
      autospec=True,
  ):
    response = test_app.post(
        "/agent-identity/finalize",
        json={
            "connector_name": "projects/p/locations/l/connectors/c",
            "user_id": "user-123",
            "user_id_validation_state": "!!!invalid_base64!!!",
            "consent_nonce": "nonce-456",
        },
    )
    assert response.status_code == 400
    assert (
        "Invalid base64 user_id_validation_state" in response.json()["detail"]
    )


def test_finalize_agent_identity_credentials_missing_dependency(test_app):
  """Test error handling when google-cloud-iamconnectorcredentials is not installed."""
  with patch.dict(
      "sys.modules", {"google.cloud.iamconnectorcredentials_v1alpha": None}
  ):
    response = test_app.post(
        "/agent-identity/finalize",
        json={
            "connector_name": "projects/p/locations/l/connectors/c",
            "user_id": "user-123",
            "user_id_validation_state": "dGVzdA",
            "consent_nonce": "nonce-456",
        },
    )
    assert response.status_code == 500
    assert "Agent Identity support requires" in response.json()["detail"]


def test_finalize_agent_identity_credentials_invalid_argument_error(test_app):
  """Test backend InvalidArgument API error handling (400 response)."""
  from google.cloud import iamconnectorcredentials_v1alpha

  with patch.object(
      iamconnectorcredentials_v1alpha,
      "IAMConnectorCredentialsServiceClient",
      autospec=True,
  ) as mock_client_cls:
    mock_client = mock_client_cls.return_value
    mock_client.finalize_credentials.side_effect = InvalidArgument(
        "Invalid consent nonce"
    )

    response = test_app.post(
        "/agent-identity/finalize",
        json={
            "connector_name": "projects/p/locations/l/connectors/c",
            "user_id": "user-123",
            "user_id_validation_state": "dGVzdA",
            "consent_nonce": "invalid-nonce",
        },
    )
    assert response.status_code == 400
    assert "Invalid credentials request" in response.json()["detail"]


def test_finalize_agent_identity_credentials_api_call_error(test_app):
  """Test backend GoogleAPICallError error handling with status code propagation."""
  from google.cloud import iamconnectorcredentials_v1alpha

  err = GoogleAPICallError("Permission denied")
  err.code = 403

  with patch.object(
      iamconnectorcredentials_v1alpha,
      "IAMConnectorCredentialsServiceClient",
      autospec=True,
  ) as mock_client_cls:
    mock_client = mock_client_cls.return_value
    mock_client.finalize_credentials.side_effect = err

    response = test_app.post(
        "/agent-identity/finalize",
        json={
            "connector_name": "projects/p/locations/l/connectors/c",
            "user_id": "user-123",
            "user_id_validation_state": "dGVzdA",
            "consent_nonce": "nonce-456",
        },
    )
    assert response.status_code == 403
    assert "Failed to finalize credentials" in response.json()["detail"]


#################################################
# Span Exporter Tests
#################################################


def _readable_span(name, *, trace_id, span_id=1, attributes=None):
  """Builds a finished span suitable for feeding a SpanExporter."""
  from opentelemetry.sdk.trace import ReadableSpan
  from opentelemetry.trace import SpanContext

  return ReadableSpan(
      name=name,
      context=SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False),
      attributes=attributes or {},
  )


def test_api_server_span_exporter_records_only_llm_and_tool_spans():
  """Only call_llm / send_data / execute_tool* spans are kept, by event id."""
  from google.adk.cli.api_server import ApiServerSpanExporter
  from opentelemetry.sdk.trace.export import SpanExportResult

  trace_dict = {}
  exporter = ApiServerSpanExporter(trace_dict)

  spans = [
      _readable_span(
          "call_llm",
          trace_id=11,
          span_id=1,
          attributes={"gcp.vertex.agent.event_id": "llm-event"},
      ),
      _readable_span(
          "send_data",
          trace_id=12,
          span_id=2,
          attributes={"gcp.vertex.agent.event_id": "data-event"},
      ),
      _readable_span(
          "execute_tool my_tool",
          trace_id=13,
          span_id=3,
          attributes={"gcp.vertex.agent.event_id": "tool-event"},
      ),
      _readable_span(
          "invocation",
          trace_id=14,
          span_id=4,
          attributes={"gcp.vertex.agent.event_id": "unrelated-event"},
      ),
  ]

  assert exporter.export(spans) == SpanExportResult.SUCCESS

  assert sorted(trace_dict) == ["data-event", "llm-event", "tool-event"]
  # The exporter augments the span attributes with its trace/span identifiers,
  # which is what the /debug/trace endpoint hands back to the UI.
  assert trace_dict["llm-event"]["trace_id"] == 11
  assert trace_dict["llm-event"]["span_id"] == 1
  assert trace_dict["tool-event"]["trace_id"] == 13


def test_api_server_span_exporter_skips_span_without_event_id():
  """A traced span carrying no event id cannot be keyed, so it is dropped."""
  from google.adk.cli.api_server import ApiServerSpanExporter

  trace_dict = {}
  exporter = ApiServerSpanExporter(trace_dict)

  exporter.export([
      _readable_span(
          "call_llm",
          trace_id=21,
          attributes={"gcp.vertex.agent.session_id": "session-a"},
      )
  ])

  assert trace_dict == {}


def test_in_memory_exporter_returns_only_spans_of_requested_session():
  """Spans are indexed per session id and looked up by trace id."""
  from google.adk.cli.api_server import InMemoryExporter

  session_trace_dict = {}
  exporter = InMemoryExporter(session_trace_dict)

  span_a1 = _readable_span(
      "call_llm",
      trace_id=101,
      span_id=1,
      attributes={"gcp.vertex.agent.session_id": "session-a"},
  )
  span_a2 = _readable_span(
      "execute_tool my_tool",
      trace_id=101,
      span_id=2,
      attributes={"gcp.vertex.agent.session_id": "session-a"},
  )
  span_b = _readable_span(
      "call_llm",
      trace_id=202,
      span_id=3,
      attributes={"gcp.vertex.agent.session_id": "session-b"},
  )

  exporter.export([span_a1, span_a2, span_b])

  # Both session-a spans share a trace, so the trace id is recorded once.
  assert session_trace_dict == {"session-a": [101], "session-b": [202]}
  assert exporter.get_finished_spans("session-a") == [span_a1, span_a2]
  assert exporter.get_finished_spans("session-b") == [span_b]
  assert exporter.get_finished_spans("session-never-seen") == []


def test_in_memory_exporter_falls_back_to_conversation_id():
  """A span with no agent session id is indexed by the conversation id."""
  from google.adk.cli.api_server import InMemoryExporter

  session_trace_dict = {}
  exporter = InMemoryExporter(session_trace_dict)

  conversation_span = _readable_span(
      "call_llm",
      trace_id=303,
      span_id=1,
      attributes={"gen_ai.conversation.id": "conversation-1"},
  )
  unattributed_span = _readable_span("call_llm", trace_id=404, span_id=2)

  exporter.export([conversation_span, unattributed_span])

  assert session_trace_dict == {"conversation-1": [303]}
  assert exporter.get_finished_spans("conversation-1") == [conversation_span]


def test_in_memory_exporter_clear_drops_spans_but_keeps_session_index():
  """clear() forgets the spans; the session -> trace id index is untouched."""
  from google.adk.cli.api_server import InMemoryExporter

  session_trace_dict = {}
  exporter = InMemoryExporter(session_trace_dict)
  span = _readable_span(
      "call_llm",
      trace_id=505,
      attributes={"gcp.vertex.agent.session_id": "session-a"},
  )
  exporter.export([span])
  assert exporter.get_finished_spans("session-a") == [span]

  exporter.clear()

  assert exporter.get_finished_spans("session-a") == []
  assert session_trace_dict == {"session-a": [505]}


def test_setup_telemetry_guards_add_span_processor_on_non_sdk_provider(
    monkeypatch,
):
  """A non-SDK TracerProvider lacking add_span_processor does not raise."""
  from unittest.mock import MagicMock

  from google.adk.cli.api_server import _setup_telemetry
  from opentelemetry import trace

  non_sdk_provider = MagicMock(spec=trace.TracerProvider)
  del non_sdk_provider.add_span_processor
  monkeypatch.setattr(trace, "get_tracer_provider", lambda: non_sdk_provider)

  exporter = MagicMock()
  _setup_telemetry(otel_to_cloud=False, internal_exporters=[exporter])


def test_setup_telemetry_adds_span_processors_when_supported(monkeypatch):
  """A TracerProvider with add_span_processor registers the internal exporters."""
  from unittest.mock import MagicMock

  from google.adk.cli.api_server import _setup_telemetry
  from opentelemetry import trace

  provider = MagicMock()
  monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)

  exporter = MagicMock()
  _setup_telemetry(otel_to_cloud=False, internal_exporters=[exporter])
  provider.add_span_processor.assert_called_once_with(exporter)


#################################################
# Request-body plumbing tests
#################################################


def test_create_session_applies_body_session_id_state_and_events(
    test_app, test_session_info
):
    """Test /run and /run_sse return 404 for missing sessions without auto_create."""

    async def run_async_session_not_found(self, **kwargs):
        raise SessionNotFoundError(
            f"Session not found: {kwargs['session_id']}"
        )
        yield  # make it an async generator  # pylint: disable=unreachable

    monkeypatch.setattr(Runner, "run_async", run_async_session_not_found)

    payload = {
        "app_name": test_session_info["app_name"],
        "user_id": test_session_info["user_id"],
        "session_id": "nonexistent_session",
        "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
    }

def test_patch_memory_unknown_session_returns_404(
    test_app, test_session_info, mock_memory_service
):
  """A request naming a missing session must not reach the memory service."""
  url = (
      f"/apps/{test_session_info['app_name']}"
      f"/users/{test_session_info['user_id']}/memory"
  )

  response = test_app.patch(url, json={"session_id": "no_such_session"})

  assert response.status_code == 404
  assert response.json()["detail"] == "Session not found"
  mock_memory_service.add_session_to_memory.assert_not_called()


#################################################
# ApiServer vs DevServer endpoint surface
#################################################


def test_dev_only_endpoints_absent_when_web_disabled(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """web=False serves ApiServer only: no eval / debug / graph routes."""
  client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      web=False,
  )

  dev_only_paths = [
      "/config/telemetry",
      "/dev/apps/test_app/eval-sets",
      "/dev/apps/test_app/eval-results",
      "/dev/apps/test_app/metrics-info",
      "/dev/apps/test_app/tests",
      "/dev/apps/test_app/graph",
      "/dev/apps/test_app/debug/trace/some-event",
  ]
  for path in dev_only_paths:
    assert client.get(path).status_code == 404, path

  # The production endpoints are still there.
  assert client.get("/health").status_code == 200
  assert client.get("/list-apps").status_code == 200


def _installed_internal_exporters(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    *,
    web: bool,
) -> list[str]:
  """Names of the span exporters the server registers on the tracer provider."""
  with patch.object(
      api_server_module, "_setup_telemetry", autospec=True
  ) as mock_setup_telemetry:
    _create_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        mock_eval_sets_manager,
        mock_eval_set_results_manager,
        web=web,
    )
  processors = mock_setup_telemetry.call_args.kwargs["internal_exporters"]
  return [type(processor.span_exporter).__name__ for processor in processors]


def test_span_buffers_not_filled_when_web_disabled(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Nothing reads the in-memory spans on web=False, so nothing writes them."""
  assert (
      _installed_internal_exporters(
          mock_session_service,
          mock_artifact_service,
          mock_memory_service,
          mock_agent_loader,
          mock_eval_sets_manager,
          mock_eval_set_results_manager,
          web=False,
      )
      == []
  )


def test_span_buffers_filled_when_web_enabled(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """The dev server's trace views need the spans, so both exporters run."""
  assert _installed_internal_exporters(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      web=True,
  ) == ["ApiServerSpanExporter", "InMemoryExporter"]


def test_app_info_rejects_special_agent_only_in_api_server_mode(
    test_app,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Internal `__` apps reach the dev server, but not the api server."""
  api_only_client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      web=False,
  )

  blocked = api_only_client.get("/apps/__internal_assistant/app-info")
  assert blocked.status_code == 403
  assert "internal special agents" in blocked.json()["detail"]

  # Same request on the dev server gets past the guard and is answered on the
  # merits of the loaded agent (which here is not an LlmAgent).
  allowed = test_app.get("/apps/__internal_assistant/app-info")
  assert allowed.status_code == 400
  assert allowed.json()["detail"] == "Root agent is not an LlmAgent"


def test_dev_endpoint_rejects_app_name_that_is_not_an_identifier(
    builder_test_client,
):
  """_get_agent_dir only accepts dot-separated Python identifiers."""
  ok = builder_test_client.get("/dev/apps/test_app/tests")
  assert ok.status_code == 200
  assert ok.json() == []

  nested_ok = builder_test_client.get("/dev/apps/pkg.test_app/tests")
  assert nested_ok.status_code == 200

  for bad_name in ("bad-name", "1app", "app%20name"):
    rejected = builder_test_client.get(f"/dev/apps/{bad_name}/tests")
    assert rejected.status_code == 400, bad_name
    assert "must be valid" in rejected.json()["detail"]


#################################################
# Eval endpoint plumbing
#################################################


def test_add_session_to_eval_set_builds_eval_case_from_session(
    test_app, test_session_info, mock_eval_sets_manager
):
  """AddSessionToEvalSetRequest turns a live session into an eval case."""
  app_name = test_session_info["app_name"]
  user_id = test_session_info["user_id"]
  mock_eval_sets_manager.create_eval_set(
      app_name=app_name, eval_set_id="my_eval_set"
  )

  sessions_url = f"/apps/{app_name}/users/{user_id}/sessions"
  created = test_app.post(
      sessions_url,
      json={
          "session_id": "eval_source_session",
          "events": [
              {
                  "author": "user",
                  "invocationId": "inv-1",
                  "content": {
                      "role": "user",
                      "parts": [{"text": "what is 2+2?"}],
                  },
              },
              {
                  "author": "dummy agent",
                  "invocationId": "inv-1",
                  "content": {"role": "model", "parts": [{"text": "4"}]},
              },
          ],
      },
  )
  assert created.status_code == 200

  response = test_app.post(
      f"/dev/apps/{app_name}/eval-sets/my_eval_set/add-session",
      json={
          "eval_id": "my_eval_case",
          "session_id": "eval_source_session",
          "user_id": user_id,
      },
  )
  assert response.status_code == 200

  eval_case = mock_eval_sets_manager.get_eval_case(
      app_name, "my_eval_set", "my_eval_case"
  )
  assert eval_case is not None
  assert eval_case.session_input.app_name == app_name
  assert eval_case.session_input.user_id == user_id
  assert [
      part.text
      for invocation in eval_case.conversation
      for part in invocation.user_content.parts
  ] == ["what is 2+2?"]


@pytest.mark.xfail(
    strict=True,
    reason="add-session maps ValueError, but the managers raise NotFoundError",
)
def test_add_session_to_eval_set_unknown_eval_set_is_a_client_error(
    test_app, create_test_session
):
  """Adding to an eval set that never existed is a client error, not a 500."""
  info = create_test_session

  response = test_app.post(
      f"/dev/apps/{info['app_name']}/eval-sets/missing_eval_set/add-session",
      json={
          "eval_id": "case-1",
          "session_id": info["session_id"],
          "user_id": info["user_id"],
      },
  )

  assert 400 <= response.status_code < 500


def test_get_eval_result_returns_saved_eval_set_result(
    test_app, mock_eval_set_results_manager
):
  """The eval-results endpoint renames EvalSetResult to EvalResult as-is."""
  mock_eval_set_results_manager.save_eval_set_result(
      "test_app", "my_eval_set", []
  )

  response = test_app.get(
      "/dev/apps/test_app/eval-results/test_app_my_eval_set_eval_result"
  )

  assert response.status_code == 200
  data = response.json()
  assert data["evalSetResultId"] == "test_app_my_eval_set_eval_result"
  assert data["evalSetId"] == "my_eval_set"


def test_create_eval_set_legacy_route_creates_eval_set(
    test_app, mock_eval_sets_manager
):
  """The deprecated create-eval-set route should create an empty eval set."""
  response = test_app.post("/dev/apps/test_app/eval_sets/legacy_eval_set")

  assert response.status_code == 200
  assert (
      mock_eval_sets_manager.get_eval_set("test_app", "legacy_eval_set")
      is not None
  )


def test_agent_run_sse_deferred_with_streaming_returns_422(
    test_app, create_test_session
):
  """Deferred plus SSE is rejected up front, not mid-stream.

  A deferred create returns an interaction id rather than a result, so it
  cannot stream. The run config is built before the response starts so the
  caller gets a status code instead of a 200 that breaks partway through.

  Args:
    test_app: The FastAPI test client.
    create_test_session: Fixture creating the session the request targets.
  """
  payload = {
      "app_name": create_test_session["app_name"],
      "user_id": create_test_session["user_id"],
      "session_id": create_test_session["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": True,
      "service_tier": "deferred",
  }

  response = test_app.post("/run_sse", json=payload)

  assert response.status_code == 422
  assert "cannot be used with StreamingMode.SSE" in response.json()["detail"]


def test_agent_run_sse_deferred_without_streaming_is_allowed(
    test_app, create_test_session, monkeypatch
):
  """Deferred is fine on /run_sse as long as the run is not streaming."""
  info = create_test_session

  async def run_async_stub(
      self,  # pylint: disable=unused-argument
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
  ):
    del user_id, session_id, invocation_id, new_message, state_delta
    assert run_config.service_tier == "deferred"
    yield Event(
        author="dummy agent",
        invocation_id="invocation_id",
        content=types.Content(role="model", parts=[types.Part(text="hi")]),
    )

  monkeypatch.setattr(Runner, "run_async", run_async_stub)

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": False,
      "service_tier": "deferred",
  }

  response = test_app.post("/run_sse", json=payload)

  assert response.status_code == 200


def test_runtime_config_endpoint_shadows_static_file(tmp_path):
  """The in-memory config must win over the file still shipped in the package.

  ApiServer registers this route before mounting StaticFiles at "/dev-ui/".
  Starlette matches in registration order, so moving the route after the mount
  would silently serve the stale on-disk file instead -- with a 200 and no
  error. Assert on the payload, not just the status code.
  """
  app = get_fast_api_app(
      agents_dir=str(tmp_path), web=True, url_prefix="/custom"
  )

  # The prefix is stripped by whatever mounts the app (reverse proxy, gateway,
  # or an outer Starlette Mount); ADK registers its routes unprefixed.
  outer = Starlette(routes=[Mount("/custom", app)])
  response = TestClient(outer).get(
      "/custom/dev-ui/assets/config/runtime-config.json"
  )

  assert response.status_code == 200
  body = response.json()
  # A stale file on disk would report "" here.
  assert body["backendUrl"] == "/custom"
  assert "telemetry" in body
  assert response.headers["cache-control"] == "no-store"


def test_runtime_config_endpoint_does_not_write_to_disk(tmp_path):
  """Serving the config must not touch the installed package directory."""
  import google.adk.cli as cli_package

  config_path = (
      Path(cli_package.__file__).parent
      / "browser"
      / "assets"
      / "config"
      / "runtime-config.json"
  )
  before = config_path.read_bytes() if config_path.exists() else None

  app = get_fast_api_app(
      agents_dir=str(tmp_path), web=True, url_prefix="/prefix"
  )
  TestClient(app).get("/dev-ui/assets/config/runtime-config.json")

  after = config_path.read_bytes() if config_path.exists() else None
  assert after == before


def test_runtime_config_rejects_half_specified_logo(tmp_path):
  """--logo-text without --logo-image-url is a config error, not a silent drop."""
  with pytest.raises(ValueError, match="Both --logo-text and --logo-image-url"):
    get_fast_api_app(agents_dir=str(tmp_path), web=True, logo_text="ACME")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
