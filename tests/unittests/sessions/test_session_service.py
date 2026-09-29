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
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timezone
import enum
import inspect
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
from unittest import mock

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.features import FeatureName
from google.adk.features import override_feature_enabled
from google.adk.sessions import database_session_service
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.schemas import v0
from google.adk.sessions.schemas import v1
from google.adk.sessions.schemas.shared import DynamicJSON
from google.adk.sessions.schemas.v0 import DynamicPickleType
from google.adk.sessions.schemas.v1 import StorageSession
from google.adk.sessions.session import Session
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types
import pytest
from sqlalchemy import delete


class SessionServiceType(enum.Enum):
    IN_MEMORY = "IN_MEMORY"
    IN_MEMORY_WITH_LIGHT_COPY_ENABLED = "IN_MEMORY_WITH_LIGHT_COPY_ENABLED"
    DATABASE = "DATABASE"
    SQLITE = "SQLITE"


def get_session_service(
    service_type: SessionServiceType = SessionServiceType.IN_MEMORY,
    tmp_path=None,
):
    """Creates a session service for testing."""
    if service_type == SessionServiceType.DATABASE:
        return DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    if service_type == SessionServiceType.SQLITE:
        return SqliteSessionService(str(tmp_path / "sqlite.db"))
    if service_type == SessionServiceType.IN_MEMORY_WITH_LIGHT_COPY_ENABLED:
        return InMemorySessionService()
    return InMemorySessionService()


@pytest.fixture(
    params=[
        SessionServiceType.IN_MEMORY,
        SessionServiceType.IN_MEMORY_WITH_LIGHT_COPY_ENABLED,
        SessionServiceType.DATABASE,
        SessionServiceType.SQLITE,
    ]
)
async def session_service(request, tmp_path):
    """Provides a session service and closes database backends on teardown."""
    if request.param == SessionServiceType.IN_MEMORY_WITH_LIGHT_COPY_ENABLED:
        override_feature_enabled(
            FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY, True
        )
    service = get_session_service(request.param, tmp_path)
    yield service
    if isinstance(service, DatabaseSessionService):
        await service.close()
    if request.param == SessionServiceType.IN_MEMORY_WITH_LIGHT_COPY_ENABLED:
        override_feature_enabled(
            FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY, False
        )


def test_database_session_service_enables_pool_pre_ping_by_default():
    captured_kwargs = {}

    def fake_create_async_engine(_db_url: str, **kwargs):
        captured_kwargs.update(kwargs)
        fake_engine = mock.Mock()
        fake_engine.dialect.name = "postgresql"
        fake_engine.sync_engine = mock.Mock()
        return fake_engine

    with mock.patch.object(
        database_session_service,
        "create_async_engine",
        side_effect=fake_create_async_engine,
    ):
        database_session_service.DatabaseSessionService(
            "postgresql+psycopg2://user:pass@localhost:5432/db"
        )

    assert captured_kwargs.get("pool_pre_ping") is True


def test_database_session_service_disables_pool_reset_on_return_for_static_pool():
  """StaticPool shares a single connection, so reset_on_return must be disabled."""
  captured_kwargs = {}

  def fake_create_async_engine(_db_url: str, **kwargs):
    captured_kwargs.update(kwargs)
    fake_engine = mock.Mock()
    fake_engine.dialect.name = 'sqlite'
    fake_engine.sync_engine = mock.Mock()
    return fake_engine

  with (
      mock.patch.object(
          database_session_service,
          'create_async_engine',
          side_effect=fake_create_async_engine,
      ),
      mock.patch.object(database_session_service.event, 'listen'),
  ):
    database_session_service.DatabaseSessionService(
        'sqlite+aiosqlite:///:memory:'
    )

  assert captured_kwargs.get('poolclass') is StaticPool
  assert captured_kwargs.get('pool_reset_on_return') is None


def test_database_session_service_respects_custom_pool_reset_on_return_for_static_pool():
  """Explicit pool_reset_on_return is respected even when StaticPool is used."""
  captured_kwargs = {}

  def fake_create_async_engine(_db_url: str, **kwargs):
    captured_kwargs.update(kwargs)
    fake_engine = mock.Mock()
    fake_engine.dialect.name = 'sqlite'
    fake_engine.sync_engine = mock.Mock()
    return fake_engine

  with (
      mock.patch.object(
          database_session_service,
          'create_async_engine',
          side_effect=fake_create_async_engine,
      ),
      mock.patch.object(database_session_service.event, 'listen'),
  ):
    database_session_service.DatabaseSessionService(
        'sqlite+aiosqlite:///:memory:',
        pool_reset_on_return='commit',
    )

  assert captured_kwargs.get('pool_reset_on_return') == 'commit'


@pytest.mark.parametrize('decorator', [DynamicJSON, DynamicPickleType])
def test_session_type_decorators_opt_into_statement_cache(decorator):
  """Session TypeDecorators must declare cache_ok to stay cacheable.

    PostgreSQL's default TIMESTAMP type is WITHOUT TIME ZONE, which cannot
    accept timezone-aware datetime objects when using asyncpg. SQLite also
    requires naive datetimes.
    """
    # Simulate the logic in create_session
    is_sqlite = dialect_name == "sqlite"
    is_postgres = dialect_name == "postgresql"

    now = datetime.now(timezone.utc)
    assert now.tzinfo is not None  # Starts with timezone

    if is_sqlite or is_postgres:
        now = now.replace(tzinfo=None)

    # Both SQLite and PostgreSQL should have timezone stripped
    assert now.tzinfo is None


def test_database_session_service_preserves_timezone_for_other_dialects():
    """Verifies that timezone info is preserved for dialects that support it."""
    # For dialects like MySQL with explicit timezone support, we don't strip
    dialect_name = "mysql"
    is_sqlite = dialect_name == "sqlite"
    is_postgres = dialect_name == "postgresql"

    now = datetime.now(timezone.utc)
    assert now.tzinfo is not None

    if is_sqlite or is_postgres:
        now = now.replace(tzinfo=None)

    # MySQL should preserve timezone (if the column type supports it)
    assert now.tzinfo is not None


def test_database_session_service_respects_pool_pre_ping_override():
    captured_kwargs = {}

    def fake_create_async_engine(_db_url: str, **kwargs):
        captured_kwargs.update(kwargs)
        fake_engine = mock.Mock()
        fake_engine.dialect.name = "postgresql"
        fake_engine.sync_engine = mock.Mock()
        return fake_engine

    with mock.patch.object(
        database_session_service,
        "create_async_engine",
        side_effect=fake_create_async_engine,
    ):
        database_session_service.DatabaseSessionService(
            "postgresql+psycopg2://user:pass@localhost:5432/db",
            pool_pre_ping=False,
        )

    assert captured_kwargs.get("pool_pre_ping") is False


def test_database_session_service_creates_read_only_engine_for_spanner():
    captured_binds = []
    fake_engine = mock.Mock()
    fake_engine.dialect.name = "spanner+spanner"
    fake_engine.sync_engine = mock.Mock()
    read_only_engine = mock.Mock()
    fake_engine.execution_options.return_value = read_only_engine

    def fake_async_sessionmaker(*, bind, expire_on_commit, **kwargs):
        del expire_on_commit
        del kwargs
        captured_binds.append(bind)
        return mock.Mock()

    with (
        mock.patch.object(
            database_session_service,
            "create_async_engine",
            return_value=fake_engine,
        ),
        mock.patch.object(
            database_session_service,
            "async_sessionmaker",
            side_effect=fake_async_sessionmaker,
        ),
    ):
        database_session_service.DatabaseSessionService(
            "spanner+spanner:///projects/test/instances/test/databases/test"
        )

    assert captured_binds == [fake_engine, read_only_engine]
    fake_engine.execution_options.assert_called_once_with(read_only=True)


def test_database_session_service_creates_read_only_engine_for_other_dialects():
    captured_binds = []
    fake_engine = mock.Mock()
    fake_engine.dialect.name = "postgresql"
    fake_engine.sync_engine = mock.Mock()
    read_only_engine = mock.Mock()
    fake_engine.execution_options.return_value = read_only_engine

    def fake_async_sessionmaker(*, bind, expire_on_commit, **kwargs):
        del expire_on_commit
        del kwargs
        captured_binds.append(bind)
        return mock.Mock()

    with (
        mock.patch.object(
            database_session_service,
            "create_async_engine",
            return_value=fake_engine,
        ),
        mock.patch.object(
            database_session_service,
            "async_sessionmaker",
            side_effect=fake_async_sessionmaker,
        ),
    ):
        database_session_service.DatabaseSessionService(
            "postgresql+psycopg2://user:pass@localhost:5432/db"
        )

    assert captured_binds == [fake_engine, read_only_engine]
    fake_engine.execution_options.assert_called_once_with(read_only=True)


@pytest.mark.asyncio
async def test_sqlite_session_service_accepts_sqlite_urls(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    service = SqliteSessionService("sqlite+aiosqlite:///./sessions.db")
    await service.create_session(app_name="app", user_id="user")
    assert (tmp_path / "sessions.db").exists()

    service = SqliteSessionService("sqlite:///./sessions2.db")
    await service.create_session(app_name="app", user_id="user")
    assert (tmp_path / "sessions2.db").exists()


@pytest.mark.asyncio
async def test_sqlite_session_service_preserves_uri_query_parameters(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "readonly.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER)")
        conn.commit()

    service = SqliteSessionService(f"sqlite+aiosqlite:///{db_path}?mode=ro")
    # `mode=ro` opens the DB read-only; schema creation should fail.
    with pytest.raises(sqlite3.OperationalError, match=r"readonly"):
        await service.create_session(app_name="app", user_id="user")


@pytest.mark.asyncio
async def test_sqlite_session_service_accepts_absolute_sqlite_urls(tmp_path):
    abs_db_path = tmp_path / "absolute.db"
    abs_url = "sqlite+aiosqlite:////" + str(abs_db_path).lstrip("/")
    service = SqliteSessionService(abs_url)
    await service.create_session(app_name="app", user_id="user")
    assert abs_db_path.exists()


@pytest.mark.asyncio
async def test_get_empty_session(session_service):
    assert not await session_service.get_session(
        app_name="my_app", user_id="test_user", session_id="123"
    )


@pytest.mark.asyncio
async def test_database_session_service_get_session_uses_read_only_factory():
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    service._prepare_tables = mock.AsyncMock()

    read_only_session = mock.AsyncMock()
    read_only_session.get = mock.AsyncMock(return_value=None)

    @asynccontextmanager
    async def fake_read_only_session():
        yield read_only_session

    service.database_session_factory = mock.Mock(
        side_effect=AssertionError("write session factory should not be used")
    )
    service._read_only_database_session_factory = mock.Mock(
        return_value=fake_read_only_session()
    )

    session = await service.get_session(
        app_name="my_app", user_id="test_user", session_id="123"
    )

    assert session is None
    service._read_only_database_session_factory.assert_called_once_with()
    service.database_session_factory.assert_not_called()

    await service.close()


@pytest.mark.asyncio
async def test_database_session_service_list_sessions_uses_read_only_factory():
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    service._prepare_tables = mock.AsyncMock()

    read_only_session = mock.AsyncMock()
    empty_result = mock.Mock()
    empty_result.scalars.return_value.all.return_value = []
    read_only_session.execute = mock.AsyncMock(return_value=empty_result)
    read_only_session.get = mock.AsyncMock(return_value=None)

    @asynccontextmanager
    async def fake_read_only_session():
        yield read_only_session

    service.database_session_factory = mock.Mock(
        side_effect=AssertionError("write session factory should not be used")
    )
    service._read_only_database_session_factory = mock.Mock(
        return_value=fake_read_only_session()
    )

    response = await service.list_sessions(
        app_name="my_app", user_id="test_user"
    )

    assert response.sessions == []
    service._read_only_database_session_factory.assert_called_once_with()
    service.database_session_factory.assert_not_called()

    await service.close()


@pytest.mark.asyncio
async def test_create_get_session(session_service):
    app_name = "my_app"
    user_id = "test_user"
    state = {"key": "value"}

    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, state=state
    )
    assert session.app_name == app_name
    assert session.user_id == user_id
    assert session.id
    assert session.state == state
    assert (
        session.last_update_time
        <= datetime.now().astimezone(timezone.utc).timestamp()
    )

    got_session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session.id
    )
    assert got_session == session
    assert (
        got_session.last_update_time
        <= datetime.now().astimezone(timezone.utc).timestamp()
    )

    session_id = session.id
    await session_service.delete_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    assert (
        await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        )
        is None
    )


@pytest.mark.asyncio
async def test_create_and_list_sessions(session_service):
    app_name = "my_app"
    user_id = "test_user"

    session_ids = ["session" + str(i) for i in range(5)]
    for session_id in session_ids:
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state={"key": "value" + session_id},
        )

    list_sessions_response = await session_service.list_sessions(
        app_name=app_name, user_id=user_id
    )
    sessions = list_sessions_response.sessions
    assert len(sessions) == len(session_ids)
    assert {s.id for s in sessions} == set(session_ids)
    for session in sessions:
        assert session.state == {"key": "value" + session.id}


@pytest.mark.asyncio
async def test_list_sessions_all_users(session_service):
    app_name = "my_app"
    user_id_1 = "user1"
    user_id_2 = "user2"

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id_1,
        session_id="session1a",
        state={"key": "value1a"},
    )
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id_1,
        session_id="session1b",
        state={"key": "value1b"},
    )
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id_2,
        session_id="session2a",
        state={"key": "value2a"},
    )

    # List sessions for user1 - should contain merged state
    list_sessions_response_1 = await session_service.list_sessions(
        app_name=app_name, user_id=user_id_1
    )
    sessions_1 = list_sessions_response_1.sessions
    assert len(sessions_1) == 2
    sessions_1_map = {s.id: s for s in sessions_1}
    assert sessions_1_map["session1a"].state == {"key": "value1a"}
    assert sessions_1_map["session1b"].state == {"key": "value1b"}

    # List sessions for user2 - should contain merged state
    list_sessions_response_2 = await session_service.list_sessions(
        app_name=app_name, user_id=user_id_2
    )
    sessions_2 = list_sessions_response_2.sessions
    assert len(sessions_2) == 1
    assert sessions_2[0].id == "session2a"
    assert sessions_2[0].state == {"key": "value2a"}

    # List sessions for all users - should contain merged state
    list_sessions_response_all = await session_service.list_sessions(
        app_name=app_name, user_id=None
    )
    sessions_all = list_sessions_response_all.sessions
    assert len(sessions_all) == 3
    sessions_all_map = {s.id: s for s in sessions_all}
    assert sessions_all_map["session1a"].state == {"key": "value1a"}
    assert sessions_all_map["session1b"].state == {"key": "value1b"}
    assert sessions_all_map["session2a"].state == {"key": "value2a"}


@pytest.mark.asyncio
async def test_app_state_is_shared_by_all_users_of_app(session_service):
    app_name = "my_app"
    # User 1 creates a session, establishing app:k1
    session1 = await session_service.create_session(
        app_name=app_name,
        user_id="u1",
        session_id="s1",
        state={"app:k1": "v1"},
    )
    # User 1 appends an event to session1, establishing app:k2
    event = Event(
        invocation_id="inv1",
        author="user",
        actions=EventActions(state_delta={"app:k2": "v2"}),
    )
    await session_service.append_event(session=session1, event=event)

    # User 2 creates a new session session2, it should see app:k1 and app:k2
    session2 = await session_service.create_session(
        app_name=app_name, user_id="u2", session_id="s2"
    )
    assert session2.state == {"app:k1": "v1", "app:k2": "v2"}

    # If we get session session1 again, it should also see both
    session1_got = await session_service.get_session(
        app_name=app_name, user_id="u1", session_id="s1"
    )
    assert session1_got.state.get("app:k1") == "v1"
    assert session1_got.state.get("app:k2") == "v2"


@pytest.mark.asyncio
async def test_user_state_is_shared_only_by_user_sessions(session_service):
    app_name = "my_app"
    # User 1 creates a session, establishing user:k1 for user 1
    session1 = await session_service.create_session(
        app_name=app_name,
        user_id="u1",
        session_id="s1",
        state={"user:k1": "v1"},
    )
    # User 1 appends an event to session1, establishing user:k2 for user 1
    event = Event(
        invocation_id="inv1",
        author="user",
        actions=EventActions(state_delta={"user:k2": "v2"}),
    )
    await session_service.append_event(session=session1, event=event)

    # Another session for User 1 should see user:k1 and user:k2
    session1b = await session_service.create_session(
        app_name=app_name, user_id="u1", session_id="s1b"
    )
    assert session1b.state == {"user:k1": "v1", "user:k2": "v2"}

    # A session for User 2 should NOT see user:k1 or user:k2
    session2 = await session_service.create_session(
        app_name=app_name, user_id="u2", session_id="s2"
    )
    assert session2.state == {}


@pytest.mark.asyncio
async def test_session_state_is_not_shared(session_service):
    app_name = "my_app"
    # User 1 creates a session session1, establishing sk1 only for session1
    session1 = await session_service.create_session(
        app_name=app_name, user_id="u1", session_id="s1", state={"sk1": "v1"}
    )
    # User 1 appends an event to session1, establishing sk2 only for session1
    event = Event(
        invocation_id="inv1",
        author="user",
        actions=EventActions(state_delta={"sk2": "v2"}),
    )
    await session_service.append_event(session=session1, event=event)

    # Getting session1 should show sk1 and sk2
    session1_got = await session_service.get_session(
        app_name=app_name, user_id="u1", session_id="s1"
    )
    assert session1_got.state.get("sk1") == "v1"
    assert session1_got.state.get("sk2") == "v2"

    # Creating another session session1b for User 1 should NOT see sk1 or sk2
    session1b = await session_service.create_session(
        app_name=app_name, user_id="u1", session_id="s1b"
    )
    assert session1b.state == {}


@pytest.mark.asyncio
async def test_temp_state_is_not_persisted_in_state_or_events(session_service):
    app_name = "my_app"
    user_id = "u1"
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id="s1"
    )
    event = Event(
        invocation_id="inv1",
        author="user",
        actions=EventActions(state_delta={"temp:k1": "v1", "sk": "v2"}),
    )
    await session_service.append_event(session=session, event=event)

    # Temp state IS available in the in-memory session (same invocation)
    assert session.state.get("temp:k1") == "v1"
    assert session.state.get("sk") == "v2"

    # Check event as stored in session does not contain temp keys in state_delta
    assert "temp:k1" not in event.actions.state_delta
    assert event.actions.state_delta.get("sk") == "v2"


@pytest.mark.asyncio
async def test_temp_state_visible_across_sequential_events(session_service):
    """Temp state set by one event should be readable before the next event.

    This simulates a SequentialAgent where agent-1 writes output_key='temp:out'
    and agent-2 needs to read it from session.state within the same invocation.
    """
    app_name = "my_app"
    user_id = "u1"
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id="s_seq"
    )

    # Agent-1 writes temp state
    event1 = Event(
        invocation_id="inv1",
        author="agent1",
        actions=EventActions(state_delta={"temp:output": "result_from_a1"}),
    )
    await session_service.append_event(session=session, event=event1)

    # Agent-2 should be able to read temp state from the same session object
    assert session.state.get("temp:output") == "result_from_a1"

    # But the event delta should NOT contain the temp key (not persisted)
    assert "temp:output" not in event1.actions.state_delta


@pytest.mark.asyncio
async def test_get_session_respects_user_id(session_service):
    app_name = "my_app"
    # u1 creates session 's1' and adds an event
    session1 = await session_service.create_session(
        app_name=app_name, user_id="u1", session_id="s1"
    )
    event = Event(invocation_id="inv1", author="user")
    await session_service.append_event(session1, event)
    # u2 creates a session with the same session_id 's1'
    await session_service.create_session(
        app_name=app_name, user_id="u2", session_id="s1"
    )
    # Check that getting s1 for u2 returns u2's session (with no events)
    # not u1's session.
    session2_got = await session_service.get_session(
        app_name=app_name, user_id="u2", session_id="s1"
    )
    assert session2_got.user_id == "u2"
    assert len(session2_got.events) == 0


@pytest.mark.asyncio
async def test_create_session_with_existing_id_raises_error(session_service):
    app_name = "my_app"
    user_id = "test_user"
    session_id = "existing_session"

    # Create the first session
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # Attempt to create a session with the same ID
    with pytest.raises(AlreadyExistsError):
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )


@pytest.mark.asyncio
async def test_append_event_bytes(session_service):
    app_name = "my_app"
    user_id = "user"

    session = await session_service.create_session(
        app_name=app_name, user_id=user_id
    )

    test_content = types.Content(
        role="user",
        parts=[
            types.Part.from_bytes(
                data=b"test_image_data", mime_type="image/png"
            ),
        ],
    )
    test_grounding_metadata = types.GroundingMetadata(
        search_entry_point=types.SearchEntryPoint(sdk_blob=b"test_sdk_blob")
    )
    event = Event(
        invocation_id="invocation",
        author="user",
        content=test_content,
        grounding_metadata=test_grounding_metadata,
    )
    await session_service.append_event(session=session, event=event)

    assert session.events[0].content == test_content

    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session.id
    )
    events = session.events
    assert len(events) == 1
    assert events[0].content == test_content
    assert events[0].grounding_metadata == test_grounding_metadata


@pytest.mark.asyncio
async def test_append_event_complete(session_service):
    app_name = "my_app"
    user_id = "user"

    session = await session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    event = Event(
        invocation_id="invocation",
        author="user",
        content=types.Content(
            role="user", parts=[types.Part(text="test_text")]
        ),
        turn_complete=True,
        partial=False,
        actions=EventActions(
            artifact_delta={
                "file": 0,
            },
            transfer_to_agent="agent",
            escalate=True,
        ),
        long_running_tool_ids={"tool1"},
        error_code="error_code",
        error_message="error_message",
        interrupted=True,
        grounding_metadata=types.GroundingMetadata(
            web_search_queries=["query1"],
        ),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1, candidates_token_count=1, total_token_count=2
        ),
        citation_metadata=types.CitationMetadata(),
        custom_metadata={"custom_key": "custom_value"},
        timestamp=1700000000.123,
        input_transcription=types.Transcription(
            text="input transcription",
            finished=True,
        ),
        output_transcription=types.Transcription(
            text="output transcription",
            finished=True,
        ),
    )
    await session_service.append_event(session=session, event=event)

    assert (
        await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        )
        == session
    )


@pytest.mark.asyncio
async def test_session_last_update_time_updates_on_event(session_service):
    app_name = "my_app"
    user_id = "user"

    session = await session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    original_update_time = session.last_update_time

    event_timestamp = original_update_time + 10
    event = Event(
        invocation_id="invocation",
        author="user",
        timestamp=event_timestamp,
    )
    await session_service.append_event(session=session, event=event)

    assert session.last_update_time == pytest.approx(event_timestamp, abs=1e-6)

    refreshed_session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session.id
    )
    assert refreshed_session is not None
    assert refreshed_session.last_update_time == pytest.approx(
        event_timestamp, abs=1e-6
    )
    assert refreshed_session.last_update_time > original_update_time


@pytest.mark.asyncio
async def test_append_event_to_stale_session():
  session_service = get_session_service(
      service_type=SessionServiceType.DATABASE
  )

  async with session_service:
    app_name = 'my_app'
    user_id = 'user'
    current_time = datetime.now().astimezone(timezone.utc).timestamp()

    original_session = await session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    event1 = Event(
        invocation_id='inv1',
        author='user',
        timestamp=current_time + 1,
        actions=EventActions(state_delta={'sk1': 'v1'}),
    )
    await session_service.append_event(original_session, event1)

    updated_session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=original_session.id
    )
    event2 = Event(
        invocation_id='inv2',
        author='user',
        timestamp=current_time + 2,
        actions=EventActions(state_delta={'sk2': 'v2'}),
    )
    await session_service.append_event(updated_session, event2)

    # original_session is now stale
    assert original_session.last_update_time < updated_session.last_update_time
    assert len(original_session.events) == 1
    assert 'sk2' not in original_session.state

    # Appending another event to stale original_session should be rejected.
    event3 = Event(
        invocation_id='inv3',
        author='user',
        timestamp=current_time + 3,
        actions=EventActions(state_delta={'sk3': 'v3'}),
    )
    with pytest.raises(StaleSessionError, match='modified in storage'):
      await session_service.append_event(original_session, event3)

    # If we fetch session from DB, it should only contain the committed events.
    session_final = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=original_session.id
    )
    assert len(session_final.events) == 2
    assert session_final.state.get('sk1') == 'v1'
    assert session_final.state.get('sk2') == 'v2'
    assert session_final.state.get('sk3') is None
    assert [e.invocation_id for e in session_final.events] == [
        'inv1',
        'inv2',
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('schema_module', [v0, v1], ids=['v0', 'v1'])
async def test_append_event_same_timestamp_single_writer_not_stale(
    schema_module: Any, tmp_path: Path
) -> None:
  """A single writer must not be rejected when two events share a timestamp.

  Regression test: `StorageSession.update_time` used to be declared with
  `onupdate=func.now()`. When an event's timestamp equalled the value
  already stored, SQLAlchemy saw no change to that column and omitted it
  from the UPDATE, so `onupdate` fired and wrote the database's own clock
  instead. The in-memory revision marker (read before commit) then no
  longer matched storage, and the next append from the same, only, writer
  was incorrectly rejected as stale.
  """
  db_path = tmp_path / f'{schema_module.__name__}.db'
  db_url = f'sqlite+aiosqlite:///{db_path}'
  if schema_module is v0:
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
      await conn.run_sync(v0.Base.metadata.create_all)
    await engine.dispose()

  session_service = DatabaseSessionService(db_url)

  async with session_service:
    app_name = 'my_app'
    user_id = 'user'
    same_timestamp = datetime.now().astimezone(timezone.utc).timestamp()

    session = await session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    event1 = Event(
        invocation_id='inv1',
        author='user',
        timestamp=same_timestamp,
    )
    await session_service.append_event(session, event1)

    # Same timestamp as the previous event, with a state change so the
    # UPDATE statement still runs for other columns.
    event2 = Event(
        invocation_id='inv2',
        author='user',
        timestamp=same_timestamp,
        actions=EventActions(state_delta={'sk1': 'v1'}),
    )
    await session_service.append_event(session, event2)

    event3 = Event(
        invocation_id='inv3',
        author='user',
        timestamp=same_timestamp + 1,
    )
    # The same writer appending a third event must not be rejected.
    await session_service.append_event(session, event3)

    assert len(session.events) == 3


@pytest.mark.asyncio
async def test_sqlite_append_event_uses_typed_stale_session_error(tmp_path):
  """The legacy SQLite backend exposes the shared stale-writer contract."""
  service = get_session_service(SessionServiceType.SQLITE, tmp_path)
  session = await service.create_session(app_name='app', user_id='user')
  stale_session = session.model_copy(deep=True)

  await service.append_event(
      session,
      Event(
          invocation_id='winner',
          author='user',
          timestamp=session.last_update_time + 1,
      ),
  )

  with pytest.raises(StaleSessionError) as error:
    await service.append_event(
        stale_session,
        Event(
            invocation_id='stale',
            author='user',
            timestamp=session.last_update_time + 1,
        ),
    )

    async with session_service:
        app_name = "my_app"
        user_id = "user"
        current_time = datetime.now().astimezone(timezone.utc).timestamp()

        original_session = await session_service.create_session(
            app_name=app_name, user_id=user_id
        )
        event1 = Event(
            invocation_id="inv1",
            author="user",
            timestamp=current_time + 1,
            actions=EventActions(state_delta={"sk1": "v1"}),
        )
        await session_service.append_event(original_session, event1)

        updated_session = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=original_session.id
        )
        event2 = Event(
            invocation_id="inv2",
            author="user",
            timestamp=current_time + 2,
            actions=EventActions(state_delta={"sk2": "v2"}),
        )
        await session_service.append_event(updated_session, event2)

        # original_session is now stale
        assert (
            original_session.last_update_time
            < updated_session.last_update_time
        )
        assert len(original_session.events) == 1
        assert "sk2" not in original_session.state

        # Appending another event to stale original_session should be rejected.
        event3 = Event(
            invocation_id="inv3",
            author="user",
            timestamp=current_time + 3,
            actions=EventActions(state_delta={"sk3": "v3"}),
        )
        with pytest.raises(ValueError, match="modified in storage"):
            await session_service.append_event(original_session, event3)

        # If we fetch session from DB, it should only contain the committed events.
        session_final = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=original_session.id
        )
        assert len(session_final.events) == 2
        assert session_final.state.get("sk1") == "v1"
        assert session_final.state.get("sk2") == "v2"
        assert session_final.state.get("sk3") is None
        assert [e.invocation_id for e in session_final.events] == [
            "inv1",
            "inv2",
        ]


@pytest.mark.asyncio
async def test_append_event_raises_if_app_state_row_missing():
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        session = await service.create_session(
            app_name="my_app", user_id="user", session_id="s1"
        )
        schema = service._get_schema_classes()
        async with service.database_session_factory() as sql_session:
            await sql_session.execute(
                delete(schema.StorageAppState).where(
                    schema.StorageAppState.app_name == session.app_name
                )
            )
            await sql_session.commit()

        event = Event(
            invocation_id="inv1",
            author="user",
            actions=EventActions(state_delta={"k": "v"}),
        )
        with pytest.raises(ValueError, match="App state missing"):
            await service.append_event(session, event)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_append_event_raises_if_user_state_row_missing():
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        session = await service.create_session(
            app_name="my_app", user_id="user", session_id="s1"
        )
        schema = service._get_schema_classes()
        async with service.database_session_factory() as sql_session:
            await sql_session.execute(
                delete(schema.StorageUserState).where(
                    schema.StorageUserState.app_name == session.app_name,
                    schema.StorageUserState.user_id == session.user_id,
                )
            )
            await sql_session.commit()

        event = Event(
            invocation_id="inv1",
            author="user",
            actions=EventActions(state_delta={"k": "v"}),
        )
        with pytest.raises(ValueError, match="User state missing"):
            await service.append_event(session, event)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_append_event_concurrent_stale_sessions_reject_stale_writer():
    session_service = get_session_service(
        service_type=SessionServiceType.DATABASE
    )

    async with session_service:
        app_name = "my_app"
        user_id = "user"
        session = await session_service.create_session(
            app_name=app_name, user_id=user_id
        )

        iteration_count = 8
        for i in range(iteration_count):
            latest_session = await session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session.id
            )
            stale_session_1 = latest_session.model_copy(deep=True)
            stale_session_2 = latest_session.model_copy(deep=True)
            base_timestamp = latest_session.last_update_time + 10.0
            event_1 = Event(
                invocation_id=f"inv{i}-1",
                author="user",
                timestamp=base_timestamp + 1.0,
                actions=EventActions(state_delta={f"sk{i}-1": f"v{i}-1"}),
            )
            event_2 = Event(
                invocation_id=f"inv{i}-2",
                author="user",
                timestamp=base_timestamp + 2.0,
                actions=EventActions(state_delta={f"sk{i}-2": f"v{i}-2"}),
            )

            results = await asyncio.gather(
                session_service.append_event(stale_session_1, event_1),
                session_service.append_event(stale_session_2, event_2),
                return_exceptions=True,
            )
            errors = [
                result for result in results if isinstance(result, Exception)
            ]
            successes = [
                result
                for result in results
                if not isinstance(result, Exception)
            ]
            assert len(successes) == 1
            assert len(errors) == 1
            assert isinstance(errors[0], ValueError)
            assert "modified in storage" in str(errors[0])

        session_final = await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        )

        for i in range(iteration_count):
            event_values = {
                session_final.state.get(f"sk{i}-1"),
                session_final.state.get(f"sk{i}-2"),
            }
            assert event_values & {f"v{i}-1", f"v{i}-2"}
            assert None in event_values
        assert len(session_final.events) == iteration_count


@pytest.mark.asyncio
async def test_append_event_allows_timestamp_drift_for_current_session():
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        session = await service.create_session(
            app_name="my_app", user_id="user", session_id="s1"
        )
        event1 = Event(
            invocation_id="inv1",
            author="user",
            timestamp=session.last_update_time + 10,
        )
        await service.append_event(session, event1)

        # Simulate a float round-trip mismatch without changing the persisted
        # session revision.
        session.last_update_time -= 0.0001

        event2 = Event(
            invocation_id="inv2",
            author="user",
            timestamp=event1.timestamp + 10,
        )
        await service.append_event(session, event2)

        refreshed_session = await service.get_session(
            app_name="my_app", user_id="user", session_id=session.id
        )
        assert [event.invocation_id for event in refreshed_session.events] == [
            "inv1",
            "inv2",
        ]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_append_event_allows_markerless_current_session():
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        session = await service.create_session(
            app_name="my_app", user_id="user", session_id="s1"
        )
        event1 = Event(
            invocation_id="inv1",
            author="user",
            timestamp=session.last_update_time + 10,
        )
        await service.append_event(session, event1)

        session._storage_update_marker = None
        session.last_update_time -= 0.0001

        event2 = Event(
            invocation_id="inv2",
            author="user",
            timestamp=event1.timestamp + 10,
        )
        await service.append_event(session, event2)

        refreshed_session = await service.get_session(
            app_name="my_app", user_id="user", session_id=session.id
        )
        assert [event.invocation_id for event in refreshed_session.events] == [
            "inv1",
            "inv2",
        ]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_get_session_with_config(session_service):
    app_name = "my_app"
    user_id = "user"

    num_test_events = 5
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    for i in range(1, num_test_events + 1):
        event = Event(author="user", timestamp=i)
        await session_service.append_event(session, event)

    # No config, expect all events to be returned.
    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session.id
    )
    events = session.events
    assert len(events) == num_test_events

    # Only expect the most recent 3 events.
    num_recent_events = 3
    config = GetSessionConfig(num_recent_events=num_recent_events)
    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session.id,
        config=config,
    )
    events = session.events
    assert len(events) == num_recent_events
    assert events[0].timestamp == num_test_events - num_recent_events + 1

    # Only expect events after timestamp 4.0 (inclusive), i.e., 2 events.
    after_timestamp = 4.0
    config = GetSessionConfig(after_timestamp=after_timestamp)
    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session.id,
        config=config,
    )
    events = session.events
    assert len(events) == num_test_events - after_timestamp + 1
    assert events[0].timestamp == after_timestamp

    # Expect no events if none are > after_timestamp.
    way_after_timestamp = num_test_events * 10
    config = GetSessionConfig(after_timestamp=way_after_timestamp)
    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session.id,
        config=config,
    )
    assert not session.events

    # Both filters applied, i.e., of 3 most recent events, only 2 are after
    # timestamp 4.0, so expect 2 events.
    config = GetSessionConfig(
        after_timestamp=after_timestamp, num_recent_events=num_recent_events
    )
    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session.id,
        config=config,
    )
    events = session.events
    assert len(events) == num_test_events - after_timestamp + 1


@pytest.mark.asyncio
async def test_partial_events_are_not_persisted(session_service):
    app_name = "my_app"
    user_id = "user"
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    event = Event(author="user", partial=True)
    await session_service.append_event(session, event)

    # Check in-memory session
    assert len(session.events) == 0
    # Check persisted session
    session_got = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session.id
    )
    assert len(session_got.events) == 0


# ---------------------------------------------------------------------------
# Rollback tests – verify _rollback_on_exception_session explicitly rolls back
# on errors
# ---------------------------------------------------------------------------
class _RollbackSpySession:
    """Wraps an AsyncSession to spy on rollback() and optionally fail commit()."""

    def __init__(self, real_session, *, fail_commit=False):
        self._real = real_session
        self._fail_commit = fail_commit
        self.rollback_called = False

    async def __aenter__(self):
        self._real = await self._real.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self._real.__aexit__(*args)

    async def commit(self):
        if self._fail_commit:
            raise RuntimeError("simulated commit failure")
        return await self._real.commit()

    async def rollback(self):
        self.rollback_called = True
        return await self._real.rollback()

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.mark.asyncio
async def test_create_session_calls_rollback_on_commit_failure():
    """Verifies that a commit failure during create_session triggers an explicit
    rollback() call via _rollback_on_exception_session, not just a close()."""
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        # Ensure tables are initialized.
        await service.create_session(
            app_name="app", user_id="user", session_id="good"
        )

        original_factory = service.database_session_factory
        spy_sessions = []

        def _spy_factory():
            spy = _RollbackSpySession(original_factory(), fail_commit=True)
            spy_sessions.append(spy)
            return spy

        service.database_session_factory = _spy_factory

        with pytest.raises(RuntimeError, match="simulated commit failure"):
            await service.create_session(
                app_name="app", user_id="user", session_id="should_fail"
            )

        # The key assertion: rollback() must have been called explicitly.
        assert len(spy_sessions) == 1
        assert spy_sessions[0].rollback_called, (
            "rollback() was not called – _rollback_on_exception_session is not"
            " protecting this path"
        )

        # Restore and verify the failed session was not persisted.
        service.database_session_factory = original_factory
        assert (
            await service.get_session(
                app_name="app", user_id="user", session_id="should_fail"
            )
            is None
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_append_event_calls_rollback_on_commit_failure():
    """Verifies that a commit failure during append_event triggers an explicit
    rollback() call via _rollback_on_exception_session."""
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        session = await service.create_session(
            app_name="app", user_id="user", session_id="s1"
        )

        # Successfully append one event first.
        event1 = Event(
            invocation_id="inv1",
            author="user",
            actions=EventActions(state_delta={"key1": "value1"}),
        )
        await service.append_event(session, event1)

        original_factory = service.database_session_factory
        spy_sessions = []

        def _spy_factory():
            spy = _RollbackSpySession(original_factory(), fail_commit=True)
            spy_sessions.append(spy)
            return spy

        service.database_session_factory = _spy_factory

        event2 = Event(
            invocation_id="inv2",
            author="user",
            actions=EventActions(state_delta={"key2": "value2"}),
        )
        with pytest.raises(RuntimeError, match="simulated commit failure"):
            await service.append_event(session, event2)

        assert len(spy_sessions) == 1
        assert spy_sessions[0].rollback_called, (
            "rollback() was not called – _rollback_on_exception_session is not"
            " protecting this path"
        )

        # Restore and verify only the first event was persisted.
        service.database_session_factory = original_factory
        got = await service.get_session(
            app_name="app", user_id="user", session_id="s1"
        )
        assert len(got.events) == 1
        assert got.events[0].invocation_id == "inv1"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_delete_session_calls_rollback_on_commit_failure():
    """Verifies that a commit failure during delete_session triggers an explicit
    rollback() call via _rollback_on_exception_session."""
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        await service.create_session(
            app_name="app", user_id="user", session_id="s1"
        )

        original_factory = service.database_session_factory
        spy_sessions = []

        def _spy_factory():
            spy = _RollbackSpySession(original_factory(), fail_commit=True)
            spy_sessions.append(spy)
            return spy

        service.database_session_factory = _spy_factory

        with pytest.raises(RuntimeError, match="simulated commit failure"):
            await service.delete_session(
                app_name="app", user_id="user", session_id="s1"
            )

        assert len(spy_sessions) == 1
        assert spy_sessions[0].rollback_called, (
            "rollback() was not called – _rollback_on_exception_session is not"
            " protecting this path"
        )

        # Restore and verify the session still exists (delete was rolled back).
        service.database_session_factory = original_factory
        got = await service.get_session(
            app_name="app", user_id="user", session_id="s1"
        )
        assert got is not None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_service_recovers_after_multiple_failures():
    """After several consecutive commit failures, every single one must trigger
    a rollback() call and the service must remain functional afterward."""
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        await service.create_session(
            app_name="app", user_id="user", session_id="seed"
        )

        original_factory = service.database_session_factory
        spy_sessions = []

        def _spy_factory():
            spy = _RollbackSpySession(original_factory(), fail_commit=True)
            spy_sessions.append(spy)
            return spy

        service.database_session_factory = _spy_factory

        num_failures = 5
        for i in range(num_failures):
            with pytest.raises(RuntimeError, match="simulated commit failure"):
                await service.create_session(
                    app_name="app", user_id="user", session_id=f"fail_{i}"
                )

        # Every failure must have triggered a rollback.
        assert len(spy_sessions) == num_failures
        for i, spy in enumerate(spy_sessions):
            assert (
                spy.rollback_called
            ), f"rollback() was not called on failure #{i}"

        # Restore and verify the service is still healthy.
        service.database_session_factory = original_factory
        session = await service.create_session(
            app_name="app", user_id="user", session_id="recovered"
        )
        assert session.id == "recovered"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_concurrent_prepare_tables_no_race_condition():
    """Verifies that concurrent calls to _prepare_tables wait for table creation.
    Reproduces the race condition from
    https://github.com/google/adk-python/issues/4445: when concurrent requests
    arrive at startup, _prepare_tables must not return before tables exist.
    Previously, the early-return guard checked _db_schema_version (set during
    schema detection) instead of _tables_created, so a second request could
    slip through after schema detection but before table creation finished.
    """
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        # Tables haven't been created yet.
        assert not service._tables_created
        assert service._db_schema_version is None

        # Launch several concurrent create_session calls, each with a unique
        # app_name to avoid IntegrityError on the shared app_states row.
        # Each will call _prepare_tables internally.  If the race condition
        # exists, some of these will fail because the "sessions" table doesn't
        # exist yet.
        num_concurrent = 5
        results = await asyncio.gather(
            *[
                service.create_session(
                    app_name=f"app_{i}", user_id="user", session_id=f"sess_{i}"
                )
                for i in range(num_concurrent)
            ],
            return_exceptions=True,
        )

        # Every call must succeed – no exceptions allowed.
        for i, result in enumerate(results):
            assert not isinstance(result, BaseException), (
                f"Concurrent create_session #{i} raised {result!r}; tables were"
                " likely not ready due to the _prepare_tables race condition."
            )

        # All sessions should be retrievable.
        for i in range(num_concurrent):
            session = await service.get_session(
                app_name=f"app_{i}", user_id="user", session_id=f"sess_{i}"
            )
            assert (
                session is not None
            ), f"Session sess_{i} not found after creation."

        assert service._tables_created
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_prepare_tables_serializes_schema_detection_and_creation():
    """Verifies schema detection and table creation happen atomically under one
    lock, so concurrent callers cannot observe a partially-initialized state.
    After _prepare_tables completes, both _db_schema_version and _tables_created
    must be set.
    """
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        assert not service._tables_created
        assert service._db_schema_version is None

        await service._prepare_tables()

        # Both must be set after a single _prepare_tables call.
        assert service._tables_created
        assert service._db_schema_version is not None

        # Verify tables actually exist by performing a real operation.
        session = await service.create_session(
            app_name="app", user_id="user", session_id="s1"
        )
        assert session is not None
        assert session.id == "s1"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_get_or_create_state_returns_existing_row():
    """_get_or_create_state returns an existing row without inserting."""
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        await service._prepare_tables()
        schema = service._get_schema_classes()

        # Pre-create the app_state row.
        async with service.database_session_factory() as sql_session:
            sql_session.add(
                schema.StorageAppState(app_name="app1", state={"k": "v"})
            )
            await sql_session.commit()

        # _get_or_create_state should find and return it.
        async with service.database_session_factory() as sql_session:
            row = await database_session_service._get_or_create_state(
                sql_session=sql_session,
                state_model=schema.StorageAppState,
                primary_key="app1",
                defaults={"app_name": "app1", "state": {}},
            )
            assert row.app_name == "app1"
            assert row.state == {"k": "v"}
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_get_or_create_state_creates_new_row():
    """_get_or_create_state creates a row when none exists."""
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        await service._prepare_tables()
        schema = service._get_schema_classes()

        async with service.database_session_factory() as sql_session:
            row = await database_session_service._get_or_create_state(
                sql_session=sql_session,
                state_model=schema.StorageAppState,
                primary_key="new_app",
                defaults={"app_name": "new_app", "state": {}},
            )
            await sql_session.commit()
            assert row.app_name == "new_app"
            assert row.state == {}

        # Verify the row was actually persisted.
        async with service.database_session_factory() as sql_session:
            persisted = await sql_session.get(
                schema.StorageAppState, "new_app"
            )
            assert persisted is not None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_get_or_create_state_handles_race_condition():
    """_get_or_create_state recovers when a concurrent INSERT wins the race.

    Simulates the race from https://github.com/google/adk-python/issues/4954:
    the initial SELECT returns None (another caller hasn't committed yet), but
    by the time we INSERT, the other caller has committed — so the INSERT fails
    with IntegrityError and we fall back to re-fetching.
    """
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        await service._prepare_tables()
        schema = service._get_schema_classes()

        # Pre-create the row to guarantee the INSERT will fail.
        async with service.database_session_factory() as sql_session:
            sql_session.add(
                schema.StorageAppState(app_name="race_app", state={})
            )
            await sql_session.commit()

        # Patch session.get to return None on the first call (simulating the
        # race window), then fall through to the real implementation.
        async with service.database_session_factory() as sql_session:
            original_get = sql_session.get
            call_count = 0

            async def patched_get(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return None  # Simulate: row not yet visible
                return await original_get(*args, **kwargs)

            sql_session.get = patched_get

            row = await database_session_service._get_or_create_state(
                sql_session=sql_session,
                state_model=schema.StorageAppState,
                primary_key="race_app",
                defaults={"app_name": "race_app", "state": {}},
            )
            assert row.app_name == "race_app"
            # The function should have called get twice: once before the INSERT
            # (patched to return None) and once after the IntegrityError.
            assert call_count == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_create_session_sequential_same_app_name():
    """Sequential create_session calls for the same app_name work correctly.

    The second call reuses the existing app_states row.
    """
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        s1 = await service.create_session(
            app_name="shared", user_id="u1", session_id="s1"
        )
        s2 = await service.create_session(
            app_name="shared", user_id="u2", session_id="s2"
        )
        assert s1.app_name == "shared"
        assert s2.app_name == "shared"

        got1 = await service.get_session(
            app_name="shared", user_id="u1", session_id="s1"
        )
        got2 = await service.get_session(
            app_name="shared", user_id="u2", session_id="s2"
        )
        assert got1 is not None
        assert got2 is not None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_prepare_tables_idempotent_after_creation():
    """Calling _prepare_tables multiple times is safe and idempotent.
    After tables are created, subsequent calls should return immediately via
    the fast path without errors.
    """
    service = DatabaseSessionService("sqlite+aiosqlite:///:memory:")
    try:
        await service._prepare_tables()
        assert service._tables_created

        # Call again — should be a no-op via the fast path.
        await service._prepare_tables()
        assert service._tables_created

        # Service should still work.
        session = await service.create_session(
            app_name="app", user_id="user", session_id="s1"
        )
        assert session.id == "s1"
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state_delta, expect_app_lock, expect_user_lock",
    [
        pytest.param(
            None,
            False,
            False,
            id="no_state_delta",
        ),
        pytest.param(
            {"session_key": "v"},
            False,
            False,
            id="session_only_delta",
        ),
        pytest.param(
            {"app:key": "v"},
            True,
            False,
            id="app_delta_only",
        ),
        pytest.param(
            {"user:key": "v"},
            False,
            True,
            id="user_delta_only",
        ),
        pytest.param(
            {"app:a": "1", "user:b": "2", "sk": "3"},
            True,
            True,
            id="all_scopes",
        ),
    ],
)
async def test_append_event_locks_only_scopes_with_deltas(
    state_delta, expect_app_lock, expect_user_lock
):
  """FOR UPDATE should only be requested for state scopes that have deltas."""
  service = DatabaseSessionService('sqlite+aiosqlite:///:memory:')

  lock_requests = []
  original_fn = database_session_service._select_required_state

  async def tracking_fn(**kwargs):
    lock_requests.append({
        'model': kwargs['state_model'].__tablename__,
        'use_row_level_locking': kwargs['use_row_level_locking'],
    })
    return await original_fn(**kwargs)

  try:
    session = await service.create_session(
        app_name='app', user_id='user', session_id='s1'
    )

    database_session_service._select_required_state = tracking_fn
    lock_requests.clear()

    event_kwargs = {'invocation_id': 'inv', 'author': 'user'}
    if state_delta is not None:
      event_kwargs['actions'] = EventActions(state_delta=state_delta)
    event = Event(**event_kwargs)
    await service.append_event(session, event)

    app_req = next(
        (r for r in lock_requests if r['model'] == 'app_states'), None
    )
    user_req = next(
        (r for r in lock_requests if r['model'] == 'user_states'), None
    )

    # SQLite doesn't support row-level locking so use_row_level_locking is
    # always False. The important check is that locking is not requested
    # when there is no delta (it must never be True without a delta).
    if not expect_app_lock:
      assert (
          app_req is None or not app_req['use_row_level_locking']
      ), 'app_states should not be locked without an app: delta'
    if not expect_user_lock:
      assert (
          user_req is None or not user_req['use_row_level_locking']
      ), 'user_states should not be locked without a user: delta'
  finally:
    database_session_service._select_required_state = original_fn
    await service.close()


@pytest.mark.asyncio
async def test_get_user_state_returns_empty_dict_when_no_state_exists(
    session_service,
):
  """Verifies get_user_state returns empty dict when no state exists."""
  state = await session_service.get_user_state(app_name='my_app', user_id='u1')
  assert not state


@pytest.mark.asyncio
async def test_get_user_state_returns_state_written_via_append_event(
    session_service,
):
  """Verifies get_user_state returns state written via append_event."""
  session = await session_service.create_session(
      app_name='my_app', user_id='u1'
  )
  await session_service.append_event(
      session,
      Event(
          author='system',
          actions=EventActions(
              state_delta={'user:profile': {'name': 'Alice'}, 'session_key': 1}
          ),
      ),
  )

  state = await session_service.get_user_state(app_name='my_app', user_id='u1')

  assert state == {'profile': {'name': 'Alice'}}
  assert 'session_key' not in state


@pytest.mark.asyncio
async def test_get_user_state_is_not_visible_across_users(session_service):
  """Verifies user state is isolated between users."""
  session = await session_service.create_session(
      app_name='my_app', user_id='u1'
  )
  await session_service.append_event(
      session,
      Event(
          author='system',
          actions=EventActions(state_delta={'user:secret': 'only-for-u1'}),
      ),
  )

  other_state = await session_service.get_user_state(
      app_name='my_app', user_id='u2'
  )
  assert not other_state


@pytest.mark.asyncio
async def test_get_user_state_is_not_visible_across_apps(session_service):
  """Verifies user state is isolated between apps."""
  session = await session_service.create_session(
      app_name='my_app', user_id='u1'
  )
  await session_service.append_event(
      session,
      Event(
          author='system',
          actions=EventActions(state_delta={'user:data': 'only-app-a'}),
      ),
  )

  other_state = await session_service.get_user_state(
      app_name='other_app', user_id='u1'
  )
  assert not other_state


@pytest.mark.asyncio
async def test_get_user_state_available_before_session_is_created(
    session_service,
):
  """Verifies user state can be retrieved before a session is created."""
  first_session = await session_service.create_session(
      app_name='my_app', user_id='u1'
  )
  await session_service.append_event(
      first_session,
      Event(
          author='system',
          actions=EventActions(state_delta={'user:ctx': {'v': 1}}),
      ),
  )

  state = await session_service.get_user_state(app_name='my_app', user_id='u1')
  assert state == {'ctx': {'v': 1}}


@pytest.mark.asyncio
async def test_get_user_state_reflects_latest_write(session_service):
  """Verifies get_user_state returns the latest state."""
  session = await session_service.create_session(
      app_name='my_app', user_id='u1'
  )
  await session_service.append_event(
      session,
      Event(
          author='system',
          actions=EventActions(state_delta={'user:counter': 1}),
      ),
  )
  await session_service.append_event(
      session,
      Event(
          author='system',
          actions=EventActions(state_delta={'user:counter': 2}),
      ),
  )

  state = await session_service.get_user_state(app_name='my_app', user_id='u1')
  assert state['counter'] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('light_copy', [False, True])
async def test_get_user_state_copies_to_session_state_depth(light_copy):
  """get_user_state copies as deeply as a session's own state is copied.

  Light copy exists to skip the recursive copy, so under it nested values stay
  shared with the service; without it they are deep-copied.
  """
  override_feature_enabled(
      FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY, light_copy
  )
  try:
    service = InMemorySessionService()
    await service.create_session(
        app_name='my_app',
        user_id='u1',
        session_id='s1',
        state={'user:profile': {'name': 'Alice'}, 'sk1': {'n': 1}},
    )

    user_state = await service.get_user_state(app_name='my_app', user_id='u1')
    user_state['profile']['name'] = 'Mallory'
    user_state['added'] = 1

    session = await service.get_session(
        app_name='my_app', user_id='u1', session_id='s1'
    )
    stored = service.sessions['my_app']['u1']['s1']
    session_state_is_shared = session.state['sk1'] is stored.state['sk1']

    assert (
        user_state['profile'] is service.user_state['my_app']['u1']['profile']
    ) == session_state_is_shared
    assert (
        service.user_state['my_app']['u1']['profile'] == {'name': 'Mallory'}
    ) == session_state_is_shared
    # A later session of the same user reads the same user state.
    later = await service.create_session(
        app_name='my_app', user_id='u1', session_id='s2'
    )
    assert (later.state['user:profile'] == {'name': 'Mallory'}) == (
        session_state_is_shared
    )
    assert 'added' not in service.user_state['my_app']['u1']
  finally:
    override_feature_enabled(
        FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY, False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('light_copy', [False, True])
@pytest.mark.parametrize('session_source', ['create', 'get', 'list'])
async def test_returned_session_scoped_state_uses_configured_copy_depth(
    light_copy, session_source
):
  """Returned sessions copy nested scoped state to the configured depth."""
  override_feature_enabled(
      FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY, light_copy
  )
  try:
    service = InMemorySessionService()
    created = await service.create_session(
        app_name='my_app',
        user_id='u1',
        session_id='s1',
        state={
            'app:config': {'theme': 'light'},
            'user:profile': {'name': 'Alice'},
        },
    )

    if session_source == 'create':
      returned = created
    elif session_source == 'get':
      returned = await service.get_session(
          app_name='my_app', user_id='u1', session_id='s1'
      )
    else:
      returned = (
          await service.list_sessions(app_name='my_app', user_id='u1')
      ).sessions[0]

    returned.state['app:config']['theme'] = 'dark'
    returned.state['user:profile']['name'] = 'Mallory'
    later = await service.create_session(
        app_name='my_app', user_id='u1', session_id='s2'
    )

    expected_theme = 'dark' if light_copy else 'light'
    expected_name = 'Mallory' if light_copy else 'Alice'
    assert later.state['app:config']['theme'] == expected_theme
    assert later.state['user:profile']['name'] == expected_name
  finally:
    override_feature_enabled(
        FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY, False
    )


@pytest.mark.asyncio
async def test_vertex_ai_session_service_raises_not_implemented_for_get_user_state():
  """Verifies VertexAiSessionService raises NotImplementedError."""
  service = VertexAiSessionService(project='proj', location='us-central1')
  with pytest.raises(NotImplementedError):
    await service.get_user_state(app_name='my_app', user_id='u1')


def test_database_session_service_visible_in_module_namespace():
  """DatabaseSessionService must be in dir() so Sphinx autodoc renders it.

  It is imported lazily via module __getattr__, so without an explicit
  __dir__ it drops out of the generated API reference.
  """
  import google.adk.sessions as sessions_module

  assert 'DatabaseSessionService' in dir(sessions_module)
  assert sessions_module.DatabaseSessionService is DatabaseSessionService


@pytest.mark.asyncio
async def test_database_session_service_with_db_url():
  """Test DatabaseSessionService initialization with db_url."""
  # Test db_url as positional argument
  service = DatabaseSessionService('sqlite+aiosqlite:///:memory:')
  app_name = 'test_app'
  user_id = 'test_user'

  # Create and retrieve a session
  session = await service.create_session(
      app_name=app_name, user_id=user_id, state={'key': 'value'}
  )
  assert session.app_name == app_name
  assert session.user_id == user_id
  assert session.state == {'key': 'value'}

  # Let's check that we can retrieve it
  retrieved = await service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id
  )
  assert retrieved == session

  # test db_url as keyword argument
  service2 = DatabaseSessionService(db_url='sqlite+aiosqlite:///:memory:')
  session2 = await service2.create_session(
      app_name=app_name, user_id=user_id, state={'key': 'value2'}
  )
  assert session2.state == {'key': 'value2'}


@pytest.mark.asyncio
async def test_database_session_service_with_db_engine():
  """Test DatabaseSessionService initialization with db_engine."""
  # Create an engine manually with StaticPool to avoid flakes
  engine = create_async_engine(
      'sqlite+aiosqlite:///:memory:',
      poolclass=StaticPool,
      connect_args={'check_same_thread': False},
  )

  # Create service with db_engine
  service = DatabaseSessionService(db_engine=engine)
  app_name = 'test_app'
  user_id = 'test_user'

  # Create and retrieve a session
  session = await service.create_session(
      app_name=app_name, user_id=user_id, state={'key': 'value'}
  )
  assert session.app_name == app_name
  assert session.user_id == user_id
  assert session.state == {'key': 'value'}

  # Let's check that we can retrieve it
  retrieved = await service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id
  )
  assert retrieved == session


@pytest.mark.asyncio
async def test_database_session_service_caller_owned_engine_not_disposed_on_close():
  """Verifies that a caller-owned engine is not disposed when the service is closed."""
  engine = create_async_engine(
      'sqlite+aiosqlite:///:memory:',
      poolclass=StaticPool,
      connect_args={'check_same_thread': False},
  )

  service = DatabaseSessionService(db_engine=engine)

  # Use the service
  session = await service.create_session(app_name='app', user_id='user')
  assert session is not None

  # Close the service
  await service.close()

  # Verify engine is still usable by running a query
  async with engine.connect() as conn:
    result = await conn.execute(text('SELECT 1;'))
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_database_session_service_requires_one_argument():
  """Test that DatabaseSessionService requires exactly one of db_url or db_engine."""
  # Neither argument provided
  with pytest.raises(
      ValueError,
      match="Exactly one of 'db_url' or 'db_engine' must be provided",
  ):
    DatabaseSessionService()

  # Both arguments provided
  engine = create_async_engine('sqlite+aiosqlite:///:memory:')
  with pytest.raises(
      ValueError,
      match="Exactly one of 'db_url' or 'db_engine' must be provided",
  ):
    DatabaseSessionService(
        db_url='sqlite+aiosqlite:///:memory:', db_engine=engine
    )


@pytest.mark.parametrize(
    'raised_error',
    [
        RuntimeError('boom'),
        ArgumentError('bad argument'),
        ImportError('no driver'),
        InvalidRequestError('not an async driver'),
    ],
)
def test_database_session_service_engine_error_hides_password(raised_error):
  """Engine creation errors must not put the DB password in the message."""
  password = 'sup3r-s3cret'
  db_url = f'postgresql+asyncpg://user:{password}@localhost:5432/db'

  with mock.patch.object(
      database_session_service,
      'create_async_engine',
      side_effect=raised_error,
  ):
    with pytest.raises(ValueError) as exc_info:
      DatabaseSessionService(db_url)

  message = str(exc_info.value)
  assert password not in message
  # The redacted URL is still there, so the error stays diagnosable.
  assert 'postgresql+asyncpg://user:***@localhost:5432/db' in message


def test_database_session_service_malformed_url_reports_usable_error():
  """A URL too malformed to parse still yields a usable, leak-free error."""
  # make_url() itself rejects this, so redaction cannot parse it either and
  # must fall back to a placeholder rather than echoing the raw string.
  db_url = 'definitely not a url sup3r-s3cret'

  with pytest.raises(ValueError) as exc_info:
    DatabaseSessionService(db_url)

  message = str(exc_info.value)
  assert 'sup3r-s3cret' not in message
  assert 'Invalid database URL format or argument' in message
  assert isinstance(exc_info.value.__cause__, ArgumentError)


def test_database_session_service_sync_driver_url_names_async_driver():
  """A synchronous URL is the common mistake, so name the driver that works."""
  with pytest.raises(ValueError) as exc_info:
    DatabaseSessionService('sqlite:///sessions.db')

  message = str(exc_info.value)
  assert 'synchronous' in message
  assert 'sqlite+aiosqlite' in message


@pytest.mark.asyncio
async def test_database_session_service_sqlite_file_timestamp_read_after_reopen(
    tmp_path,
):
  """Regression test for SQLite REAL-affinity timestamp reads."""
  # SQLite REAL-affinity columns can end up storing raw Unix epoch floats
  # instead of the text format SQLAlchemy's DateTime type normally writes
  # (for example, if the row was written by a different code path than the
  # SQLAlchemy ORM). Reading such a row back must not raise TypeError, since
  # TypeDecorator.process_result_value runs after the impl's own result
  # processor, which chokes on a float before process_result_value can run.
  # This test forces that condition directly via raw SQL.
  db_path = tmp_path / 'timestamp_regression.db'
  db_url = f'sqlite+aiosqlite:///{db_path}'
  app_name = 'my_app'
  user_id = 'user'

  service = DatabaseSessionService(db_url)
  try:
    session = await service.create_session(app_name=app_name, user_id=user_id)
    event = Event(author='user', timestamp=time.time())
    await service.append_event(session, event)
  finally:
    await service.close()

  # Directly overwrite the stored timestamp with a raw float via raw SQL,
  # simulating a REAL-affinity column value that bypassed SQLAlchemy's
  # normal text-based DateTime serialization.
  raw_epoch_float = time.time()
  conn = sqlite3.connect(str(db_path))
  try:
    cursor = conn.execute(
        'UPDATE events SET timestamp = ? WHERE session_id = ?',
        (raw_epoch_float, session.id),
    )
    # Without a row here the reopen below never sees a float, and the test
    # would pass without exercising the REAL-affinity path at all.
    assert cursor.rowcount == 1
    conn.commit()
  finally:
    conn.close()

  # Read it back with a fresh service instance; this must not raise
  # TypeError: fromisoformat: argument must be str.
  service2 = DatabaseSessionService(db_url)
  try:
    retrieved_session = await service2.get_session(
        app_name=app_name, user_id=user_id, session_id=session.id
    )
  finally:
    await service2.close()

  assert retrieved_session is not None
  assert len(retrieved_session.events) == 1
  # The returned timestamp is deserialized from the event_data blob rather than
  # from the DATETIME column overwritten above, so it still holds the value the
  # event was created with. Comparing it to the wall clock read for the raw
  # write instead only holds while both reads land in the same second.
  assert retrieved_session.events[0].timestamp == event.timestamp


@pytest.fixture
def local_timezone_with_dst():
  """Runs the test in a local timezone that repeats an hour every autumn.

  ``time.tzset`` is POSIX-only, so on other platforms the test runs in the
  host zone instead. Restoring ``TZ`` without a second ``tzset`` would leave
  the C library pinned for the rest of the session, so both are undone.
  """
  if not hasattr(time, 'tzset'):
    yield
    return
  original_tz = os.environ.get('TZ')
  os.environ['TZ'] = 'America/New_York'
  time.tzset()
  try:
    yield
  finally:
    if original_tz is None:
      del os.environ['TZ']
    else:
      os.environ['TZ'] = original_tz
    time.tzset()


@pytest.mark.asyncio
async def test_get_session_keeps_exact_epoch_across_a_repeated_local_hour(
    session_service, local_timezone_with_dst
):
  """Events written during a repeated local hour read back at the same instant.

  2024-11-03 06:00 and 06:30 UTC are 01:00 and 01:30 in US Eastern for the
  second time that morning; the same local wall-clock times already happened
  an hour earlier. A round trip that reconstructs the epoch from local wall
  clock alone cannot tell the two passes apart, so those events come back an
  hour early and sort into the wrong place in the conversation.
  """
  app_name = 'my_app'
  user_id = 'user'
  # Both instants fall in the repeated hour, so their local times are
  # ambiguous.
  repeated_hour_epochs = [1730613600.0, 1730615400.0]

  session = await session_service.create_session(
      app_name=app_name, user_id=user_id
  )
  for epoch in repeated_hour_epochs:
    await session_service.append_event(
        session, Event(author='user', timestamp=epoch)
    )

  retrieved_session = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id
  )

  assert [
      event.timestamp for event in retrieved_session.events
  ] == repeated_hour_epochs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type', [SessionServiceType.DATABASE, SessionServiceType.SQLITE]
)
@pytest.mark.parametrize('append_ids_in_reverse', [False, True])
async def test_get_session_orders_tied_timestamps_by_id(
    service_type, append_ids_in_reverse, tmp_path
):
  """Events sharing a timestamp come back in a stable, id-ordered sequence.

  Without a tiebreaker the database is free to return tied events in any
  order, so a replayed conversation shuffles between fetches and
  `num_recent_events` truncates at an arbitrary point inside the tie. Ordering
  on id as well also keeps the last returned event consistent with the event
  the stale-session check treats as the latest one.
  """
  app_name = 'my_app'
  user_id = 'user'
  event_ids = ['event_a', 'event_m', 'event_z']
  shared_timestamp = 100.0

  service = get_session_service(service_type, tmp_path)
  try:
    session = await service.create_session(app_name=app_name, user_id=user_id)
    append_order = (
        list(reversed(event_ids)) if append_ids_in_reverse else event_ids
    )
    for event_id in append_order:
      await service.append_event(
          session,
          Event(author='user', id=event_id, timestamp=shared_timestamp),
      )

    retrieved_session = await service.get_session(
        app_name=app_name, user_id=user_id, session_id=session.id
    )
  finally:
    if isinstance(service, DatabaseSessionService):
      await service.close()

  assert [event.id for event in retrieved_session.events] == event_ids


def test_delete_session_sync_removes_only_the_targeted_users_session():
  """Deleting is scoped to one (app, user, session) triple."""
  service = InMemorySessionService()
  app_name = 'my_app'
  service.create_session_sync(app_name=app_name, user_id='u1', session_id='s1')
  service.create_session_sync(app_name=app_name, user_id='u2', session_id='s1')

  service.delete_session_sync(app_name=app_name, user_id='u1', session_id='s1')

  assert (
      service.get_session_sync(app_name=app_name, user_id='u1', session_id='s1')
      is None
  )
  other_user_session = service.get_session_sync(
      app_name=app_name, user_id='u2', session_id='s1'
  )
  assert other_user_session is not None
  assert other_user_session.id == 's1'


def test_delete_session_sync_unknown_session_is_a_noop():
  """Deleting something that is not stored leaves the store untouched."""
  service = InMemorySessionService()
  app_name = 'my_app'
  service.create_session_sync(app_name=app_name, user_id='u1', session_id='s1')

  service.delete_session_sync(
      app_name=app_name, user_id='u1', session_id='unknown_session'
  )
  service.delete_session_sync(
      app_name=app_name, user_id='unknown_user', session_id='s1'
  )
  service.delete_session_sync(
      app_name='unknown_app', user_id='u1', session_id='s1'
  )

  assert (
      service.get_session_sync(app_name=app_name, user_id='u1', session_id='s1')
      is not None
  )


@pytest.mark.asyncio
async def test_list_sessions_sync_strips_events_and_merges_scoped_state():
  """Listed sessions carry merged app/user state but never their events."""
  service = InMemorySessionService()
  app_name = 'my_app'
  session = await service.create_session(
      app_name=app_name,
      user_id='u1',
      session_id='s1',
      state={
          'app:a': 'av',
          'user:u': 'uv',
          'sk': 'sv',
          'temp:t': 'tv',
      },
  )
  await service.append_event(
      session=session,
      event=Event(
          invocation_id='inv1',
          author='user',
          actions=EventActions(state_delta={'sk2': 'sv2'}),
      ),
  )

  response = service.list_sessions_sync(app_name=app_name, user_id='u1')

  assert [s.id for s in response.sessions] == ['s1']
  listed = response.sessions[0]
  # Events are deliberately dropped from the listing.
  assert listed.events == []
  # app: and user: values are merged back in under their prefixes, session
  # state is kept as-is, and temp: state is never stored.
  assert listed.state == {
      'app:a': 'av',
      'user:u': 'uv',
      'sk': 'sv',
      'sk2': 'sv2',
  }


def test_list_sessions_sync_unknown_app_or_user_returns_empty_response():
  """Listing an unknown app or user yields a response with no sessions."""
  service = InMemorySessionService()
  service.create_session_sync(app_name='my_app', user_id='u1', session_id='s1')

  assert service.list_sessions_sync(app_name='unknown_app').sessions == []
  assert (
      service.list_sessions_sync(
          app_name='my_app', user_id='unknown_user'
      ).sessions
      == []
  )
  assert [
      s.id for s in service.list_sessions_sync(app_name='my_app').sessions
  ] == ['s1']


# ---------------------------------------------------------------------------
# Regression tests for duplicate-event deduplication (issue #5723)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'session_service',
    [InMemorySessionService()],
    ids=['in_memory'],
)
async def test_append_event_is_idempotent_for_same_event_id(session_service):
  """Re-delivering an event must not duplicate entries or double-apply state.

  A broadcast can re-deliver the same event either as the same object or as
  an equal copy, so both must be deduplicated.
  """
  app_name = 'test_app'
  user_id = 'user_dup'
  session = await session_service.create_session(
      app_name=app_name, user_id=user_id, session_id='session_dup'
  )

  event = Event(
      invocation_id='inv_dup',
      author='user',
      actions=EventActions(state_delta={'session:counter': 1}),
  )

  # Re-deliver as the same object and again as an equal copy (a broadcast
  # to several concurrent session references can produce either).
  await session_service.append_event(session=session, event=event)
  await session_service.append_event(session=session, event=event)
  await session_service.append_event(
      session=session, event=event.model_copy(deep=True)
  )

  # The storage session must contain the event exactly once.
  retrieved = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id='session_dup'
  )
  matching = [e for e in retrieved.events if e.id == event.id]
  assert (
      len(matching) == 1
  ), f'Expected 1 occurrence of event {event.id!r}, got {len(matching)}'

  # State must not be double-applied.
  assert (
      retrieved.state.get('session:counter') == 1
  ), 'State was applied more than once — duplicate event caused double-apply'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'session_service',
    [InMemorySessionService()],
    ids=['in_memory'],
)
async def test_append_different_events_not_deduplicated(session_service):
  """Distinct events must both be stored, even when they share an event id.

  Deduplication is keyed on object identity, not event id: a caller can
  legitimately reuse an id across genuinely different events (e.g. a test
  that patches uuid4 to a constant), and keying on id alone would silently
  drop the later events.
  """
  app_name = 'test_app'
  user_id = 'user_multi'
  session = await session_service.create_session(
      app_name=app_name, user_id=user_id, session_id='session_multi'
  )

  # Two genuinely different events that share an id, as happens when a test
  # patches uuid generation to a fixed value.
  shared_id = 'shared-event-id'
  e1 = Event(id=shared_id, invocation_id='inv_a', author='user')
  e2 = Event(id=shared_id, invocation_id='inv_b', author='agent')

  await session_service.append_event(session=session, event=e1)
  await session_service.append_event(session=session, event=e2)

  retrieved = await session_service.get_session(
      app_name=app_name, user_id=user_id, session_id='session_multi'
  )
  assert (
      len(retrieved.events) == 2
  ), f'Expected 2 distinct events, got {len(retrieved.events)}'
  assert [e.author for e in retrieved.events] == ['user', 'agent']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'service_type',
    [
        SessionServiceType.IN_MEMORY,
        SessionServiceType.SQLITE,
        SessionServiceType.DATABASE,
    ],
)
async def test_append_event_applies_and_trims_temp_state_once(
    service_type: SessionServiceType, tmp_path
):
  """Persistent session services must not invoke _apply_temp_state/_trim_temp_delta_state twice."""
  session_service = get_session_service(service_type, tmp_path)
  session = await session_service.create_session(
      app_name='test_app', user_id='user_1', session_id='session_1'
  )
  event = Event(
      invocation_id='inv_1',
      author='agent',
      actions=EventActions(
          state_delta={'temp:scratch': 'ephemeral', 'persisted': 'val'}
      ),
  )
  with (
      mock.patch.object(
          session_service,
          '_apply_temp_state',
          wraps=session_service._apply_temp_state,
      ) as spy_apply,
      mock.patch.object(
          session_service,
          '_trim_temp_delta_state',
          wraps=session_service._trim_temp_delta_state,
      ) as spy_trim,
  ):
    await session_service.append_event(session=session, event=event)
    assert spy_apply.call_count == 1
    assert spy_trim.call_count == 1
  assert session.state.get('temp:scratch') == 'ephemeral'
  assert session.state.get('persisted') == 'val'
