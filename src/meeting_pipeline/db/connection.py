from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from meeting_pipeline.config import Settings, get_settings

if TYPE_CHECKING:
    from collections.abc import Iterator

    from psycopg2.extensions import connection as PsycopgConnection


class DatabaseConnectionError(RuntimeError):
    """Raised when the application cannot establish a PostgreSQL connection."""


def _build_connection_kwargs(settings: Settings, application_name: str | None) -> dict[str, object]:
    app_name = application_name.strip() if application_name else settings.app_name
    return {
        "host": settings.postgres_host,
        "port": settings.postgres_port,
        "dbname": settings.postgres_db,
        "user": settings.postgres_user,
        "password": settings.postgres_password.get_secret_value(),
        "application_name": app_name,
        "connect_timeout": 10,
    }


def create_connection(
    settings: Settings | None = None, *, application_name: str | None = None
) -> PsycopgConnection:
    runtime_settings = settings or get_settings()

    try:
        import psycopg2
        from psycopg2 import OperationalError
    except ModuleNotFoundError as exc:
        raise DatabaseConnectionError(
            "psycopg2 is not installed. Install service dependencies with "
            "'uv sync --group dev --extra services'."
        ) from exc

    kwargs = _build_connection_kwargs(runtime_settings, application_name)

    try:
        return psycopg2.connect(**kwargs)
    except OperationalError as exc:
        raise DatabaseConnectionError(
            "Could not connect to PostgreSQL. Verify POSTGRES_HOST, POSTGRES_PORT, "
            "POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, and ensure the server is running."
        ) from exc


@contextmanager
def connection_scope(
    settings: Settings | None = None, *, application_name: str | None = None
) -> Iterator[PsycopgConnection]:
    connection = create_connection(settings=settings, application_name=application_name)
    try:
        yield connection
    finally:
        connection.close()


def build_postgres_dsn(settings: Settings) -> str:
    return settings.postgres_dsn
