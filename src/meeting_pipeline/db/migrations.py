from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol

from meeting_pipeline.db.connection import connection_scope

LOGGER = logging.getLogger(__name__)
DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"


class CursorProtocol(Protocol):
    rowcount: int

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None: ...

    def fetchall(self) -> list[tuple[str]]: ...

    def __enter__(self) -> CursorProtocol: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class ConnectionProtocol(Protocol):
    def cursor(self) -> CursorProtocol: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


@dataclass(frozen=True)
class MigrationRunResult:
    applied: list[str]
    skipped: list[str]


class MigrationError(RuntimeError):
    """Raised when a migration cannot be applied successfully."""


def load_migration_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def discover_migration_files(migrations_dir: Path) -> list[Path]:
    return sorted(
        (path for path in migrations_dir.glob("*.sql") if path.is_file()),
        key=lambda p: p.name,
    )


def _ensure_schema_migrations_table(connection: ConnectionProtocol) -> None:
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """)
    connection.commit()


def _get_applied_migrations(connection: ConnectionProtocol) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM schema_migrations")
        rows = cursor.fetchall()
    return {row[0] for row in rows}


def apply_migrations(
    connection: ConnectionProtocol, migrations_dir: Path = DEFAULT_MIGRATIONS_DIR
) -> MigrationRunResult:
    if not migrations_dir.exists() or not migrations_dir.is_dir():
        raise FileNotFoundError(f"Migrations directory not found: {migrations_dir}")

    _ensure_schema_migrations_table(connection)
    known_migrations = _get_applied_migrations(connection)

    applied: list[str] = []
    skipped: list[str] = []

    for migration_path in discover_migration_files(migrations_dir):
        migration_name = migration_path.name

        if migration_name in known_migrations:
            LOGGER.info("Skipping migration %s (already applied)", migration_name)
            skipped.append(migration_name)
            continue

        migration_sql = load_migration_sql(migration_path).strip()
        if not migration_sql:
            LOGGER.info("Skipping migration %s (empty SQL file)", migration_name)
            skipped.append(migration_name)
            continue

        try:
            with connection.cursor() as cursor:
                cursor.execute(migration_sql)
                cursor.execute(
                    "INSERT INTO schema_migrations (name) VALUES (%s) "
                    "ON CONFLICT (name) DO NOTHING",
                    (migration_name,),
                )
            connection.commit()
        except Exception as exc:
            connection.rollback()
            raise MigrationError(
                f"Failed to apply migration '{migration_name}'. "
                "Inspect migration SQL and database compatibility."
            ) from exc

        known_migrations.add(migration_name)
        applied.append(migration_name)
        LOGGER.info("Applied migration %s", migration_name)

    return MigrationRunResult(applied=applied, skipped=skipped)


def run_migrations(migrations_dir: Path = DEFAULT_MIGRATIONS_DIR) -> MigrationRunResult:
    with connection_scope(application_name="meeting_pipeline:migrations") as connection:
        return apply_migrations(connection=connection, migrations_dir=migrations_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply PostgreSQL schema migrations.")
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=DEFAULT_MIGRATIONS_DIR,
        help="Directory containing .sql migration files.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_migrations(migrations_dir=args.migrations_dir)
    LOGGER.info(
        "Migration summary: applied=%d skipped=%d",
        len(result.applied),
        len(result.skipped),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
