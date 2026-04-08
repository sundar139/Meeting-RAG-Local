from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from meeting_pipeline.db.migrations import apply_migrations, discover_migration_files


@dataclass
class FakeMigrationCursor:
    state: FakeMigrationState
    rowcount: int = 0
    _fetchall_data: list[tuple[str]] = field(default_factory=list)

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self.state.executed.append((query, params))
        normalized = " ".join(query.split()).lower()

        if normalized.startswith("select name from schema_migrations"):
            self._fetchall_data = [(name,) for name in sorted(self.state.applied)]
            return

        if normalized.startswith("insert into schema_migrations"):
            assert params is not None
            migration_name = str(params[0])
            if migration_name in self.state.applied:
                self.rowcount = 0
            else:
                self.state.applied.add(migration_name)
                self.rowcount = 1

    def fetchall(self) -> list[tuple[str]]:
        return list(self._fetchall_data)

    def __enter__(self) -> FakeMigrationCursor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        return None


@dataclass
class FakeMigrationState:
    applied: set[str]
    executed: list[tuple[str, tuple[object, ...] | None]] = field(default_factory=list)


class FakeMigrationConnection:
    def __init__(self, state: FakeMigrationState) -> None:
        self.state = state
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> FakeMigrationCursor:
        return FakeMigrationCursor(self.state)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _write_migration(path: Path, sql: str) -> None:
    path.write_text(sql, encoding="utf-8")


def test_discover_migration_files_sorted(tmp_path: Path) -> None:
    _write_migration(tmp_path / "0002_second.sql", "SELECT 2;")
    _write_migration(tmp_path / "0001_first.sql", "SELECT 1;")

    discovered = discover_migration_files(tmp_path)

    assert [path.name for path in discovered] == ["0001_first.sql", "0002_second.sql"]


def test_apply_migrations_tracks_applied_and_skipped(tmp_path: Path) -> None:
    _write_migration(tmp_path / "0001_first.sql", "SELECT 1;")
    _write_migration(tmp_path / "0002_second.sql", "SELECT 2;")

    state = FakeMigrationState(applied={"0001_first.sql"})
    connection = FakeMigrationConnection(state)

    result = apply_migrations(connection=connection, migrations_dir=tmp_path)

    assert result.applied == ["0002_second.sql"]
    assert result.skipped == ["0001_first.sql"]
    assert "0002_second.sql" in state.applied
    assert any("schema_migrations" in query for query, _ in state.executed)


def test_apply_migrations_is_idempotent(tmp_path: Path) -> None:
    _write_migration(tmp_path / "0001_first.sql", "SELECT 1;")

    state = FakeMigrationState(applied=set())
    connection = FakeMigrationConnection(state)

    first_run = apply_migrations(connection=connection, migrations_dir=tmp_path)
    second_run = apply_migrations(connection=connection, migrations_dir=tmp_path)

    assert first_run.applied == ["0001_first.sql"]
    assert second_run.applied == []
    assert second_run.skipped == ["0001_first.sql"]
