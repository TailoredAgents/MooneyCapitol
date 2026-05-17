from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text


MIGRATION_LOCK_ID = 2026051701


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required to run migrations")

    alembic_cfg = Config(str(Path("alembic.ini")))
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)

    if database_url.startswith(("postgresql://", "postgresql+")):
        engine = create_engine(database_url, pool_pre_ping=True, future=True)
        with engine.connect() as connection:
            connection.execute(text("select pg_advisory_lock(:lock_id)"), {"lock_id": MIGRATION_LOCK_ID})
            try:
                command.upgrade(alembic_cfg, "head")
            finally:
                connection.execute(text("select pg_advisory_unlock(:lock_id)"), {"lock_id": MIGRATION_LOCK_ID})
        engine.dispose()
        return

    command.upgrade(alembic_cfg, "head")


if __name__ == "__main__":
    main()
