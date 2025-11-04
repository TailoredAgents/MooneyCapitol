from __future__ import annotations

from app.db.session import engine
from app.db.models import Base


def main():
    Base.metadata.create_all(bind=engine)
    print("DB tables created.")


if __name__ == "__main__":
    main()

