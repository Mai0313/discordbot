import sqlite3

import pandas as pd
from pydantic import Field, BaseModel
from sqlalchemy import create_engine
import autorootcwd  # noqa: F401

from src.discordbot.types.database import DatabaseConfig


class DatabaseMigration(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    def migrate(self, path_or_data: pd.DataFrame | str) -> None:
        data = pd.read_csv(path_or_data) if isinstance(path_or_data, str) else path_or_data
        data["attachments"] = data["attachments"].apply(
            lambda x: x.replace("data/", "data/attachments/")
            if isinstance(x, str) and not x.startswith("data/attachments/")
            else x
        )
        engine = create_engine(f"sqlite:///{self.database.sqlite.sqlite_file_path}")
        groups = data.groupby("channel_name")
        for name, group in groups:
            if not name.startswith("DM"):
                group.to_sql(name=f"channel_{name}", con=engine, if_exists="replace", index=False)

    def split(self, name: str) -> None:
        engine = create_engine(f"sqlite:///{self.database.sqlite.sqlite_file_path}")
        data = pd.read_sql_table(table_name=name, con=engine)
        self.migrate(path_or_data=data)
        # remove the original table
        with engine.connect() as conn:
            conn.execute(f"DROP TABLE {name}")

    def migrate_to_postgresql(self, path: str) -> None:
        sqlite_conn = sqlite3.connect(path)
        pg_engine = create_engine(self.database.postgres.postgres_url)
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables: list[str] = [row[0] for row in cursor.fetchall()]

        for table_name in tables:
            *_, channel_id = table_name.split("_")
            data = pd.read_sql_query(f"SELECT * FROM `{table_name}`", sqlite_conn)  # noqa: S608
            data.to_sql(f"channel_{channel_id}", pg_engine, if_exists="append", index=False)
        sqlite_conn.commit()
        sqlite_conn.close()


if __name__ == "__main__":
    db_migration = DatabaseMigration()
    # db_migration.migrate(path_or_data="./data/messages.csv")
    db_migration.migrate_to_postgresql(path="./data/messages.db")
    # db_migration.split(name="messages")
