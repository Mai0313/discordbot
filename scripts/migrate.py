import sqlite3

import pandas as pd
from pydantic import Field, BaseModel
from sqlalchemy import create_engine

from discordbot.typings.database import DatabaseConfig


class DatabaseMigration(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    @staticmethod
    def sanitize_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        for col in data.select_dtypes(include=["object"]).columns:
            data[col] = data[col].astype(str).str.replace("\x00", "", regex=False)
        return data

    def migrate_to_postgresql(self, path: str) -> None:
        sqlite_conn = sqlite3.connect(path)
        psg_engine = create_engine(self.database.postgres.postgres_url)
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables: list[str] = [row[0] for row in cursor.fetchall()]

        for table_name in tables:
            print(f"Migrating table: {table_name}")  # noqa: T201
            data = pd.read_sql_query(f"SELECT * FROM `{table_name}`", sqlite_conn)  # noqa: S608
            data = self.sanitize_dataframe(data)
            data.to_sql(
                name=table_name,
                con=psg_engine,
                if_exists="append",
                index=False,
                chunksize=10_000,
                method="multi",
            )
        sqlite_conn.commit()
        sqlite_conn.close()


if __name__ == "__main__":
    db_migration = DatabaseMigration()
    db_migration.migrate_to_postgresql(path="./data/messages.db")
