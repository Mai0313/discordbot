import pandas as pd
from pydantic import Field, BaseModel
from sqlalchemy import create_engine
from src.types.database import DatabaseConfig


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


if __name__ == "__main__":
    db_migration = DatabaseMigration()
    db_migration.migrate(path_or_data="./data/messages.csv")
    # db_migration.split(name="messages")
