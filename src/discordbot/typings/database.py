from typing import Any
from functools import cached_property

from redis import Redis
import dotenv
import logfire
from pydantic import Field, BaseModel, AliasChoices, computed_field
from sqlalchemy import text, create_engine
from sqlalchemy.exc import SQLAlchemyError
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class PostgreSQLConfig(BaseSettings):
    postgres_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/messages",
        validation_alias=AliasChoices("POSTGRES_URL"),
        title="PostgreSQL Url",
        description="The URL to connect to the PostgreSQL database.",
        frozen=False,
        deprecated=False,
    )

    def init_db(self) -> None:
        try:
            engine = create_engine(self.postgres_url)

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                logfire.info("PostgreSQL database connection successful")

        except SQLAlchemyError:
            try:
                postgres_url = self.postgres_url
                if "/" in postgres_url:
                    base_url = postgres_url.rsplit("/", 1)[0]
                    database_name = postgres_url.rsplit("/", 1)[1]
                    admin_engine = create_engine(f"{base_url}/postgres")

                    with admin_engine.connect() as admin_conn:
                        admin_conn.execute(text("COMMIT"))
                        admin_conn.execute(text(f"CREATE DATABASE {database_name}"))
                        logfire.info(f"Successfully created database: {database_name}")

                        test_engine = create_engine(postgres_url)
                        with test_engine.connect() as test_conn:
                            test_conn.execute(text("SELECT 1"))
                            logfire.info("New database connection verified")

            except SQLAlchemyError as create_error:
                logfire.error(f"Failed to create database: {create_error}", _exc_info=True)
        except Exception as e:
            logfire.error(f"Unexpected error ensuring database: {e}", _exc_info=True)


class SQLiteConfig(BaseSettings):
    sqlite_file_path: str = Field(
        default="sqlite:///data/messages.db",
        validation_alias=AliasChoices("SQLITE_FILE_PATH"),
        title="SQLite File Path",
        description="The file path to the SQLite database file.",
        frozen=False,
        deprecated=False,
    )


class RedisConfig(BaseSettings):
    redis_url: str = Field(
        default="redis://redis:6379/0",
        validation_alias=AliasChoices("REDIS_URL"),
        title="Redis Url",
        description="The URL to connect to the Redis server.",
        frozen=False,
        deprecated=False,
    )

    @computed_field
    @cached_property
    def redis_instance(self) -> Redis:
        return Redis.from_url(url=self.redis_url)

    @computed_field
    @property
    def hkeys(self) -> list[str]:
        # 這裡的 hkeys 沒有指定要對哪一個 hash 進行操作，
        # 因此假設是要取得所有的 key (相當於 Redis 的 keys *)
        # 若想要針對特定 hash，請更改此處為 self.redis_instance.hkeys('your_hash_key')
        all_keys: list[bytes] = self.redis_instance.keys("*")
        return [key.decode("utf-8") for key in all_keys]

    def hvalues(self, key: str) -> list[str]:
        # 取得特定 hash key 中所有的值
        values: list[bytes] = self.redis_instance.hvals(key)
        return [val.decode("utf-8") for val in values]

    def save(self, key: str, data: dict[str, str]) -> dict[str, Any]:
        # 使用 hset 將 data 字典保存到指定的 hash key 中
        # data 格式例如: {"field1": "value1", "field2": "value2"}
        self.redis_instance.hset(key, mapping=data)  # type: ignore
        return data

    def load(self, key: str) -> dict[str, str]:
        # 使用 hgetall 取得指定 hash key 中所有 field-value
        raw_data: dict[bytes, bytes] = self.redis_instance.hgetall(key)
        if not raw_data:
            return {}
        return {k.decode("utf-8"): v.decode("utf-8") for k, v in raw_data.items()}

    def delete(self, key: str) -> None:
        # 刪除整個 hash key
        self.redis_instance.delete(key)


class DatabaseConfig(BaseModel):
    postgres: PostgreSQLConfig = PostgreSQLConfig()
    sqlite: SQLiteConfig = SQLiteConfig()
    redis: RedisConfig = RedisConfig()
