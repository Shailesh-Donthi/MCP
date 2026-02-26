"""MongoDB connection helpers for MCP server."""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from mcp.core.config import settings

_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None


async def connect_to_mongodb() -> AsyncIOMotorDatabase:
    global _client, _database

    if _database is not None:
        return _database

    _client = AsyncIOMotorClient(settings.MONGODB_URI)
    _database = _client[settings.MONGODB_DB_NAME]
    await _database.command("ping")
    return _database


def get_database() -> AsyncIOMotorDatabase | None:
    return _database


async def close_mongodb_connection() -> None:
    global _client, _database

    if _client is not None:
        _client.close()
    _client = None
    _database = None
