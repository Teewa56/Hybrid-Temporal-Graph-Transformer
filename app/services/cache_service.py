import os
import json
import redis.asyncio as aioredis
from typing import Any


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class CacheService:
    """
    Redis-backed in-memory cache for sub-millisecond transaction
    payload retrieval during the inference window.
    Avoids hitting the database during fraud checks.
    """

    def __init__(self):
        self._client: aioredis.Redis = None

    async def connect(self):
        self._client = await aioredis.from_url(REDIS_URL, decode_responses=True)
        await self._client.ping()
        print("✅ Redis connected.")

    async def disconnect(self):
        if self._client:
            await self._client.close()

    async def set(self, key: str, value: Any, ttl: int = 300):
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        await self._client.setex(key, ttl, value)

    async def get(self, key: str) -> Any:
        value = await self._client.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    async def delete(self, key: str):
        await self._client.delete(key)

    async def push_to_list(self, key: str, value: Any, max_len: int = 50):
        """
        Push a transaction to a user's history list.
        Trims to max_len so we always have the last N transactions ready.
        """
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        pipe = self._client.pipeline()
        pipe.lpush(key, value)
        pipe.ltrim(key, 0, max_len - 1)
        await pipe.execute()

    async def get_list(self, key: str, limit: int = 50) -> list:
        items = await self._client.lrange(key, 0, limit - 1)
        result = []
        for item in items:
            try:
                result.append(json.loads(item))
            except json.JSONDecodeError:
                result.append(item)
        return result

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(key))