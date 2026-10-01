import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI
from google.genai import types
from handler.status import status_router
from handler.file import file_router
from handler.user import user_router
from clients.ai import create_gemini_client
from processors.embedder import create_embed_model
from processors.compression import create_compression_model
from repository.postgresql.common import create_postgres_client
from repository.redis.common import create_redis_client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_gemini_client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_opts=types.HttpOptions(
            timeout=2 * 60 * 1000,
            retry_options=types.HttpRetryOptions(
                attempts=3,
                jitter=3
            ),
        ),
    )
    await create_embed_model()
    await create_compression_model()
    await create_postgres_client(
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
        database=os.environ.get("POSTGRES_DB"),
        host=os.environ.get("POSTGRES_HOST"),
        port=int(os.environ.get("POSTGRES_PORT")),
    )

    await create_redis_client(
        host=os.environ.get("REDIS_HOST"),
        port=int(os.environ.get("REDIS_PORT")),
    )

    app.state.up_since = datetime.now(timezone.utc)
    yield


app = FastAPI(title="RAGAPP", lifespan=lifespan)
app.include_router(status_router)
app.include_router(file_router)
app.include_router(user_router)
