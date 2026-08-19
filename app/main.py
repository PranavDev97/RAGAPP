import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI
from google.genai import types
from handler.status import status_router
from handler.chat import chat_router
from clients.ai import create_gemini_client
from processors.embedder import create_embed_model
from repository.postgresql.common import create_postgres_client

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
    await create_postgres_client(
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "password"),
        database=os.environ.get("POSTGRES_DB", "rag"),
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
    )

    app.state.up_since = datetime.now(timezone.utc)
    yield


app = FastAPI(title="RAGAPP", lifespan=lifespan)
app.include_router(status_router)
app.include_router(chat_router)
