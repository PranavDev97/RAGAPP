from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from handler.status import status_router
from handler.chat import chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.up_since = datetime.now(timezone.utc)
    yield


app = FastAPI(title="RAGAPP", lifespan=lifespan)
app.include_router(status_router)
app.include_router(chat_router)
