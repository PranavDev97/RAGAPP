from fastapi import APIRouter, Request

status_router = APIRouter()


@status_router.get("/status")
def status(request: Request):
    return {
        "status": "ok", 
        "message": f"api is up and running since {request.app.state.up_since.isoformat()}"
    }
