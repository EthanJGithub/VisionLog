import hashlib
import re
import secrets

from starlette.responses import JSONResponse
from src.privacy import workspace


async def isolate_workspace(request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    capability = request.headers.get("X-Workspace-Key") or request.cookies.get("visionlog_workspace")
    if capability and not re.fullmatch(r"[a-f0-9]{64}", capability):
        return JSONResponse({"detail": "Invalid workspace key"}, status_code=400)
    capability = capability or secrets.token_hex(32)
    token = workspace.set(hashlib.sha256(capability.encode()).hexdigest())
    try:
        response = await call_next(request)
        response.set_cookie("visionlog_workspace", capability, httponly=True,
                            secure=request.url.scheme == "https", samesite="strict", max_age=2592000)
        return response
    finally:
        workspace.reset(token)
