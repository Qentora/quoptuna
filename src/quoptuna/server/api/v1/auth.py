"""
Auth0 login/logout/profile routes (server-side flow, mounted at /auth).
"""

import logging

from auth0_server_python.auth_types import LogoutOptions, StartInteractiveLoginOptions
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from quoptuna.server.core.auth import get_auth_client, get_current_user
from quoptuna.server.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

RETURN_TO_COOKIE = "_a0_return_to"


def _safe_return_to(value: str) -> str:
    """Only allow same-site paths or whitelisted origins as post-login targets."""
    if value.startswith("/") and not value.startswith("//"):
        return value
    for origin in [settings.APP_BASE_URL, *settings.CORS_ORIGINS]:
        if value == origin or value.startswith(origin.rstrip("/") + "/"):
            return value
    return "/"


def _require_configured() -> None:
    if not settings.AUTH_ENABLED:
        raise HTTPException(status_code=501, detail="Auth0 is not configured")


@router.get("/login")
async def login(request: Request):
    _require_configured()
    params = dict(request.query_params)
    return_to = _safe_return_to(params.pop("returnTo", "/"))
    response = RedirectResponse(url="/", status_code=302)
    url = await get_auth_client().start_interactive_login(
        options=StartInteractiveLoginOptions(authorization_params=params),
        store_options={"request": request, "response": response},
    )
    response.headers["location"] = url
    response.set_cookie(
        RETURN_TO_COOKIE,
        return_to,
        max_age=300,
        httponly=True,
        samesite="lax",
        secure=settings.APP_BASE_URL.startswith("https://"),
    )
    return response


@router.get("/callback")
async def callback(request: Request):
    _require_configured()
    return_to = _safe_return_to(request.cookies.get(RETURN_TO_COOKIE, "/"))
    response = RedirectResponse(url=return_to, status_code=302)
    response.delete_cookie(RETURN_TO_COOKIE)
    try:
        await get_auth_client().complete_interactive_login(
            url=str(request.url),
            store_options={"request": request, "response": response},
        )
    except Exception:
        logger.exception("Auth0 callback error")
        return JSONResponse(
            status_code=400,
            content={"detail": "Login failed. Check server logs for details."},
        )
    return response


@router.get("/logout")
async def logout(request: Request):
    _require_configured()
    # Send the user back where they came from (e.g. the :3000 dev frontend).
    # The target must also be in the Auth0 app's Allowed Logout URLs.
    return_to = _safe_return_to(request.query_params.get("returnTo", "/"))
    if return_to.startswith("/"):
        return_to = settings.APP_BASE_URL.rstrip("/") + return_to
    response = RedirectResponse(url="/", status_code=302)
    url = await get_auth_client().logout(
        options=LogoutOptions(return_to=return_to),
        store_options={"request": request, "response": response},
    )
    response.headers["location"] = url
    return response


@router.get("/profile")
async def profile(request: Request):
    """Current user's claims, 401 when not logged in. Used by the SPA."""
    if not settings.AUTH_ENABLED:
        return {"user": None, "auth_enabled": False}
    user = await get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": user, "auth_enabled": True}
