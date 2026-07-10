"""
Auth0 integration (auth0-server-python SDK) with encrypted cookie stores.

The SDK needs two stores: one for session data (user profile, tokens) and one
for short-lived OAuth transaction data (PKCE verifiers, state params). Both are
kept client-side in encrypted cookies, so no server-side session storage is
required. Cookies are read from the incoming Starlette request and written to
the outgoing response, both passed via ``store_options``.
"""

import logging
from typing import Any, Optional

from auth0_server_python.auth_server.server_client import ServerClient
from auth0_server_python.auth_types import StateData, TransactionData
from auth0_server_python.store.abstract import AbstractDataStore
from fastapi import HTTPException, Request, Response

from quoptuna.server.core.config import settings

logger = logging.getLogger(__name__)

SESSION_COOKIE = "_a0_session"
TRANSACTION_COOKIE = "_a0_tx"


class CookieStore(AbstractDataStore):
    """Store SDK state in an encrypted, httponly cookie."""

    def __init__(self, secret: str, cookie_name: str, max_age: int, model: type):
        super().__init__({"secret": secret})
        self.cookie_name = cookie_name
        self.max_age = max_age
        self.model = model

    def _response(self, options: Optional[dict[str, Any]]) -> Response:
        response = (options or {}).get("response")
        if response is None:
            raise RuntimeError(
                f"CookieStore({self.cookie_name}) requires a 'response' in store_options"
            )
        return response

    async def set(
        self,
        identifier: str,
        state: Any,
        remove_if_expires: bool = False,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        data = state.model_dump() if hasattr(state, "model_dump") else state
        self._response(options).set_cookie(
            self.cookie_name,
            self.encrypt(identifier, data),
            httponly=True,
            samesite="lax",
            secure=settings.APP_BASE_URL.startswith("https://"),
            max_age=self.max_age,
        )

    async def get(
        self, identifier: str, options: Optional[dict[str, Any]] = None
    ) -> Optional[Any]:
        request: Optional[Request] = (options or {}).get("request")
        if request is None:
            return None
        encrypted = request.cookies.get(self.cookie_name)
        if not encrypted:
            return None
        try:
            return self.model.model_validate(self.decrypt(identifier, encrypted))
        except Exception:
            logger.warning("Failed to decrypt cookie %s", self.cookie_name, exc_info=True)
            return None

    async def delete(
        self, identifier: str, options: Optional[dict[str, Any]] = None
    ) -> None:
        self._response(options).delete_cookie(self.cookie_name)


_client: Optional[ServerClient] = None


def get_auth_client() -> ServerClient:
    """Lazily build the singleton Auth0 ServerClient from settings."""
    global _client  # noqa: PLW0603 - lazy singleton
    if _client is None:
        if not settings.AUTH_ENABLED:
            raise RuntimeError("Auth0 is not configured (missing AUTH0_* env vars)")
        secret = settings.AUTH0_SECRET
        _client = ServerClient(
            domain=settings.AUTH0_DOMAIN,
            client_id=settings.AUTH0_CLIENT_ID,
            client_secret=settings.AUTH0_CLIENT_SECRET,
            redirect_uri=f"{settings.APP_BASE_URL}/auth/callback",
            authorization_params={"scope": "openid profile email"},
            secret=secret,
            state_store=CookieStore(secret, SESSION_COOKIE, 259200, StateData),  # 3 days
            transaction_store=CookieStore(secret, TRANSACTION_COOKIE, 300, TransactionData),  # 5 min
        )
    return _client


async def get_current_user(request: Request) -> Optional[dict]:
    """Return the logged-in user's claims, or None."""
    if not settings.AUTH_ENABLED:
        return None
    return await get_auth_client().get_user({"request": request})


async def require_user(request: Request) -> Optional[dict]:
    """FastAPI dependency: 401 unless a valid session exists.

    A no-op when Auth0 is not configured, so local/dev/test setups without
    credentials keep working.
    """
    if not settings.AUTH_ENABLED:
        return None
    user = await get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
