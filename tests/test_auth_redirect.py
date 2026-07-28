"""Server-boundary authentication redirect tests."""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from quoptuna.server.core import auth

REDIRECT_STATUS = 307
UNAUTHORIZED_STATUS = 401


def build_test_app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(auth.redirect_unauthenticated_requests)

    @app.get("/")
    async def index():
        return {"page": "index"}

    @app.get("/settings/")
    async def settings_page():
        return {"page": "settings"}

    @app.get("/api/v1/info")
    async def info():
        return {"version": "test"}

    @app.get("/api/v1/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/auth/profile")
    async def profile():
        raise HTTPException(status_code=UNAUTHORIZED_STATUS, detail="Not authenticated")

    return app


def enable_auth(monkeypatch) -> None:
    monkeypatch.setattr(auth.settings, "AUTH0_DOMAIN", "tenant.example.com")
    monkeypatch.setattr(auth.settings, "AUTH0_CLIENT_ID", "client")
    monkeypatch.setattr(auth.settings, "AUTH0_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr(auth.settings, "AUTH0_SECRET", "session-secret")
    monkeypatch.setattr(auth.settings, "AUTH_ALLOWED_EMAILS", ["approved@example.com"])
    monkeypatch.setattr(auth, "_client", None)


def disable_auth(monkeypatch) -> None:
    monkeypatch.setattr(auth.settings, "AUTH0_DOMAIN", "")
    monkeypatch.setattr(auth.settings, "AUTH0_CLIENT_ID", "")
    monkeypatch.setattr(auth.settings, "AUTH0_CLIENT_SECRET", "")
    monkeypatch.setattr(auth.settings, "AUTH0_SECRET", "")
    monkeypatch.setattr(auth, "_client", None)


def test_unauthenticated_page_redirects_to_login_with_return_path(monkeypatch):
    enable_auth(monkeypatch)

    response = TestClient(build_test_app()).get(
        "/settings/?tab=runs", follow_redirects=False
    )

    assert response.status_code == REDIRECT_STATUS
    assert (
        response.headers["location"]
        == "/auth/login?returnTo=%2Fsettings%2F%3Ftab%3Druns"
    )


def test_unauthenticated_api_request_redirects_to_login(monkeypatch):
    enable_auth(monkeypatch)

    response = TestClient(build_test_app()).get("/api/v1/info", follow_redirects=False)

    assert response.status_code == REDIRECT_STATUS
    assert response.headers["location"] == "/auth/login?returnTo=%2Fapi%2Fv1%2Finfo"


def test_auth_and_health_routes_remain_public(monkeypatch):
    enable_auth(monkeypatch)
    client = TestClient(build_test_app())

    profile = client.get("/auth/profile", follow_redirects=False)
    health = client.get("/api/v1/health", follow_redirects=False)

    assert profile.status_code == UNAUTHORIZED_STATUS
    assert health.is_success


def test_auth_disabled_keeps_local_app_accessible(monkeypatch):
    disable_auth(monkeypatch)

    response = TestClient(build_test_app()).get("/", follow_redirects=False)

    assert response.is_success
