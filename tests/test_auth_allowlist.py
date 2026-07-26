"""Approved-email authorization tests."""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from quoptuna.server.core import auth
from quoptuna.server.core.config import Settings

HTTP_FORBIDDEN = 403


def configure_allowlist(monkeypatch, emails, *, require_verified=True):
    monkeypatch.setattr(auth.settings, "AUTH_ALLOWED_EMAILS", emails)
    monkeypatch.setattr(auth.settings, "AUTH_REQUIRE_VERIFIED_EMAIL", require_verified)


def test_approved_verified_email_is_case_insensitive(monkeypatch):
    configure_allowlist(monkeypatch, ["researcher@example.com"])
    user = {"email": " Researcher@Example.com ", "email_verified": True}
    assert auth.enforce_approved_user(user) is user


def test_unapproved_email_is_rejected(monkeypatch):
    configure_allowlist(monkeypatch, ["approved@example.com"])
    with pytest.raises(HTTPException) as error:
        auth.enforce_approved_user({"email": "other@example.com", "email_verified": True})
    assert error.value.status_code == HTTP_FORBIDDEN


def test_unverified_email_is_rejected(monkeypatch):
    configure_allowlist(monkeypatch, ["approved@example.com"])
    with pytest.raises(HTTPException) as error:
        auth.enforce_approved_user({"email": "approved@example.com", "email_verified": False})
    assert error.value.status_code == HTTP_FORBIDDEN


def test_comma_separated_allowlist_is_supported(monkeypatch):
    configure_allowlist(monkeypatch, "one@example.com, two@example.com")
    user = {"email": "two@example.com", "email_verified": True}
    assert auth.enforce_approved_user(user) is user


def test_empty_allowlist_retains_local_auth_compatibility(monkeypatch):
    configure_allowlist(monkeypatch, [])
    user = {"email": "local@example.com", "email_verified": True}
    assert auth.enforce_approved_user(user) is user


def test_production_auth_requires_an_allowlist():
    with pytest.raises(ValidationError, match="AUTH_ALLOWED_EMAILS"):
        Settings(
            _env_file=None,
            APP_ENV="production",
            AUTH0_DOMAIN="tenant.example.com",
            AUTH0_CLIENT_ID="client",
            AUTH0_CLIENT_SECRET=str(1),
            AUTH0_SECRET=str(2),
            AUTH_ALLOWED_EMAILS=[],
        )


def test_production_auth_accepts_an_allowlist():
    configured = Settings(
        _env_file=None,
        APP_ENV="production",
        AUTH0_DOMAIN="tenant.example.com",
        AUTH0_CLIENT_ID="client",
        AUTH0_CLIENT_SECRET=str(1),
        AUTH0_SECRET=str(2),
        AUTH_ALLOWED_EMAILS="Researcher@Example.com",
    )
    assert configured.AUTH_ALLOWED_EMAILS == ["researcher@example.com"]
