"""
Simple self-signup authentication.

File-backed user store (JSON), bcrypt-hashed passwords, no external auth
service. Sessions are managed by the server layer via Starlette's
SessionMiddleware; this module just owns credentials and lookups.

Storage shape (outputs/__users__/users.json):
    {
      "users": [
        {"id": "<uuid4>", "username": "...", "name": "...",
         "team": "...", "password_hash": "<bcrypt>", "created_at": "..."}
      ]
    }
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone

import bcrypt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default store location. Tests monkeypatch this to a tmp dir.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_FILE: str = os.path.join(_PROJECT_ROOT, "outputs", "__users__", "users.json")

# Fixed team choices. Edit here to add/remove teams.
TEAMS: list[str] = ["Sales", "Audit", "Quality", "Operations", "Management"]

MIN_PASSWORD_LEN = 6
MAX_PASSWORD_LEN = 128
MAX_USERNAME_LEN = 40
MAX_NAME_LEN = 80

# File-level lock to keep read-modify-write atomic under concurrent signups.
_STORE_LOCK = threading.Lock()


class SignupError(ValueError):
    """Raised when a signup request is rejected (validation or uniqueness)."""


# -----------------------------------------------------------------------------
# Password primitives
# -----------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def _check_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# -----------------------------------------------------------------------------
# Storage
# -----------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_store() -> dict:
    if not os.path.exists(USERS_FILE):
        return {"users": []}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "users" not in data:
            return {"users": []}
        return data
    except (OSError, json.JSONDecodeError):
        return {"users": []}


def _save_store(store: dict) -> None:
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    # Write-then-rename for crash safety.
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    os.replace(tmp, USERS_FILE)


def _safe_projection(user: dict) -> dict:
    """Return a user dict without the password hash (safe to expose over HTTP)."""
    return {k: v for k, v in user.items() if k != "password_hash"}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def list_teams() -> list[str]:
    return list(TEAMS)


def signup(username: str, name: str, team: str, password: str) -> dict:
    """Create a new user. Returns a safe projection (no password hash).

    Raises SignupError on validation failure or duplicate username.
    """
    username = (username or "").strip()
    name = (name or "").strip()
    team = (team or "").strip()
    password = password or ""

    # Validate
    if not username:
        raise SignupError("Username is required.")
    if len(username) > MAX_USERNAME_LEN:
        raise SignupError(f"Username must be at most {MAX_USERNAME_LEN} characters.")
    if not all(c.isalnum() or c in "._-" for c in username):
        raise SignupError("Username may contain letters, digits, '.', '_', and '-' only.")
    if not name:
        raise SignupError("Name is required.")
    if len(name) > MAX_NAME_LEN:
        raise SignupError(f"Name must be at most {MAX_NAME_LEN} characters.")
    if team not in TEAMS:
        raise SignupError(f"Team must be one of: {', '.join(TEAMS)}.")
    if len(password) < MIN_PASSWORD_LEN:
        raise SignupError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    if len(password) > MAX_PASSWORD_LEN:
        raise SignupError(f"Password must be at most {MAX_PASSWORD_LEN} characters.")

    with _STORE_LOCK:
        store = _load_store()
        lower = username.lower()
        if any(u.get("username", "").lower() == lower for u in store["users"]):
            raise SignupError("That username is already taken.")

        user = {
            "id": str(uuid.uuid4()),
            "username": username,
            "name": name,
            "team": team,
            "password_hash": _hash_password(password),
            "created_at": _now_iso(),
        }
        store["users"].append(user)
        _save_store(store)

    return _safe_projection(user)


def verify_credentials(username: str, password: str) -> dict | None:
    """Return the safe-projected user on match, else None. Case-insensitive username."""
    if not username or not password:
        return None
    store = _load_store()
    lower = username.lower()
    for u in store["users"]:
        if u.get("username", "").lower() == lower:
            if _check_password(password, u.get("password_hash", "")):
                return _safe_projection(u)
            return None
    return None


def get_user_by_id(user_id: str) -> dict | None:
    if not user_id:
        return None
    store = _load_store()
    for u in store["users"]:
        if u.get("id") == user_id:
            return _safe_projection(u)
    return None


def get_user_by_username(username: str) -> dict | None:
    if not username:
        return None
    store = _load_store()
    lower = username.lower()
    for u in store["users"]:
        if u.get("username", "").lower() == lower:
            return _safe_projection(u)
    return None


def user_exists() -> bool:
    """Returns True if any user has ever signed up. Used to toggle auth enforcement."""
    store = _load_store()
    return len(store.get("users", [])) > 0
