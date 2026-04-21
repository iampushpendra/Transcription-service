"""Tests for the self-signup authentication module."""

import json
import os
import shutil
import tempfile

import pytest

from pipeline import auth


@pytest.fixture
def tmp_users(monkeypatch):
    d = tempfile.mkdtemp(prefix="authtest_")
    store = os.path.join(d, "users.json")
    monkeypatch.setattr(auth, "USERS_FILE", store)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---- Signup + uniqueness ---------------------------------------------------

def test_signup_creates_user_and_hashes_password(tmp_users):
    u = auth.signup(username="alice", name="Alice Sharma", team="Sales", password="s3cret6")
    assert u["username"] == "alice"
    assert u["name"] == "Alice Sharma"
    assert u["team"] == "Sales"
    assert "id" in u
    assert "created_at" in u
    assert "password_hash" not in u  # safe projection — never leak the hash

    # The hash IS in storage, distinct from plaintext.
    with open(auth.USERS_FILE) as f:
        stored = json.load(f)
    assert stored["users"][0]["password_hash"] != "s3cret6"
    assert stored["users"][0]["password_hash"].startswith("$2")  # bcrypt prefix


def test_signup_rejects_duplicate_username(tmp_users):
    auth.signup(username="alice", name="A", team="Sales", password="abc123")
    with pytest.raises(auth.SignupError):
        auth.signup(username="alice", name="Another", team="Audit", password="def456")


def test_signup_rejects_duplicate_username_case_insensitive(tmp_users):
    auth.signup(username="Alice", name="A", team="Sales", password="abc123")
    with pytest.raises(auth.SignupError):
        auth.signup(username="ALICE", name="B", team="Audit", password="def456")


def test_signup_validates_password_length(tmp_users):
    with pytest.raises(auth.SignupError, match="at least"):
        auth.signup(username="alice", name="A", team="Sales", password="ab")  # too short


def test_signup_validates_required_fields(tmp_users):
    with pytest.raises(auth.SignupError):
        auth.signup(username="", name="A", team="Sales", password="abc123")
    with pytest.raises(auth.SignupError):
        auth.signup(username="alice", name="", team="Sales", password="abc123")


def test_signup_validates_team(tmp_users):
    with pytest.raises(auth.SignupError, match="[Tt]eam"):
        auth.signup(username="alice", name="A", team="NotARealTeam", password="abc123")


def test_signup_accepts_each_configured_team(tmp_users):
    for i, t in enumerate(auth.list_teams()):
        auth.signup(username=f"u{i}", name="X", team=t, password="abc123")


# ---- Login -----------------------------------------------------------------

def test_verify_credentials_happy_path(tmp_users):
    u = auth.signup(username="alice", name="A", team="Sales", password="s3cret6")
    checked = auth.verify_credentials("alice", "s3cret6")
    assert checked["id"] == u["id"]


def test_verify_credentials_wrong_password(tmp_users):
    auth.signup(username="alice", name="A", team="Sales", password="s3cret6")
    assert auth.verify_credentials("alice", "wrong!!") is None


def test_verify_credentials_unknown_user(tmp_users):
    assert auth.verify_credentials("bob", "anything") is None


def test_verify_credentials_is_case_insensitive_on_username(tmp_users):
    auth.signup(username="Alice", name="A", team="Sales", password="s3cret6")
    assert auth.verify_credentials("alice", "s3cret6") is not None
    assert auth.verify_credentials("ALICE", "s3cret6") is not None


# ---- Lookups ---------------------------------------------------------------

def test_get_user_by_id(tmp_users):
    u = auth.signup(username="alice", name="A", team="Sales", password="abc123")
    fetched = auth.get_user_by_id(u["id"])
    assert fetched is not None
    assert fetched["username"] == "alice"
    assert "password_hash" not in fetched


def test_get_user_by_id_unknown_returns_none(tmp_users):
    assert auth.get_user_by_id("not-a-real-id") is None


def test_list_teams_returns_non_empty_list():
    teams = auth.list_teams()
    assert isinstance(teams, list) and teams
    assert all(isinstance(t, str) for t in teams)


# ---- Password hashing primitive ------------------------------------------

def test_hash_and_check_password_roundtrips():
    h = auth._hash_password("hello6chars")
    assert h != "hello6chars"
    assert auth._check_password("hello6chars", h) is True
    assert auth._check_password("nope", h) is False
