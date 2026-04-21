# History UX + Self-signup Authentication

**Date:** 2026-04-21
**Status:** Approved, implementing

---

## Sprint 1 — History UX

### Sort options

History grid gets a sort selector. Options:

- **Date added** (default, descending)
- **Duration** (descending)
- **Filename** (A → Z)
- **Calls** (chain size, descending — only affects chain rows)

Sort is applied client-side on the cached history response.

### View tabs

Three tabs above the sort selector:

- **All** (default) — current behavior, chains first then singles
- **Single Calls** — only un-chained calls
- **Chains** — only chain rows

Tab state + sort state persist in `sessionStorage` across route changes.

---

## Sprint 2 — Authentication

### Scope

- Per-user isolation of transcripts and chains.
- Self-signup (no admin CLI).
- Signup fields: **username, name, team (dropdown), password (min 6 chars)**.
- Session cookie–based auth (server-side session, HTTPOnly).
- Legacy items (no `owner_user_id`) stay **visible to all** in a **collapsed "Shared archive" section at the bottom** of history.

### User storage

File-based, matching the project's no-database posture. Location: `outputs/__users__/users.json` (gitignored by existing `outputs/` rule).

```json
{
  "users": [
    {
      "id": "<uuid4>",
      "username": "alice",
      "name": "Alice Sharma",
      "team": "Sales",
      "password_hash": "<bcrypt>",
      "created_at": "ISO8601"
    }
  ]
}
```

Teams — fixed list (configurable by editing a constant):
```
["Sales", "Audit", "Quality", "Operations", "Management"]
```

Password policy: min 6 chars, max 128, trimmed. Hashed with bcrypt cost 12.

Username uniqueness enforced at signup.

### Session

Starlette `SessionMiddleware` with secret from `SESSION_SECRET_KEY` env; falls back to an ephemeral random key in dev (logs a warning — sessions invalidate on server restart in that case).

Cookie: HTTPOnly, SameSite=Lax, lifetime 14 days.

Session payload: `{"user_id": "<uuid>"}`.

### API

| Method | Path | Behavior |
|---|---|---|
| POST | `/api/auth/signup` | Body `{username, name, team, password}` → creates user, logs in, returns safe user dict |
| POST | `/api/auth/login` | Body `{username, password}` → sets session cookie |
| POST | `/api/auth/logout` | Clears session |
| GET | `/api/auth/me` | Returns the current user (or 401) |
| GET | `/api/auth/teams` | Returns the configured teams list (for the signup dropdown) |

All other `/api/*` endpoints require a valid session via a FastAPI dependency. Static files remain public so the login page can render.

### Ownership tagging

- **Single calls:** At save time, `transcript.json` gets `owner_user_id: "<uuid>"`. No back-fill for existing outputs — they stay unowned (legacy).
- **Chains:** `manifest.json` gets `owner_user_id` at creation. Appending a call to someone else's chain is forbidden.

### History filtering

`GET /api/history` filters by current user:

```json
{
  "my_chains":      [ ...chains owned by current user... ],
  "my_singles":     [ ...singles owned by current user... ],
  "legacy_chains":  [ ...chains without owner_user_id... ],
  "legacy_singles": [ ...singles without owner_user_id... ],
  "history":        [ ...flat combined list... ]    // legacy backwards-compat
}
```

Frontend renders `my_*` grouped at top; `legacy_*` in a collapsed "Shared archive" section at the bottom (expand chevron reveals them).

### Frontend

- **Unauth gate:** On app load, fetch `/api/auth/me`. If 401, show a login/signup overlay (covers the whole viewport) until successful.
- **Signup view:** Inputs for username, name, team (select), password. Client-side validation for min length + required fields.
- **Login view:** username + password only.
- **User badge:** Top-right, shows `Name · Team`. Logout button dropdown.
- **History:** My items rendered as today (with new sort/tab); shared archive as a collapsed section.

### Edge cases

| Case | Handling |
|---|---|
| Signup with existing username | 409 + form error |
| Invalid password on login | 401 + generic "Invalid credentials" error |
| Session expired | Automatic 401 on API call → frontend re-shows login overlay |
| User deletes an already-owned chain owned by another user | 403 (owner mismatch) |
| Someone navigates to chain UUID they don't own | `/api/chain/{id}` returns 403 |
| Legacy (unowned) chain | Everyone can view it, but only admin-style moves could re-assign (out of scope) |
| Signup password too short | 400 with clear message |
| Upload while unauth'd | 401 (gate applies) |

### Non-goals (deferred)

- Password reset flow (no email infrastructure)
- Admin role / user management UI
- Bulk-claim tool for legacy items
- Rate limiting on signup/login
- Audit log of who accessed what

### Tests

- **Auth unit:** signup rejects duplicate usernames; password hash round-trips; login verifies credential pair; logout clears; `get_current_user` raises on missing/invalid session.
- **Ownership:** `transcript.json` gets `owner_user_id` stamped by the queue worker; chain creation stamps manifest.
- **History filtering:** user A sees only their items + legacy; user B sees theirs + same legacy; neither sees the other's items.
- **Endpoint guard:** unauth'd GET /api/history → 401.

### Deployment note

Existing users upgrading this codebase will need to set `SESSION_SECRET_KEY` (either env var or `.env`). Missing key = warning + ephemeral session (sessions reset every restart).
