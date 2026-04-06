# THAI-MOD Basic Auth Flow (Demo)

This project uses a simple cookie-based session login to protect moderator/admin access for course demonstration.

## Why this exists

- THAI-MOD is a **decision-support** system.
- The primary user is a **moderator/admin** reviewing model recommendations.
- Admin-facing UI should not be publicly accessible without protection.

## What is protected

- `/admin` route (always protected)
- `/api/admin/overview` (always protected)
- Optional: `/` and prediction APIs when `THAI_MOD_PROTECT_ANALYZER=true`

## Auth method (simple)

- Login via `POST /api/auth/login` with one demo username/password.
- On success, backend stores session in signed cookie (`thai_mod_session`).
- Logout via `POST /api/auth/logout` clears session.
- Frontend checks session via `GET /api/auth/me`.

## Demo credentials setup

Set environment variables before running:

```bash
export THAI_MOD_AUTH_USERNAME="moderator"
export THAI_MOD_AUTH_PASSWORD="thai-mod-demo-2026"
export THAI_MOD_SESSION_SECRET="set-a-random-secret-for-your-demo"
export THAI_MOD_PROTECT_ANALYZER="false"
```

The app will not start until these values are explicitly provided.
If a project-root `.env` file is present, the app loads it automatically.

Then run:

```bash
uvicorn src.thai_mod_api.main:app --reload
```

## Presentation user flow (short)

1. Open `/admin` while logged out → redirected to `/login`.
2. Sign in with demo credential → redirected back to `/admin`.
3. Show admin monitoring cards load successfully.
4. Click logout.
5. Re-open `/admin` → redirected to `/login` again.

This demonstrates basic authentication and access control for moderator/admin workflow.

## Notes

- This is intentionally minimal for demo/course requirement.
- Not intended as production-grade authentication.
