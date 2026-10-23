# local

Local apps' files: everything here but this README is gitignored. How to add one: [caddy/](../../README.md#local-apps).

**Notes**
- Files here are read by Caddy and the landing page, never served (`/local/*` answers 404).
- Validate before restarting: a broken local file stops Caddy, and every app with it.
- An app's `.caddy` or `.routes` file names a host and port outside the stack, and its `.html` shows on the landing page, which has no login: keep secrets (tokens) out of all three.
