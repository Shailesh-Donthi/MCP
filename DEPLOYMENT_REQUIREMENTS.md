# Deployment Requirements

## Project Layout (Deployed)

- `client/` (static frontend files)
- `server/` (FastAPI backend code)
- Root wrappers may be used for compatibility:
  - `mcp_server.py`
  - `requirements.txt`

## Required Components

- Static web server (Nginx/Caddy/Apache) to serve `client/`
- Python application runtime for backend (`server/`)
- MongoDB (required)
- Reverse proxy with HTTPS termination (required for production)

## Optional Components

- Redis (`REDIS_URL`) for cache/shared state
- LLM provider API access for smarter routing:
  - OpenAI (`OPENAI_API_KEY`)
  - Anthropic (`ANTHROPIC_API_KEY`)

## Runtime Requirements

- Python 3.10+ (3.11 recommended)
- `pip` and virtual environment support
- Backend dependencies installed from `server/requirements.txt`

## Network / Ports

- Public:
  - `443/TCP` (HTTPS)
  - `80/TCP` (HTTP -> HTTPS redirect, optional)
- Internal:
  - `8090/TCP` backend app (`MCP_PORT`)

## Reverse Proxy Requirements

- Serve `client/chatbot.html` at `/`
- Serve frontend static assets from `client/`
- Proxy `/api/*` to backend (`127.0.0.1:8090` or configured host/port)
- Preserve proxy headers:
  - `Host`
  - `X-Forwarded-For`
  - `X-Forwarded-Proto`

## Required Backend Configuration (`.env`)

- `MONGODB_URI`
- `MONGODB_DB_NAME`
- `MCP_HOST`
- `MCP_PORT`
- `ALLOWED_ORIGINS`
- `JWT_SECRET_KEY`
- `JWT_ALGORITHM`

## Optional Backend Configuration (`.env`)

- `REDIS_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `ANTHROPIC_API_KEY`
- `MCP_LOG_LEVEL`
- `MCP_LOG_FILE`
- `MCP_LOG_MAX_BYTES`
- `MCP_LOG_BACKUP_COUNT`

## Production Security Requirements

- HTTPS enabled for all user traffic
- `ALLOWED_ORIGINS` restricted to explicit frontend origin(s) (no `*`)
- Strong non-default `JWT_SECRET_KEY`
- Backend app port not exposed publicly
- MongoDB/Redis access restricted to trusted hosts/networks
- Debug/diagnostic endpoints restricted or disabled in production (for example `/api/v1/db-test`)
- Secrets (`.env`) not committed to source control

## Data / Persistence Requirements

- MongoDB reachable from backend with valid credentials
- Required MongoDB data/collections present
- Backup/restore process defined for MongoDB

## Chat Memory Requirement (Current Constraint)

- Current server-side chat memory is in-process only
- Deployment requirement:
  - Single backend instance only for reliable behavior, **or**
  - Move chat memory to shared persistence (Redis/MongoDB) before multi-instance deployment

## Logging / Monitoring Requirements

- Writable backend log path configured (if file logging enabled)
- Log rotation configured (`MCP_LOG_MAX_BYTES`, `MCP_LOG_BACKUP_COUNT`)
- Health endpoints reachable for monitoring:
  - `/health`
  - `/ready`
- `/metrics` exposure restricted to internal/protected access if enabled

## Process Management Requirements

- Backend must run under a process manager/service supervisor (e.g. `systemd`, NSSM)
- Restart policy enabled
- Startup command standardized (one of):
  - `python mcp_server.py`
  - `uvicorn mcp.server_http:app --host 127.0.0.1 --port 8090`

## Go-Live Validation Requirements

- Frontend loads successfully from deployed URL
- `/health` returns success
- `/ready` returns success
- `/api/v1/ask` returns a valid response
- MongoDB-backed query succeeds
- Chat APIs work:
  - create chat
  - list chats
  - get chat
  - send query and persist messages
- CORS works from production frontend origin
- HTTPS certificate valid

## Release Tracking (Fill Per Deployment)

- Environment: `Dev / UAT / Prod`
- Version / Commit SHA: `____________________`
- Deployment date/time: `____________________`
- Deployed by: `____________________`

