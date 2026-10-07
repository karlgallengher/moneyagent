# Web access and invited accounts

The Web API requires an invited account. There is no public registration endpoint.
CLI and stdio MCP are local trusted interfaces; do not expose either as a public
network service without an equivalent authorization boundary.

## Create an account

From the project root, using the `moneyagent` Python environment:

```powershell
python -m scripts.web_auth owner --claim-legacy
python -m scripts.web_auth colleague
```

Passwords are entered interactively and are never passed on the command line.
Use `--claim-legacy` only for the intended owner of existing unowned sessions
and uploaded domains. Without this flag, old unowned data stays inaccessible
through the Web API. Built-in knowledge domains remain readable by all users.

Restart the Web API after updating its code. Open the Web UI and sign in with
the account created above. Each user can manage only their own sessions and
uploaded knowledge domains. Two users can use the same domain identifier: the
server namespaces it internally.

## Configuration before public deployment

- Serve the Web UI and API over HTTPS. Non-localhost login cookies require HTTPS;
  cookies are HttpOnly and SameSite=Strict. Keep both on the same origin or
  configure `MONEYAGENT_ALLOWED_ORIGINS` as a comma-separated list of exact
  frontend origins. Every state-changing request requires `X-CSRF-Token`.
- `MONEYAGENT_STATE_DIR`: persistent directory for the SQLite session and
  account database. The default is `enterprise_qa_agent/outputs/chat_sessions`.
- `MONEYAGENT_DATA_ROOT`: persistent directory containing `processed/user_domains.json`,
  `processed/user_page_indexes`, and `public_dataset_upload/raw_md`.
  The default is the repository root.
- `MONEYAGENT_CHAT_PER_HOUR`: maximum chat requests per user in a rolling
  hour (default 20). `MONEYAGENT_BUILDS_PER_HOUR` defaults to 5.
  `MONEYAGENT_DOMAINS_PER_USER` defaults to 8.
  `MONEYAGENT_FILES_PER_DOMAIN` defaults to 50, and
  `MONEYAGENT_UPLOAD_BYTES_PER_DOMAIN` defaults to 64 MiB.
  Individual uploads are limited to 8 MiB by the API.
- Back up existing local data before migration. Move the existing SQLite file,
  domain registry, uploads, and index directories together when switching
  storage roots. Confirm that uploaded domains still build and search after
  moving them.

The above makes the local Web API private by default, but it is not a complete
AgentArts deployment. Select durable cloud storage, configure HTTPS, backups,
monitoring, instance/concurrency policy, and gateway limits before public launch.
The Web API uses SQLite and a process-local index lock, so multi-instance
deployment requires replacing these coordination and storage mechanisms.
