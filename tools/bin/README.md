# bin

The commands an SSH key may run on the service account, one script each: the key's `authorized_keys` line forces that script (`restrict,command=...`), so the key can run it and nothing else, whatever the client asks for. `ctl.sh up` installs them from `FORCED_COMMANDS` in `services/ctl.sh`.

| Command | Does | Key (in `$STATE_DIR`) | Used by |
|---|---|---|---|
| `dynamo-up` | `services/ctl.sh dynamo up`: serve the models whose MLflow alias `dynamo` is set ([dynamo/](../../services/dynamo/README.md)) | `dynamo/sync_key` (Gitea secret `DYNAMO_SSH_KEY`) | Gitea workflow `sync-dynamo` |
| `setup-envs` | the three installers in `services/gitea/setup/`, side by side ([setup/](../../services/gitea/setup/README.md)); `--dry-run` (the one argument it reads) only checks them | `setup-envs/ssh_key` (Gitea secret `SETUP_SSH_KEY`) | Gitea workflow `setup-envs`, dispatched by `ctl.sh up` |
| `triton-up` | `services/ctl.sh triton up -d`: recreate the Triton server after its image, GPUs or flags change, and wait until it's ready ([triton/](../../services/triton/README.md)); `--dry-run` shows what it would recreate | `triton/up_key` (Gitea secret `TRITON_UP_SSH_KEY`) | Gitea workflow `triton-up` |

1. Add a command: write `tools/bin/<name>` (executable; it runs as the service account, from this checkout), add `"<name> <key file>"` to `FORCED_COMMANDS`, and pass the key to whatever needs it.
2. Install it: `services/ctl.sh up` (it makes a missing key and rewrites the command's `authorized_keys` line when the key or the checkout's path changes).

**Notes**
- `~/.ssh/authorized_keys` is shared by the cluster's nodes, so a key works on each of them; the scripts only ever run what they name.
- The scripts run from this checkout as it is on disk, uncommitted edits included.
