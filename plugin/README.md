# mdqm Claude Code plugin

The client side of `mdqm`: a Claude Code plugin that connects a user's desktop Claude Code to the
central `mdqm` server. It adds three things.

| Piece | What it does |
|---|---|
| `.mcp.json` | Points Claude Code at the server's `/mcp` endpoint with the user's own bearer token. The five tools are always loaded, so a weak agent does not have to discover them. |
| `hooks/hooks.json` | A `UserPromptSubmit` hook that asks the server `POST /context` for the semantic facts matching the user's question and injects them into the prompt. |
| `skills/mdqm-query/SKILL.md` | Tells the agent how to use the five tools: the JSON query spec, three worked examples, and how to act on an error. |

The plugin holds **no token, no host and no port**. Both come from the user's environment, so one
checkout installs anywhere.

## Before you install

You need:

1. **Network access to the `mdqm` server** from the desktop.
2. **A personal bearer token** issued by whoever runs the server. Never share one; the server's
   `history` and audit trail are per person.
3. **Python 3.9 or newer** on the desktop, for the context hook only. The MCP tools do not use it.

Set both values in the environment Claude Code is launched from:

```sh
# macOS / Linux (bash, zsh)
export MDQM_URL="https://mdqm.example.internal"
export MDQM_TOKEN="<your personal token>"
```

```powershell
# Windows PowerShell
$env:MDQM_URL = "https://mdqm.example.internal"
$env:MDQM_TOKEN = "<your personal token>"
```

`MDQM_URL` is the server's base URL, with no `/mcp` and no trailing slash. If you paste the MCP
endpoint by mistake, both the plugin and the hook drop the suffix for you.

Put those two lines in your shell profile (or `$PROFILE`) so every session has them. An unset
`MDQM_TOKEN` is not a secret leak, but the server answers `401` and the tools will not work; an
unset `MDQM_URL` leaves the `${MDQM_URL}` text unexpanded, which `/mcp` reports as an invalid URL.

## Install

### From an internal marketplace

Add the marketplace that hosts this plugin once, then install it by name:

```
/plugin marketplace add <your-internal-git-url-or-path-to-repo>
/plugin install mdqm@<marketplace-name>
```

The marketplace is a repository or directory with a `.claude-plugin/marketplace.json` that lists
plugins and where to fetch each one. Ask whoever runs the `mdqm` server for the URL.

Choose an install scope when prompted:

* **User scope** — every project on this computer. The usual choice for a data tool.
* **Project scope** — everyone who works in one repository, through its committed
  `.claude/settings.json`. Each collaborator still installs it on their own machine.

Then start a session: `/mcp` shows the server as `plugin:mdqm:mdqm`, and `/plugin` lists the
plugin's skill as `/mdqm:mdqm-query`.

### From a local directory

While developing, or to try the plugin from a checkout without publishing it:

```sh
claude --plugin-dir /path/to/mdqm/plugin/mdqm
```

`--plugin-dir` loads the folder for that session only, so nothing is installed and nothing needs
removing afterwards. Restart the session after editing a plugin file, or run `/reload-plugins`.

### Choose the Python interpreter

On first enable, Claude Code asks for the interpreter that runs the context hook. It defaults to
`python3`; on Windows, set it to `python`. It must be a **real executable** on `PATH` — the hook is
spawned directly with no shell involved, which is why it works the same on macOS, Linux and Windows
with no bash anywhere. The value is editable later under `/config`.

The hook cannot be run for you: if the interpreter is wrong, the hook fails to start, which is
logged but not fatal. You lose the injected context and everything else keeps working. Confirm with
`claude --debug-file <path>` and search for `Hooks: UserPromptSubmit`.

## Check it works

1. `/mcp` — the server appears as `plugin:mdqm:mdqm` with five tools: `find_data`, `query`,
   `get_more`, `history`, `feedback`.
2. Ask a data question, for example *"average credit limit by status last 3 months"*. The agent
   should call `query` with a JSON spec, not write SQL.
3. If the server is up, the prompt arrives with the matching measures and value maps already in
   context. `claude --debug-file <path>` shows a `Hooks: UserPromptSubmit` line with a duration.

Failures are quiet by design: the hook prints nothing and exits `0` when the server is down, the
token is rejected, or the network times out, so a broken server never blocks a prompt. If context
stops appearing, the hook is the thing to check, not the prompt.

## How the pieces talk to each other

```
Claude Code --prompt--> hooks/hooks.json --stdin JSON--> scripts/context_hook.py
                                                            |
                                                    POST /context  (Bearer token)
                                                            |
                                        <-- additionalContext --> prompt + semantic facts
Claude Code --tool call--> .mcp.json --> POST /mcp   (same Bearer token) --> mdqm server
```

The hook and the MCP entry use the same two environment variables, so configuring the plugin once
configures both. `POST /context` and the `find_data` tool build the same block from the same call,
so the injected facts and the tool's answer cannot disagree.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `/mcp` shows `'url' is not a valid URL` | `MDQM_URL` is unset or empty. Export it, or unset-and-set rather than setting it to `""`. |
| `/mcp` shows a `401` or a failed connection | `MDQM_TOKEN` is wrong or expired. Ask for a new one; tokens are per person. |
| Tools appear but a query is refused with `NOT_SUPPORTED` | This deployment has no approved semantic layer yet. The `fix` field says so. |
| No context in the prompt, tools work | The hook's interpreter is wrong or the server's `/context` is unreachable. Check `/config` and the debug log. |
| A corporate TLS proxy | The hook and the MCP client use the system trust store. Install the internal CA where the OS expects it; the plugin has no TLS setting of its own. |

## Layout

```
plugin/mdqm/
  .claude-plugin/plugin.json    manifest: name, and the Python-interpreter option
  .mcp.json                     the http MCP server entry (url/headers from env, alwaysLoad)
  hooks/hooks.json              UserPromptSubmit -> scripts/context_hook.py, exec form
  scripts/context_hook.py       stdlib only, Python 3.9+, fail-open, UTF-8 out
  skills/mdqm-query/SKILL.md    how a weak agent uses the five tools
```

`plugin/README.md` (this file) is not loaded by Claude Code; everything under `plugin/mdqm/` is.
