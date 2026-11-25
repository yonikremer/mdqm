@AGENTS.md

# Claude Code notes

- Start with `python scripts/next_tasks.py ready --priority P0,P1` and follow section 8 of AGENTS.md.
- Use plan mode for tasks in lanes C, S, M if the task touches more than three files; otherwise just implement.
- Prefer running the task's own test command while iterating; run `python scripts/check.py` before integrating.
- Windows: use PowerShell-compatible commands (no `&&`, no bash-isms) in anything you document or script.
