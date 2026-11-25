# mdqm: Multi DB Query MCP

Start here: `AGENTS.md` (rules and protocol for coding agents), `TASKS.md` (backlog), `docs/DESIGN.md` (design).

```
python -m venv .venv
.venv\Scripts\activate         # Windows.  Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,pg]"
python scripts/check.py
python scripts/next_tasks.py ready --priority P0,P1
```
