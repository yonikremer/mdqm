# Prompt to give each overnight agent (copy as is; one agent per terminal/thread)

You are a coding agent on the `mdqm` repository (current directory). First read `AGENTS.md` completely, then `docs/DESIGN.md`.
Follow section 8 of AGENTS.md (the overnight protocol) exactly:
1. Run `python scripts/next_tasks.py ready --priority P0,P1`, pick the first task, claim it with `python scripts/next_tasks.py claim <ID> --agent <your-name>`. If the claim fails, pick another.
2. Work in your own git worktree on branch `task/<ID>`. Read the task with `python scripts/next_tasks.py show <ID>`.
3. Implement it, write the tests, make the task's test command and `python scripts/check.py` pass. Never weaken tests or skip them to get green.
4. Write `progress/<ID>.md` in the format from AGENTS.md, commit, run `python scripts/integrate.py <ID>`.
5. Repeat from step 1 until there are no ready P0/P1 tasks, or you have worked about 8 hours, or you are repeatedly blocked.
If a task is blocked or exceeds ~90 minutes, write `status: blocked`, release the claim and take another task.
Do not ask the user questions: choose the conservative option, record the decision in your progress file, continue.
Do not push to any remote. Do not touch anything outside this repository and its `mdqm_*` Docker resources.
