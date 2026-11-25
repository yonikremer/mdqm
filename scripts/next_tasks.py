"""Pick, claim and release tasks from tasks.json.

    python scripts/next_tasks.py ready [--priority P0,P1] [--lane C]   # tasks whose deps are done and unclaimed
    python scripts/next_tasks.py show T-ID                             # full task text
    python scripts/next_tasks.py claim T-ID [--agent NAME]             # atomic; fails if already claimed
    python scripts/next_tasks.py release T-ID
    python scripts/next_tasks.py status                                # done / claimed / ready / blocked counts

A task is DONE when progress/<ID>.md exists on main with the line `status: done` (see AGENTS.md).
Claims live in the shared .git directory (not in the working tree), so all worktrees see them.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from _common import common_dir, die, done_ids, load_tasks, main_root

STALE_HOURS = 4.0


def claims_dir():
    d = common_dir() / "mdqm-claims"
    d.mkdir(exist_ok=True)
    return d


def claimed() -> dict[str, dict]:
    out = {}
    for f in claims_dir().glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out[f.stem] = {"agent": "?", "ts": 0}
    return out


def is_stale(c: dict) -> bool:
    return (time.time() - c.get("ts", 0)) > STALE_HOURS * 3600


def ready_tasks(tasks, priorities=None, lane=None):
    done, cl = done_ids(), claimed()
    res = []
    for t in tasks.values():
        tid = t["id"]
        if tid in done:
            continue
        if tid in cl and not is_stale(cl[tid]):
            continue
        if any(d not in done for d in t["depends_on"]):
            continue
        if priorities and t["priority"] not in priorities:
            continue
        if lane and t["lane"] != lane:
            continue
        res.append(t)
    return sorted(res, key=lambda t: (t["wave"], t["priority"], t["id"]))


def cmd_ready(a, tasks):
    pr = set(a.priority.split(",")) if a.priority else None
    r = ready_tasks(tasks, pr, a.lane)
    if not r:
        print("(no ready tasks)")
    for t in r:
        print(f'{t["id"]:6} wave{t["wave"]} {t["priority"]} lane={t["lane"]:2} {t["title"]}')


def cmd_show(a, tasks):
    t = tasks.get(a.id) or die(f"unknown task {a.id}")
    print(f'{t["id"]}: {t["title"]}   [phase {t["phase"]}, priority {t["priority"]}, wave {t["wave"]}]')
    print("depends_on:", ", ".join(t["depends_on"]) or "-")
    print("owns:", ", ".join(t["owns"]))
    print("\n" + t["description"])
    print("\nAcceptance:")
    for x in t["acceptance"]:
        print(" -", x)
    print("\nTest command:\n ", t["test"])
    if t.get("notes"):
        print("\nNotes:", t["notes"])


def cmd_claim(a, tasks):
    t = tasks.get(a.id) or die(f"unknown task {a.id}")
    done = done_ids()
    if a.id in done:
        die(f"{a.id} is already done", 2)
    missing = [d for d in t["depends_on"] if d not in done]
    if missing:
        die(f"{a.id} is blocked by: {', '.join(missing)}", 2)
    path = claims_dir() / f"{a.id}.json"
    payload = json.dumps({"agent": a.agent, "ts": time.time()})
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        c = json.loads(path.read_text(encoding="utf-8") or "{}")
        if is_stale(c):
            print(f"WARNING: stealing stale claim on {a.id} from {c.get('agent')}")
            path.write_text(payload, encoding="utf-8")
            print(f"claimed {a.id}")
            return
        die(f"{a.id} already claimed by {c.get('agent')}", 3)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(payload)
    print(f"claimed {a.id}")


def cmd_release(a, tasks):
    p = claims_dir() / f"{a.id}.json"
    if p.exists():
        p.unlink()
    print(f"released {a.id}")


def cmd_status(a, tasks):
    done, cl = done_ids(), claimed()
    ready = {t["id"] for t in ready_tasks(tasks)}
    blocked = [i for i in tasks if i not in done and i not in cl and i not in ready]
    print(f"done={len(done)} claimed={len(cl)} ready={len(ready)} blocked={len(blocked)} total={len(tasks)}")
    for i, c in sorted(cl.items()):
        print(f"  claimed {i} by {c.get('agent')}{' (STALE)' if is_stale(c) else ''}")
    print(f"(progress dir: {main_root() / 'progress'})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ready")
    r.add_argument("--priority")
    r.add_argument("--lane")
    for name in ("show", "claim", "release"):
        s = sub.add_parser(name)
        s.add_argument("id")
        if name == "claim":
            s.add_argument("--agent", default=os.environ.get("MDQM_AGENT", "agent"))
    sub.add_parser("status")
    a = p.parse_args()
    tasks = load_tasks()
    {"ready": cmd_ready, "show": cmd_show, "claim": cmd_claim, "release": cmd_release, "status": cmd_status}[a.cmd](a, tasks)


if __name__ == "__main__":
    main()
