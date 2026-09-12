"""Resumable experiment supervisor.

Start it any time (after a reboot, after closing the lid, after closing the
Claude app): it skips finished work, adopts jobs that are already running,
and relaunches the rest. Every job it starts is itself resumable, so nothing
finished is ever recomputed and at most the last unsaved piece is repeated.

  python pipeline.py            run until all steps in pipeline_steps.py are done
  python pipeline.py --status   print the state of every step and exit

Files (all under results/exp_logs/):
  pipeline.log          supervisor log (launches, completions, failures)
  pipe_<step>.log       stdout/stderr of each step
  pipeline_status.json  current state of every step
  pipeline.lock         PID of the running supervisor (prevents two copies)
"""
import datetime
import importlib
import json
import os
import subprocess
import sys
import time

import psutil

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(ROOT, "results", "exp_logs")
os.makedirs(LOGDIR, exist_ok=True)
LOG = os.path.join(LOGDIR, "pipeline.log")
LOCK = os.path.join(LOGDIR, "pipeline.lock")
STATUS = os.path.join(LOGDIR, "pipeline_status.json")
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
LANES = ("train", "eval")
MAX_FAILS = 3          # consecutive failures before a step is parked
POLL_S = 30
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

sys.path.insert(0, ROOT)
import pipeline_steps  # noqa: E402


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _cmdline(p):
    try:
        return " ".join(p.cmdline()).replace("\\", "/")
    except (psutil.Error, OSError):
        return ""


def find_running(sig):
    """PID of a live python process whose command line contains every substring in sig."""
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name"]):
        if p.pid == me or "python" not in (p.info["name"] or "").lower():
            continue
        c = _cmdline(p)
        if c and all(s in c for s in sig):
            return p.pid
    return None


def acquire_lock():
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
            if psutil.pid_exists(pid) and "pipeline.py" in _cmdline(psutil.Process(pid)):
                print(f"pipeline already running (PID {pid}); exiting.")
                sys.exit(0)
        except (ValueError, psutil.Error, OSError):
            pass
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))


def safe(fn, default=False):
    try:
        return bool(fn())
    except Exception:
        return default


def status_table(S, running, fails):
    rows = {}
    for s in S:
        if s["name"] in running:
            st = "running"
        elif safe(s["done"]):
            st = "done"
        elif fails.get(s["name"], 0) >= MAX_FAILS:
            st = "parked (failed)"
        elif not safe(s["ready"]):
            st = "waiting for inputs"
        else:
            st = "queued"
        rows[s["name"]] = {"lane": s["lane"], "state": st, "fails": fails.get(s["name"], 0)}
    return rows


def main():
    if "--status" in sys.argv:
        importlib.reload(pipeline_steps)
        for k, v in status_table(pipeline_steps.steps(), {}, {}).items():
            print(f"{k:28s} {v['lane']:6s} {v['state']}")
        return
    acquire_lock()
    log(f"supervisor started (PID {os.getpid()})")
    running = {}   # lane -> dict(step, proc or pid, fh, t0, adopted)
    fails = {}
    try:
        while True:
            importlib.reload(pipeline_steps)
            S = pipeline_steps.steps()
            by_name = {s["name"]: s for s in S}

            # 1. reap finished jobs
            for lane, job in list(running.items()):
                name = job["name"]
                if job["adopted"]:
                    alive = psutil.pid_exists(job["pid"])
                    rc = None if alive else 0
                else:
                    rc = job["proc"].poll()
                if rc is None:
                    continue
                if job.get("fh"):
                    job["fh"].close()
                del running[lane]
                mins = (time.time() - job["t0"]) / 60
                if name in by_name and safe(by_name[name]["done"]):
                    fails.pop(name, None)
                    log(f"DONE    {name} ({mins:.0f} min)")
                elif job["adopted"]:
                    log(f"ENDED   {name} (adopted process exited; will relaunch if unfinished)")
                else:
                    fails[name] = fails.get(name, 0) + 1
                    log(f"FAILED  {name} rc={rc} after {mins:.0f} min "
                        f"(attempt {fails[name]}/{MAX_FAILS}; see pipe_{name}.log)")

            # 2. finished?
            pending = [s for s in S if not safe(s["done"])]
            if not pending and not running:
                log("ALL STEPS DONE")
                break

            # 3. launch / adopt one job per free lane
            busy = {j["name"] for j in running.values()}
            for lane in LANES:
                if lane in running:
                    continue
                for s in pending:
                    if s["lane"] != lane or s["name"] in busy:
                        continue
                    if fails.get(s["name"], 0) >= MAX_FAILS or not safe(s["ready"]):
                        continue
                    pid = find_running(s["sig"])
                    if pid:
                        running[lane] = dict(name=s["name"], pid=pid, adopted=True, t0=time.time())
                        log(f"ADOPT   {s['name']} (already running as PID {pid})")
                    else:
                        fh = open(os.path.join(LOGDIR, f"pipe_{s['name']}.log"), "a", encoding="utf-8")
                        fh.write(f"\n===== launch {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
                        fh.flush()
                        proc = subprocess.Popen([PY] + s["cmd"], cwd=ROOT, stdout=fh,
                                                stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
                        running[lane] = dict(name=s["name"], proc=proc, fh=fh, adopted=False,
                                             t0=time.time())
                        log(f"LAUNCH  {s['name']} (PID {proc.pid})")
                    busy.add(s["name"])
                    break

            with open(STATUS + ".tmp", "w") as f:
                json.dump({"updated": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                           "steps": status_table(S, busy, fails)}, f, indent=1)
            os.replace(STATUS + ".tmp", STATUS)
            if not running and all(fails.get(s["name"], 0) >= MAX_FAILS or not safe(s["ready"])
                                   for s in pending):
                log("nothing runnable (remaining steps failed or wait for inputs that cannot "
                    "appear); stopping")
                break
            time.sleep(POLL_S)
    finally:
        try:
            if open(LOCK).read().strip() == str(os.getpid()):
                os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    main()
