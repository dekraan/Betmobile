import os
import subprocess
from datetime import datetime

BASE = r"C:\Users\Gebruiker\Documents\Betmobile"
PY = r"C:\Users\Gebruiker\AppData\Local\Programs\Python\Python313\python.exe"
BOOTSTRAP = os.path.join(BASE, "betmobile_bootstrap.py")
LOG = os.path.join(BASE, "logs", "daily_maintenance_log.txt")

env = os.environ.copy()
env["RUN_MODE"] = "MAINTENANCE"

# Zelfde horizon als snapshots → consistent dataset
env["HORIZON_PAST_DAYS"] = "1"
env["HORIZON_FUTURE_DAYS"] = "6"

# 🔥 Hier zit het verschil:
env["LOAD_EVENTS"] = "1"
env["LOAD_STATS"] = "1"

cmd = [PY, BOOTSTRAP]

with open(LOG, "a", encoding="utf-8") as f:
    f.write(f"\n[{datetime.now().isoformat()}] DAILY MAINTENANCE RUN\n")
    f.write("CMD: " + " ".join(cmd) + "\n")
    f.write(
        f"ENV: RUN_MODE={env['RUN_MODE']} "
        f"HORIZON_PAST_DAYS={env['HORIZON_PAST_DAYS']} "
        f"HORIZON_FUTURE_DAYS={env['HORIZON_FUTURE_DAYS']} "
        f"LOAD_EVENTS={env['LOAD_EVENTS']} "
        f"LOAD_STATS={env['LOAD_STATS']}\n\n"
    )

    proc = subprocess.Popen(
        cmd,
        cwd=BASE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in proc.stdout:
        f.write(line)

    rc = proc.wait()
    f.write(f"\nExit code: {rc}\n")
    f.write("------------------------------\n")