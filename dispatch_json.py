#!/usr/bin/env python3
# dispatch_json.py — JSON-based slot scheduling with deadline check + final miss rate logging
# BG dispatch to all slaves (nohup), wait-until-finished using host-done marker
# 2025-09-23 pm 7:02

import subprocess, sys, json, os, datetime, time
from pathlib import Path

# ===== Config =====
SLAVE_IPS = ["192.168.0.96", "192.168.0.53", "192.168.0.57"]

REMOTE_INPUT_DIR = "/home/pi/transcode_inputs"
REMOTE_JOBS_BASE = "/home/pi/transcode_jobs"

RESULTS_DIR = "plans_results"

SSH_OPTS = [
    "-o","ConnectTimeout=10",
    "-o","ServerAliveInterval=15",
    "-o","ServerAliveCountMax=3",
    "-o","BatchMode=yes",
    "-o","StrictHostKeyChecking=no",
    "-o","UserKnownHostsFile=/dev/null",
]

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)

def ssh(ip: str, *args: str) -> subprocess.CompletedProcess:
    return run(["ssh", f"pi@{ip}", *SSH_OPTS, *args])

def run_remote_script_bg(ip: str, script: str) -> subprocess.CompletedProcess:
    """
    원격에 스크립트 저장 후 nohup 백그라운드 실행. 즉시 반환.
    """
    remote = (
        "bash -lc "
        "'cat > /tmp/dispatch_plan.sh ; "
        "chmod +x /tmp/dispatch_plan.sh ; "
        "nohup bash /tmp/dispatch_plan.sh </dev/null >/dev/null 2>&1 & "
        "echo $!'"
    )
    return run(["ssh", f"pi@{ip}", *SSH_OPTS, remote], input=script)

# ---- JSON 기반 빌드 ----
def build_script_from_json(slot_jobs: dict, slot_limits: dict, ladder: list) -> str:
    """
    slot_jobs: { '0': [ {video, versions:[...]}, ... ], ... }
    slot_limits: { '0': 268.9, ... }  (초, 소수 올림)
    ladder: [ {ver,width,height,label}, ... ]
    """
    hdr = f"""#!/usr/bin/env bash
set -euo pipefail
INPUT_DIR="{REMOTE_INPUT_DIR}"
JOB_BASE="{REMOTE_JOBS_BASE}"
AGG="$JOB_BASE/_aggregate.log"
DONE_MARK="$JOB_BASE/_host_done"
mkdir -p "$JOB_BASE"
# 이전 결과 마커 지우기
rm -f "$DONE_MARK"
touch "$AGG"

echo "[DISPATCH] start on $(hostname) at $(date)" | tee -a "$AGG"
"""

    # 슬롯이 하나도 없어도 부트스트랩 + 완료마커 남기기
    if not slot_jobs:
        return hdr + 'echo "[INFO] no slots assigned for this host" | tee -a "$AGG"\n' + \
               'echo "[HOST_DONE] no slots" | tee -a "$AGG"\n' + \
               'touch "$DONE_MARK"\n'

    body = ""
    for slot in sorted(slot_jobs.keys(), key=lambda s: int(s)):
        jobs = slot_jobs[slot]
        limit = int(float(slot_limits[str(slot)]) + 0.999)

        body += f"""
# --- Slot {slot} (limit={limit}s) ---
SLOT_ELAPSED=0
"""

        for job in jobs:
            video = job["video"]
            versions = job.get("versions", [])
            if not isinstance(versions, (list, tuple)):
                versions = [versions]
            for ver in versions:
                lad = ladder[int(ver)]
                width, height, label = lad["width"], lad["height"], lad["label"]
                body += f"""
if (( SLOT_ELAPSED >= {limit} )); then
  echo "[MISS] slot {slot} deadline exceeded before {video} v{ver}" | tee -a "$AGG"
else
  F="$INPUT_DIR/{video}"
  OUTDIR="$JOB_BASE/slot{slot}/{video}"
  PROG="$OUTDIR/progress.log"
  mkdir -p "$OUTDIR"

  echo "[HOST] $(hostname) SLOT:{slot} VIDEO:{video} VER:{ver} {label} $(date)" | tee -a "$PROG" "$AGG"
  start=$(date +%s)
  ffmpeg -hide_banner -nostdin -y -i "$F" -s {width}x{height} -c:v libx264 -preset fast "$OUTDIR/{video}_v{ver}.mp4" 2>&1 \\
    | stdbuf -oL tr "\\r" "\\n" | tee -a "$PROG" "$AGG" >/dev/null || true
  end=$(date +%s)
  dur=$((end-start))
  SLOT_ELAPSED=$((SLOT_ELAPSED+dur))
  if (( SLOT_ELAPSED > {limit} )); then
    echo "[MISS] slot {slot} deadline exceeded after {video} v{ver} (dur=${{dur}}s, elapsed=$SLOT_ELAPSED)" | tee -a "$PROG" "$AGG"
  else
    echo "[OK] slot {slot} completed {video} v{ver} in ${{dur}}s (elapsed=$SLOT_ELAPSED/{limit})" | tee -a "$PROG" "$AGG"
  fi
fi
"""
        body += f'\necho "[SLOT] {slot} accounted=$SLOT_ELAPSED/{limit}" | tee -a "$AGG"\n'

    # 모든 슬롯 끝: 완료 마커/로그 남기기
    body += '\necho "[HOST_DONE] finished all slots at $(date)" | tee -a "$AGG"\n'
    body += 'touch "$DONE_MARK"\n'

    return hdr + body

# ---- 완료 대기 & 집계 ----
def is_host_running(ip: str) -> bool:
    """dispatch_plan.sh 가 아직 돌고있는지 검사."""
    r = ssh(ip, "pgrep", "-f", "dispatch_plan.sh")
    return r.returncode == 0  # 0이면 프로세스 있음

def has_host_done(ip: str) -> bool:
    """_host_done 마커가 생겼는지 검사."""
    r = ssh(ip, "test", "-f", f"{REMOTE_JOBS_BASE}/_host_done")
    return r.returncode == 0

def wait_all_hosts_finished(poll_sec: int = 5):
    """모든 슬레이브에서 dispatch_plan.sh 종료 + _host_done 생성까지 대기."""
    print("[WAIT] waiting for all hosts to finish ...")
    pending = set(SLAVE_IPS)
    while pending:
        done_now = []
        for ip in list(pending):
            running = is_host_running(ip)
            done = has_host_done(ip)
            if (not running) and done:
                print(f"[WAIT] {ip} done.")
                done_now.append(ip)
        for ip in done_now:
            pending.discard(ip)
        if pending:
            time.sleep(poll_sec)
    print("[WAIT] all hosts finished.")

def collect_miss_rate(policy_name: str, total_jobs: int):
    """[MISS] 로그 라인 수를 집계해 미스율 저장."""
    miss_jobs = 0
    for ip in SLAVE_IPS:
        r = ssh(ip, "grep", "-c", "\\[MISS\\]", f"{REMOTE_JOBS_BASE}/_aggregate.log")
        if r.returncode == 0 and r.stdout.strip().isdigit():
            miss_jobs += int(r.stdout.strip())
        else:
            print(f"[WARN] {ip}: miss count grep failed")

    miss_rate = miss_jobs / total_jobs if total_jobs > 0 else 0.0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR,
        f"{policy_name}_missrate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(out_path, "w") as f:
        f.write(f"policy={policy_name}\n")
        f.write(f"total_jobs={total_jobs}\n")
        f.write(f"miss_jobs={miss_jobs}\n")
        f.write(f"miss_rate={miss_rate:.4f}\n")
    print(f"[RESULT] {policy_name} miss rate saved -> {out_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: dispatch_json.py <plan.json>")
        return 1

    plan_path = Path(sys.argv[1])
    if not plan_path.exists():
        print(f"[ERROR] Plan file not found: {plan_path}")
        return 1

    with open(plan_path) as f:
        plan = json.load(f)

    policy_name = plan.get("policy", plan_path.stem)
    n_slots = int(plan["n_slots"])
    n_deadlines = int(plan["n_deadlines"])
    ladder = plan["ladder"]
    slot_jobs = plan["slots"]
    slot_limits = plan["time_limits"]

    # 전체 계획 건수(= versions 총합)
    total_jobs = 0
    for jobs in slot_jobs.values():
        for job in jobs:
            vers = job.get("versions", [])
            if not isinstance(vers, (list, tuple)):
                vers = [vers]
            total_jobs += len(vers)

    # 슬롯 블록 분배 (0–9 / 10–19 / 20–29)
    servers = len(SLAVE_IPS)
    block = max(1, n_slots // servers)  # 30//3 = 10
    slots_per_server = {ip: {} for ip in SLAVE_IPS}
    for s in range(n_slots):
        idx = min(s // block, servers - 1)
        ip = SLAVE_IPS[idx]
        slots_per_server[ip][str(s)] = slot_jobs[str(s)]

    print(f"[PLAN] policy={policy_name} n_slots={n_slots} servers={servers} block={block}")
    for i, ip in enumerate(SLAVE_IPS):
        print(f"[ASSIGN] server#{i+1} {ip}: {len(slots_per_server[ip])} slots")

    # 각 서버에 백그라운드로 전달
    for ip in SLAVE_IPS:
        script = build_script_from_json(slots_per_server[ip], slot_limits, ladder)
        r = run_remote_script_bg(ip, script)
        if r.returncode != 0:
            print(f"[ERROR] {ip}: dispatch failed")
            print("STDOUT:\n" + r.stdout)
            print("STDERR:\n" + r.stderr)
        else:
            pid = r.stdout.strip()
            print(f"[OK] {ip}: started bg job PID {pid} ({len(slots_per_server[ip])} slots)")

    print("\n[INFO] Watch logs on each slave:")
    for ip in SLAVE_IPS:
        print(f"  ssh pi@{ip} 'tail -F {REMOTE_JOBS_BASE}/_aggregate.log'")

    # === 여기서 끝까지 대기 ===
    wait_all_hosts_finished(poll_sec=5)

    # 모든 서버 완료 후 miss율 집계
    collect_miss_rate(policy_name, total_jobs)

if __name__ == "__main__":
    sys.exit(main() or 0)
