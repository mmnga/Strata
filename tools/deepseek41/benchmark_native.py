#!/usr/bin/env python3
"""Measure native Strata decode and resident-model I/O with fixed token IDs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time


def counters(pid):
    root = Path(f"/proc/{pid}")
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    io = {k: int(v) for k, v in (line.split(":") for line in (root / "io").read_text().splitlines())}
    return {"major_faults": int(fields[9]), "disk_read_bytes": io["read_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", default="3")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--suite", choices=["all", "standard", "heldout"], default="all")
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--check-batch", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cases = [c for c in json.loads(args.cases.read_text()) if args.suite == "all" or c["suite"] == args.suite]
    depths = [int(n) for n in args.depths.split(",")]
    if not cases or any(n < 0 or n > 5 for n in depths):
        parser.error("need cases and depths in 0..5")
    environment = dict(os.environ, **config.get("env", {}))
    environment["CUDA_MODULE_LOADING"] = "EAGER"
    command = [config["exe"], *config["args"], "--serve"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    result = {"command": command, "environment": config.get("env", {}),
              "binary_sha256": hashlib.sha256(Path(config["exe"]).read_bytes()).hexdigest(),
              "timed_token_rule": "output_tokens - 1", "cases": []}
    # Read bytes directly: buffered readline plus select can hide already-read
    # lines and wait forever when an engine becomes idle after DONE/PONG.
    with log_path.open("wb", buffering=0) as log:
        process = subprocess.Popen(command, cwd=config["cwd"], env=environment,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
        pending = bytearray()

        def line(timeout=1800):
            deadline = time.monotonic() + timeout
            while b"\n" not in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                    raise TimeoutError("native engine response timeout")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f"native engine exited: {process.poll()}; see {log_path}")
                pending.extend(chunk)
            at = pending.index(b"\n")
            value = bytes(pending[:at+1]); del pending[:at+1]
            log.write(value)
            return value.decode().strip()

        def send(value):
            process.stdin.write((value + "\n").encode()); process.stdin.flush()

        def settle():
            send("PING")
            while line() != "PONG":
                pass

        def request(case, maximum, depth):
            settle()
            before = counters(process.pid)
            send(f"GEN {maximum} temperature=0 draft_depth={depth} " + ",".join(map(str, case["ids"])))
            ids = []
            while True:
                value = line()
                if value.startswith("ERR "):
                    raise RuntimeError(value)
                if value.startswith("T "):
                    ids.append(int(value[2:]))
                if value.startswith("DONE "):
                    done = value.split(); break
            settle()
            after = counters(process.pid)
            timed = max(0, len(ids)-1)
            ms = float(done[4])
            return {"suite": case["suite"], "language": case["language"], "draft_depth": depth,
                    "prompt_ids": case["ids"], "output_ids": ids, "output_tokens": len(ids),
                    "timed_tokens": timed, "decode_ms": ms, "tok_s": timed*1000/ms if ms else 0,
                    "prompt_ms": float(done[3]), "finish": done[5], "accepted": int(done[6]),
                    "offered": int(done[7]), **{k: after[k]-before[k] for k in before}}

        try:
            while not line().startswith("READY "):
                pass
            print(f"READY native Strata pid={process.pid}", flush=True)
            if args.check_batch:
                seed = cases[0]["ids"]
                ids = (seed * (136//len(seed)+1))[:136]
                send("CHECK " + ",".join(map(str, ids)))
                while True:
                    value = line()
                    if value.startswith("CHECK FAIL") or value.startswith("ERR "):
                        raise RuntimeError(value)
                    if value.startswith("CHECK PASS"):
                        print(value, flush=True); break
            for depth in depths:
                if args.warmup:
                    for case in cases[:2]:
                        request(case, args.warmup, depth)
                for case in cases:
                    measured = request(case, args.tokens or case["max_tokens"], depth)
                    result["cases"].append(measured)
                    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                    print(json.dumps({k: v for k, v in measured.items() if not k.endswith("_ids")}, ensure_ascii=False), flush=True)
            for depth in depths:
                rows = [c for c in result["cases"] if c["draft_depth"] == depth]
                print(f"depth={depth} weighted_tok_s={1000*sum(c['timed_tokens'] for c in rows)/sum(c['decode_ms'] for c in rows):.6f}", flush=True)
        finally:
            if process.poll() is None:
                try:
                    send("QUIT"); process.wait(timeout=60)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    process.terminate(); process.wait(timeout=60)
        if process.returncode:
            raise RuntimeError(f"native engine exit {process.returncode}")


if __name__ == "__main__":
    main()
