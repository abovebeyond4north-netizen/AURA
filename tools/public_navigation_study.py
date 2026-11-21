"""Paired, immutable-input public navigation confirmation; not external blind evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

CONTROLLERS = ("EfficientRouteCamera", "PersistentPoseCamera", "ComposedRouteCamera")


def validate_manifest(m: dict) -> None:
    if m.get("protocol") != "AURA-public-confirmation-v1" or m.get("independent") is not False:
        raise ValueError("Unsupported protocol; public testing must not claim independence")
    if m.get("controllers") != list(CONTROLLERS) or m.get("action_cap") != 65:
        raise ValueError("Controller/order/action budget changed")
    seeds, layouts, noises, reverses = (m.get(k) for k in ("seeds", "layouts", "observation_noise", "reverse_after_key"))
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid or duplicated seeds")
    if not isinstance(layouts, list) or not layouts or len({tuple(x) for x in layouts}) != len(layouts):
        raise ValueError("Invalid or duplicated layouts")
    if any(len(x) != 2 or any(type(v) is not int for v in x) or x[0] < 7 or x[0] % 2 == 0 or not 1 <= x[1] < x[0]-1 for x in layouts):
        raise ValueError("Invalid layout geometry")
    if not isinstance(noises, list) or not noises or any(type(x) not in (float, int) or not 0 <= x <= .5 for x in noises) or len(set(noises)) != len(noises):
        raise ValueError("Invalid observation corruption")
    if reverses != [False, True]:
        raise ValueError("Movement-regime coverage changed")


def summarize(rows: list[dict], manifest: dict) -> dict:
    counts = Counter(r["controller"] for r in rows)
    keys = Counter((r["seed"], r["size"], r["column"], r["reverse"], r["noise"], r["controller"]) for r in rows)
    expected = {(seed,size,col,rev,noise,name) for seed in manifest["seeds"]
                for size,col in manifest["layouts"] for rev in manifest["reverse_after_key"]
                for noise in manifest["observation_noise"] for name in CONTROLLERS}
    if set(keys) != expected or any(n != 1 for n in keys.values()):
        raise ValueError("Incomplete or duplicate paired trials")
    if any(len(r["actions"]) > manifest["action_cap"] or any(type(a) is not int or a < 0 or a > 4 for a in r["actions"]) for r in rows):
        raise ValueError("Invalid action trace or exceeded budget")
    data = {}
    for name in CONTROLLERS:
        arm = [r for r in rows if r["controller"] == name]
        wins = sum(r["success"] is True for r in arm)
        data[name] = {"successes": wins, "trials": counts[name], "success_rate": wins / len(arm),
                      "pose_error_trials": sum(bool(r["pose_errors"]) for r in arm),
                      "actions_total": sum(len(r["actions"]) for r in arm),
                      "failures_by_reason": dict(Counter(str(r["reason"]) for r in arm if not r["success"]))}
    k0 = lambda r: (r["seed"],r["size"],r["column"],r["reverse"],r["noise"])
    baselines = {k0(r): r for r in rows if r["controller"] == "PersistentPoseCamera"}
    composed = {k0(r): r for r in rows if r["controller"] == "ComposedRouteCamera"}
    gain = sum(bool(composed[k]["success"]) and not bool(baselines[k]["success"]) for k in baselines)
    loss = sum(not bool(composed[k]["success"]) and bool(baselines[k]["success"]) for k in baselines)
    return {"by_controller": data, "paired_composed_vs_persistent": {"improved": gain, "regressed": loss,
             "net": gain-loss}, "total_trials": len(rows), "independent": False,
            "acceptance_promoted": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    raw = a.manifest.read_bytes()
    manifest = json.loads(raw)
    validate_manifest(manifest)
    from aura.exploration.efficient_routes import EfficientRouteCamera
    from aura.exploration.persistent_pose import PersistentPoseCamera
    from aura.exploration.composed_routes import ComposedRouteCamera
    from experiments.persistent_pose_study import trial
    controllers = (EfficientRouteCamera,PersistentPoseCamera,ComposedRouteCamera)
    rows = [trial(seed,size,col,rev,noise,cls) for seed in manifest["seeds"]
            for size,col in manifest["layouts"] for rev in manifest["reverse_after_key"]
            for noise in manifest["observation_noise"] for cls in controllers]
    summary = summarize(rows,manifest)
    result = {"protocol": manifest["protocol"], "manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "summary": summary, "rows": rows}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        p.error("Refusing to overwrite a prior evaluation")
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"manifest_sha256":result["manifest_sha256"], "summary":summary},sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
