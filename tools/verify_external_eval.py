"""Fail-closed intake of externally signed evaluation evidence.

A detached signature is necessary but not sufficient for genuine evaluator
independence. This script NEVER promotes any acceptance criterion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from tools.public_navigation_study import summarize, validate_manifest

def verify(receipt_path: Path, manifest_path: Path, signature_path: Path,
           key_path: Path, trusted_fingerprint: str) -> dict:
    expected = trusted_fingerprint.strip().upper()
    if not re.fullmatch(r"[0-9A-F]{40}", expected):
        raise ValueError("Require an independently established 40-hex GPG fingerprint")
    for p in (receipt_path, manifest_path, signature_path, key_path):
        if not p.is_file():
            raise ValueError(f"Missing required external evidence: {p}")
    with tempfile.TemporaryDirectory(prefix="aura-external-key-") as home:
        subprocess.run(["gpg", "--batch", "--no-tty", "--homedir", home,
                        "--import", str(key_path.resolve())],
                       check=True, capture_output=True, text=True, timeout=30)
        result = subprocess.run(["gpg", "--batch", "--no-tty", "--homedir", home,
                                 "--status-fd", "1", "--verify",
                                 str(signature_path.resolve()), str(receipt_path.resolve())],
                                capture_output=True, text=True, timeout=30)
        fingerprints = re.findall(r"\[GNUPG:\] VALIDSIG ([A-Fa-f0-9]{40})\b", result.stdout)
        if result.returncode != 0 or expected not in [x.upper() for x in fingerprints]:
            raise ValueError("Invalid signature or untrusted signing fingerprint")
    receipt = json.loads(receipt_path.read_bytes())
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    schema = {**manifest, "independent": False, "protocol": "AURA-public-confirmation-v1"}
    validate_manifest(schema)
    if receipt.get("protocol") != "AURA-external-v1":
        raise ValueError("Unsupported receipt protocol")
    if not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("source_commit", ""))):
        raise ValueError("Receipt must bind an immutable source commit")
    digest = hashlib.sha256(manifest_raw).hexdigest()
    if receipt.get("manifest_sha256") != digest:
        raise ValueError("Manifest does not match signed receipt")
    summary = summarize(receipt["rows"], schema)
    if receipt.get("summary") != summary:
        raise ValueError("Signed summary does not match signed raw trials")
    return {"signature_valid": True, "signer_fingerprint": expected,
            "source_commit": receipt["source_commit"], "manifest_sha256": digest,
            "total_trials": summary["total_trials"], "independence_verified": False,
            "acceptance_promoted": False,
            "review_required": "Independent key ownership, blind manifest control, no evaluator conflict, and replay"}

def main():
    p = argparse.ArgumentParser()
    for name in ("receipt", "manifest", "signature", "public-key"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--trusted-fingerprint", required=True)
    args = p.parse_args()
    print(json.dumps(verify(args.receipt, args.manifest, args.signature,
                            args.public_key, args.trusted_fingerprint), indent=2))

if __name__ == "__main__":
    main()
