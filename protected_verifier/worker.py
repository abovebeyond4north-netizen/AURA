"""Standalone independent-process entrypoint; accepts only serialized whitelist."""
import json,sys,os,math
from dataclasses import asdict
from .os_limits import enforce_worker_limits
from .network_limits import enforce_network_limits
from .resource_limits import validate_budget
from .version_manager import validate_version

MAX_REQUEST_BYTES=65536
MAX_REQUEST_DEPTH=32
MAX_REQUEST_NODES=8192

def validate_json_depth(value):
    """Bound nested input complexity before constructing typed evaluator objects."""
    stack=[(value, 1)]
    visited=0
    while stack:
        item, depth=stack.pop()
        visited+=1
        if visited > MAX_REQUEST_NODES:
            raise ValueError("Evaluator request node count exceeds limit")
        if depth > MAX_REQUEST_DEPTH:
            raise ValueError("Evaluator request nesting exceeds limit")
        if isinstance(item, str):
            if any(0xD800 <= ord(ch) <= 0xDFFF for ch in item):
                raise ValueError("Evaluator request contains unpaired Unicode surrogate")
        if isinstance(item, dict):
            stack.extend((key, depth+1) for key in item)
            stack.extend((child, depth+1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth+1) for child in item)


def _close_range_all():
    """Close every non-stdio descriptor using the Linux kernel, without procfs."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    close_range = libc.close_range
    close_range.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_int)
    close_range.restype = ctypes.c_int
    if close_range(3, ctypes.c_uint(-1).value, 0) != 0:
        raise OSError(ctypes.get_errno(), 'close_range failed')


def close_inherited_descriptors():
    """Drop inherited handles before reading requests, including above soft limits.

    Prefer kernel range closure. On older or restricted hosts retain the existing
    fail-closed procfs inventory path; never substitute a soft-limit loop.
    """
    if sys.platform != 'linux':
        raise RuntimeError('Verifier descriptor isolation requires Linux')
    try:
        _close_range_all()
        return
    except (AttributeError, OSError):
        pass
    try:
        descriptors = [int(entry) for entry in os.listdir('/proc/self/fd')
                       if entry.isdecimal()]
    except OSError as exc:
        raise RuntimeError('Cannot inspect inherited descriptors') from exc
    for fd in descriptors:
        if fd > 2:
            try:
                os.close(fd)
            except OSError as exc:
                if exc.errno != 9:  # EBADF: enumeration's own directory descriptor
                    raise RuntimeError(f'Cannot close inherited descriptor {fd}') from exc


def verify_temporary_directory_support():
    """Fail before evaluation if syscall policy blocks required temp directories.

    This does not weaken the seccomp policy or claim filesystem isolation.
    """
    import tempfile
    try:
        with tempfile.TemporaryDirectory(prefix='aura-verifier-probe-'):
            pass
    except (OSError, RuntimeError) as exc:
        raise RuntimeError('Verifier temporary-directory capability unavailable under security policy') from exc


def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError("Duplicate JSON key")
        result[key]=value
    return result

def reject_constant(value):
    raise ValueError("Non-finite JSON constant")

def finite_float(value):
    parsed=float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite JSON number")
    return parsed

def read_request(stream):
    raw=stream.read(MAX_REQUEST_BYTES+1)
    if len(raw)>MAX_REQUEST_BYTES: raise ValueError("Oversized evaluator request")
    # Require one canonical wire encoding: json.loads(bytes) otherwise auto-detects UTF-16/32.
    # Reject BOMs and malformed UTF-8 before parsing untrusted evaluator data.
    try:
        decoded=raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("Evaluator request must be UTF-8") from exc
    if decoded.startswith("\ufeff"):
        raise ValueError("Evaluator request must not contain a BOM")
    payload=json.loads(decoded,object_pairs_hook=unique_object,parse_constant=reject_constant,parse_float=finite_float)
    if not isinstance(payload,dict) or set(payload)!={'config','modification','steps','replicates','parent','candidate'}:
        raise ValueError("Unexpected evaluator parameters")
    validate_json_depth(payload)
    validate_budget(payload["steps"],payload["replicates"])
    for key in ("parent","candidate"):
        validate_version(payload[key])
    if payload["parent"] == payload["candidate"]:
        raise ValueError("Candidate version must differ from parent")
    if not isinstance(payload["config"], dict):
        raise ValueError("Invalid configuration")
    if not isinstance(payload["modification"],dict) or set(payload["modification"]) != {"target", "operation", "parameters", "parent_version"}:
        raise ValueError("Invalid modification schema")
    if not isinstance(payload["modification"]["parameters"], dict):
        raise ValueError("Invalid modification parameters")
    if not isinstance(payload["modification"]["target"], str) or not isinstance(payload["modification"]["operation"], str):
        raise ValueError("Invalid modification operation")
    validate_version(payload["modification"]["parent_version"])
    if "parent_version" in payload["modification"] and payload["modification"]["parent_version"] != payload["parent"]:
        raise ValueError("Modification parent mismatch")
    return payload

def main():
    try:
        close_inherited_descriptors()
        enforce_worker_limits()
        enforce_network_limits()
        payload=read_request(sys.stdin.buffer)
        verify_temporary_directory_support()
        from aura.config import Config
        from aura.types import Modification
        from .evaluator import compare
        report=compare(Config.model_validate(payload['config']),Modification(**payload['modification']),
                       payload['steps'],payload['replicates'],payload['parent'],payload['candidate'])
        print(json.dumps({'ok':True,'report':asdict(report)},allow_nan=False))
    except Exception as exc:
        print(json.dumps({'ok':False,'error':type(exc).__name__+': '+str(exc)}))
        sys.exit(1)
if __name__=='__main__':main()
