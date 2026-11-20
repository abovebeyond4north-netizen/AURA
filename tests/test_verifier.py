import json
import subprocess,sys
from aura.types import Modification
from aura.config import Config
from protected_verifier.promotion import decide
from protected_verifier.resource_limits import validate_budget
from protected_verifier.metrics import paired_bootstrap

def test_promotion_not_granted_for_small_sample():
    report=decide('v1','v0',[.05]*2,[.1]*2,[.05]*2,{'training':4})
    assert not report.accepted and report.reason=='INCONCLUSIVE_INSUFFICIENT_SEEDS'
    assert paired_bootstrap([.1]*5)[0]>.09

def test_policy_rejects_untrusted_parameters():
    import pytest
    with pytest.raises(ValueError): validate_budget(0,5)
    with pytest.raises(ValueError): validate_budget(50,21)

def test_verifier_runs_in_subprocess_fails_closed_on_forbidden_modification():
    bad={'config':Config().model_dump(), 'modification':{
        'target':'protected_verifier','operation':'set','parameters':{'value':99},'parent_version':'v0'},
        'steps':5,'replicates':1,'parent':'v0','candidate':'bad'}
    result=subprocess.run([sys.executable,'-m','protected_verifier.worker'],input=json.dumps(bad),capture_output=True,text=True,timeout=25)
    assert result.returncode!=0 and json.loads(result.stdout)['ok'] is False


def test_promotion_rejects_corrupt_evidence():
    import pytest
    valid=([.01]*5,[.1]*5,[.09]*5)
    for gains,old,new in [
        ([],[],[]),
        ([.01]*5,[.1]*4,[.09]*5),
        ([float('nan')]*5,[.1]*5,[.09]*5),
        ([.01]*5,[float('inf')]*5,[.09]*5),
        ([.01]*5,[-.1]*5,[-.11]*5),
        ([.02]*5,[.1]*5,[.09]*5),
        ([True]*5,[.1]*5,[.09]*5),
    ]:
        with pytest.raises(ValueError):
            decide('v1','v0',gains,old,new,{'steps':5})
    assert decide('v1','v0',*valid,{'steps':5}).accepted


def test_worker_request_rejects_budget_coercion_and_ambiguous_json():
    import io, pytest
    from protected_verifier.worker import read_request, MAX_REQUEST_BYTES
    payload={'config':{},'modification':{'target':'learning_rate','operation':'multiply','parameters':{'value':1.1},'parent_version':'v0'},'steps':5,'replicates':1,'parent':'v0','candidate':'v1'}
    def parse(p): return read_request(io.BytesIO(json.dumps(p).encode()))
    assert parse(payload)==payload
    for key in ('steps','replicates'):
        for invalid in (True, 5.9, '5', None, float('nan'), float('inf')):
            with pytest.raises(ValueError): parse(dict(payload, **{key:invalid}))
    raw=json.dumps(payload).encode()
    for malformed in (raw[:-1]+b',"steps":5}', raw+b' '*MAX_REQUEST_BYTES, b'[]'):
        with pytest.raises(ValueError): read_request(io.BytesIO(malformed))
    for invalid in (True, 5.9, '5'):
        with pytest.raises(ValueError): validate_budget(invalid,1)


def test_worker_rejects_coercion_at_process_boundary():
    bad={'config':Config().model_dump(),'modification':{'target':'learning_rate','operation':'multiply','parameters':{'value':1.1},'parent_version':'v0'},'steps':5.9,'replicates':1,'parent':'v0','candidate':'v1'}
    result=subprocess.run([sys.executable,'-m','protected_verifier.worker'],input=json.dumps(bad),capture_output=True,text=True,timeout=25)
    assert result.returncode!=0
    assert json.loads(result.stdout)=={'ok':False,'error':'ValueError: Invalid evaluation steps'}


def test_kernel_worker_limits_in_child(tmp_path):
    import os, pytest
    pytest.importorskip('resource')
    probe = '''
import json, os, resource
from protected_verifier.os_limits import enforce_worker_limits, FILE_BYTES
resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
enforce_worker_limits()
result = {name: list(resource.getrlimit(getattr(resource, name))) for name in
          ('RLIMIT_CPU', 'RLIMIT_FSIZE', 'RLIMIT_NOFILE', 'RLIMIT_CORE')}
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (129, 129))
except (ValueError, PermissionError):
    result['raise_denied'] = True
with open(os.environ['PROBE_FILE'], 'wb') as f:
    f.truncate(FILE_BYTES)
    try:
        f.truncate(FILE_BYTES + 1)
    except OSError:
        result['file_denied'] = True
print(json.dumps(result))
'''
    result = subprocess.run([sys.executable, '-c', probe], text=True, capture_output=True,
                            env={**os.environ, 'PROBE_FILE':str(tmp_path/'probe')}, timeout=20, check=True)
    data = json.loads(result.stdout)
    assert data['RLIMIT_CPU'][0] == data['RLIMIT_CPU'][1] <= 600
    assert data['RLIMIT_FSIZE'] == [16777216, 16777216]
    assert data['RLIMIT_NOFILE'] == [128, 128]
    assert data['RLIMIT_CORE'] == [0, 0]
    assert data['raise_denied'] and data['file_denied']


def test_worker_refuses_evaluation_if_limits_fail(monkeypatch, capsys):
    import pytest
    from protected_verifier import worker
    def fail():
        raise OSError('limit setup failed')
    monkeypatch.setattr(worker, 'enforce_worker_limits', fail)
    monkeypatch.setattr(worker, 'close_inherited_descriptors', lambda: None)
    with pytest.raises(SystemExit) as exc:
        worker.main()
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out) == {'ok':False,'error':'OSError: limit setup failed'}


def test_valid_worker_request_under_kernel_limits():
    """Fail closed until tempfile compatibility is restored under seccomp."""
    import pytest
    from aura.self_improvement.runner import evaluate_candidate
    with pytest.raises(RuntimeError, match='temporary-directory capability unavailable under security policy'):
        evaluate_candidate(Config(), Modification('learning_rate','multiply',{'value':1.1},'v0'),steps=5,replicates=1)


def test_kernel_cpu_limit_terminates_busy_child():
    import signal, pytest
    pytest.importorskip('resource')
    probe = '''
import resource
from protected_verifier.os_limits import enforce_worker_limits
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
enforce_worker_limits()
while True:
    pass
'''
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, timeout=15)
    assert result.returncode == -signal.SIGKILL


def test_network_filter_denies_socket_creation_in_worker_and_descendant():
    probe = '''
import errno, socket, subprocess, sys
from protected_verifier.network_limits import enforce_network_limits
enforce_network_limits()
for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
    try:
        socket.socket(family)
    except OSError as exc:
        assert exc.errno == errno.EPERM
    else:
        raise AssertionError('socket creation allowed')
child = subprocess.run([sys.executable, '-c', 'import socket; socket.socket()'], capture_output=True)
assert child.returncode != 0 and b'PermissionError' in child.stderr
print('network denied')
'''
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=20, check=True)
    assert result.stdout.strip() == 'network denied'


def test_worker_refuses_evaluation_if_network_filter_fails(monkeypatch, capsys):
    import pytest
    from protected_verifier import worker
    monkeypatch.setattr(worker, 'enforce_worker_limits', lambda: None)
    def fail():
        raise RuntimeError('network setup failed')
    monkeypatch.setattr(worker, 'enforce_network_limits', fail)
    monkeypatch.setattr(worker, 'close_inherited_descriptors', lambda: None)
    with pytest.raises(SystemExit) as exc:
        worker.main()
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out) == {'ok': False, 'error': 'RuntimeError: network setup failed'}


def test_seccomp_denies_filesystem_mutation_and_inherits_to_child(tmp_path):
    """Check actual kernel denial; no assertion about openat write confinement."""
    import os
    probe = '''
import errno, os, subprocess, sys
from protected_verifier.network_limits import enforce_network_limits
enforce_network_limits()
path = os.environ['AURA_PROBE_PATH']
for operation in (lambda: os.unlink(path), lambda: os.rename(path, path + '.renamed'),
                  lambda: os.mkdir(path + '.dir'), lambda: os.chmod(path, 0o600),
                  lambda: os.truncate(path, 0)):
    try:
        operation()
    except OSError as exc:
        assert exc.errno == errno.EPERM, exc
    else:
        raise AssertionError('filesystem mutation unexpectedly allowed')
child = subprocess.run([sys.executable, '-c', 'import os,sys; os.unlink(sys.argv[1])', path], capture_output=True)
assert child.returncode != 0 and b'PermissionError' in child.stderr
print('mutations denied')
'''
    target = tmp_path / 'sentinel'
    target.write_text('unchanged')
    result = subprocess.run([sys.executable, '-c', probe], text=True, capture_output=True,
                            env={**os.environ, 'AURA_PROBE_PATH': str(target)}, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'mutations denied'
    assert target.read_text() == 'unchanged'


def test_seccomp_filesystem_rules_fail_closed_on_missing_syscall(monkeypatch):
    """A failed filter installation must not reach evaluator imports."""
    import pytest
    from protected_verifier import worker
    monkeypatch.setattr(worker, 'enforce_worker_limits', lambda: None)
    monkeypatch.setattr(worker, 'enforce_network_limits', lambda: (_ for _ in ()).throw(RuntimeError('filter failure')))
    monkeypatch.setattr(worker, 'read_request', lambda stream: pytest.fail('request processed after failed filter'))
    monkeypatch.setattr(worker, 'close_inherited_descriptors', lambda: None)
    with pytest.raises(SystemExit) as exc:
        worker.main()
    assert exc.value.code == 1


def test_worker_closes_inherited_high_numbered_descriptors(tmp_path):
    """Verify real inherited descriptor removal, including fd above soft limit."""
    import os
    import resource
    sentinel = tmp_path / 'secret'
    sentinel.write_text('private')
    probe = r'''
import os, sys
from protected_verifier.worker import close_inherited_descriptors
fd = int(sys.argv[1])
assert os.read(fd, 7) == b'private'
os.lseek(fd, 0, 0)
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
# Simulate absent procfs without changing the process filesystem.
os.listdir = lambda path: (_ for _ in ()).throw(FileNotFoundError(path))
close_inherited_descriptors()
try:
    os.fstat(fd)
except OSError as exc:
    assert exc.errno == 9
else:
    raise AssertionError('inherited descriptor still accessible')
assert os.fstat(0) and os.fstat(1) and os.fstat(2)
print('closed')
'''
    fd = os.open(sentinel, os.O_RDONLY)
    try:
        high_fd = os.dup2(fd, 220)
        try:
            result = subprocess.run([sys.executable, '-c', probe, str(high_fd)],
                                    pass_fds=(high_fd,), capture_output=True, text=True,
                                    timeout=20)
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == 'closed'
        finally:
            os.close(high_fd)
    finally:
        os.close(fd)


def test_worker_descriptor_procfs_fallback():
    probe = r"""
import os
from protected_verifier import worker
fd = os.open('/dev/null', os.O_RDONLY)
worker._close_range_all = lambda: (_ for _ in ()).throw(OSError('unsupported'))
worker.close_inherited_descriptors()
try:
    os.fstat(fd)
except OSError as exc:
    assert exc.errno == 9
else:
    raise AssertionError('fallback left descriptor open')
print('closed')
"""
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'closed'


def test_worker_fails_closed_when_descriptor_inventory_unavailable(monkeypatch, capsys):
    import pytest
    from protected_verifier import worker
    monkeypatch.setattr(worker, '_close_range_all', lambda: (_ for _ in ()).throw(OSError('no close_range')))
    monkeypatch.setattr(worker.os, 'listdir', lambda path: (_ for _ in ()).throw(OSError('no proc')))
    monkeypatch.setattr(worker, 'read_request', lambda stream: pytest.fail('untrusted input read'))
    with pytest.raises(SystemExit) as exc:
        worker.main()
    assert exc.value.code == 1
    assert 'Cannot inspect inherited descriptors' in capsys.readouterr().out


def test_worker_rejects_unsafe_versions_and_parent_mismatch():
    import io, pytest
    from protected_verifier.worker import read_request
    base = {'config': {}, 'modification': {'target':'learning_rate','operation':'multiply','parameters':{'value':1.1},'parent_version': 'v0'},
            'steps': 5, 'replicates': 1, 'parent': 'v0', 'candidate': 'v1'}
    assert read_request(io.BytesIO(json.dumps(base).encode())) == base
    for bad in ('../escape', 'folder/name', 'x\\y', 'bad\nname', '', 'a'*129):
        for field in ('parent', 'candidate'):
            payload = dict(base, **{field: bad})
            with pytest.raises(ValueError):
                read_request(io.BytesIO(json.dumps(payload).encode()))
    for mod in (None, [], {'parent_version': 'another'}):
        with pytest.raises(ValueError):
            read_request(io.BytesIO(json.dumps(dict(base, modification=mod)).encode()))


def test_verifier_tempdir_preflight_detects_seccomp_conflict():
    """Reproduce the v12 baseline failure without weakening the security filter."""
    probe = """
import json
from protected_verifier.network_limits import enforce_network_limits
from protected_verifier.worker import verify_temporary_directory_support
enforce_network_limits()
try:
    verify_temporary_directory_support()
except RuntimeError as exc:
    assert 'temporary-directory capability unavailable' in str(exc)
    print(json.dumps({'blocked': True}))
else:
    raise AssertionError('unexpected temp directory access')
"""
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'blocked': True}

def test_worker_tempdir_preflight_fails_before_request(monkeypatch, capsys):
    import pytest
    from protected_verifier import worker
    monkeypatch.setattr(worker, 'close_inherited_descriptors', lambda: None)
    monkeypatch.setattr(worker, 'enforce_worker_limits', lambda: None)
    monkeypatch.setattr(worker, 'enforce_network_limits', lambda: None)
    monkeypatch.setattr(worker, 'verify_temporary_directory_support', lambda: (_ for _ in ()).throw(RuntimeError('tempdir blocked')))
    monkeypatch.setattr(worker, 'read_request', lambda stream: {'steps': 5})
    with pytest.raises(SystemExit) as exc:
        worker.main()
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out) == {'ok': False, 'error': 'RuntimeError: tempdir blocked'}


def test_worker_rejects_deeply_nested_request_before_evaluation():
    import io, pytest
    from protected_verifier.worker import read_request, MAX_REQUEST_DEPTH
    payload={'config':{},'modification':{'target':'learning_rate','operation':'multiply','parameters':{'value':1.1},'parent_version':'v0'},'steps':5,'replicates':1,'parent':'v0','candidate':'v1'}
    nested='leaf'
    for _ in range(MAX_REQUEST_DEPTH + 2):
        nested=[nested]
    payload['config']={'nested':nested}
    with pytest.raises(ValueError, match='nesting exceeds limit'):
        read_request(io.BytesIO(json.dumps(payload).encode()))
    # The depth guard must not reject ordinary valid requests.
    payload['config']={}
    assert read_request(io.BytesIO(json.dumps(payload).encode())) == payload

def test_worker_rejects_deeply_nested_request_at_subprocess_boundary():
    from protected_verifier.worker import MAX_REQUEST_DEPTH
    payload={'config':{},'modification':{'target':'learning_rate','operation':'multiply','parameters':{'value':1.1},'parent_version':'v0'},'steps':5,'replicates':1,'parent':'v0','candidate':'v1'}
    nested=0
    for _ in range(MAX_REQUEST_DEPTH + 2):
        nested={'child':nested}
    payload['config']=nested
    result=subprocess.run([sys.executable,'-m','protected_verifier.worker'],
                          input=json.dumps(payload),capture_output=True,text=True,timeout=25)
    assert result.returncode != 0
    assert 'nesting exceeds limit' in json.loads(result.stdout)['error']


def test_seccomp_denies_cross_process_memory_and_handle_access():
    """Kernel denial of direct process-memory/descriptor access; not a full sandbox."""
    probe = r'''
import ctypes, errno, os
from protected_verifier.network_limits import enforce_network_limits
enforce_network_limits()
libc = ctypes.CDLL(None, use_errno=True)
for name, args in (
    ('ptrace', (0, os.getpid(), 0, 0)),
    ('process_vm_readv', (os.getpid(), 0, 0, 0, 0, 0)),
    ('process_vm_writev', (os.getpid(), 0, 0, 0, 0, 0)),
    ('pidfd_getfd', (-1, 0, 0)),
    ('open_by_handle_at', (-1, 0, 0)),
):
    fn = getattr(libc, name)
    fn.restype = ctypes.c_long
    ctypes.set_errno(0)
    assert fn(*args) == -1 and ctypes.get_errno() == errno.EPERM, name
print('memory and handle syscalls denied')
'''
    result = subprocess.run([sys.executable, '-c', probe], text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'memory and handle syscalls denied'


def test_worker_rejects_malformed_typed_envelopes_before_evaluation():
    import io, pytest
    from protected_verifier.worker import read_request
    valid = {"config": {}, "modification": {"target":"learning_rate", "operation":"multiply", "parameters":{"value":1.1}, "parent_version":"v0"}, "steps":5, "replicates":1, "parent":"v0", "candidate":"v1"}
    def parse(payload):
        return read_request(io.BytesIO(json.dumps(payload).encode()))
    assert parse(valid) == valid
    bad = [
        dict(valid, config=[]),
        dict(valid, modification={**valid["modification"], "unexpected": 1}),
        dict(valid, modification={k:v for k,v in valid["modification"].items() if k != "target"}),
        dict(valid, modification={**valid["modification"], "parameters": []}),
        dict(valid, modification={**valid["modification"], "target": 7}),
        dict(valid, modification={**valid["modification"], "operation": None}),
        dict(valid, modification={**valid["modification"], "parent_version": "../secret"}),
    ]
    for payload in bad:
        with pytest.raises(ValueError):
            parse(payload)

def test_worker_rejects_extra_modification_fields_in_subprocess():
    from aura.config import Config
    request={"config":Config().model_dump(), "modification":{"target":"learning_rate", "operation":"multiply", "parameters":{"value":1.1}, "parent_version":"v0", "unexpected":"payload"}, "steps":5, "replicates":1, "parent":"v0", "candidate":"v1"}
    result=subprocess.run([sys.executable,"-m","protected_verifier.worker"],input=json.dumps(request),capture_output=True,text=True,timeout=25)
    assert result.returncode != 0
    assert json.loads(result.stdout) == {"ok":False,"error":"ValueError: Invalid modification schema"}


def test_worker_rejects_overflowing_json_float_at_parse_boundary():
    """JSON exponent overflow must not become an infinite model parameter."""
    import io, pytest
    from protected_verifier.worker import read_request
    valid={"config":{},"modification":{"target":"learning_rate","operation":"multiply",
           "parameters":{"value":1.1},"parent_version":"v0"},
           "steps":5,"replicates":1,"parent":"v0","candidate":"v1"}
    raw=json.dumps(valid).replace('1.1','1e999').encode()
    with pytest.raises(ValueError, match='Non-finite JSON number'):
        read_request(io.BytesIO(raw))
    assert read_request(io.BytesIO(json.dumps(valid).encode())) == valid

def test_worker_rejects_overflowing_json_float_in_subprocess():
    import json, subprocess, sys
    payload={"config":{},"modification":{"target":"learning_rate","operation":"multiply",
             "parameters":{"value":1.1},"parent_version":"v0"},
             "steps":5,"replicates":1,"parent":"v0","candidate":"v1"}
    raw=json.dumps(payload).replace('1.1','1e999')
    result=subprocess.run([sys.executable,'-m','protected_verifier.worker'],
                          input=raw,text=True,capture_output=True,timeout=25)
    assert result.returncode != 0
    assert json.loads(result.stdout) == {'ok':False,'error':'ValueError: Non-finite JSON number'}


def test_worker_rejects_wide_json_request_before_evaluation():
    """A shallow 64-KiB request must not create an unbounded object traversal."""
    import io, pytest
    from protected_verifier.worker import read_request, MAX_REQUEST_NODES
    payload={"config":{"wide":[0] * (MAX_REQUEST_NODES + 1)},
             "modification":{"target":"learning_rate","operation":"multiply",
                             "parameters":{"value":1.1},"parent_version":"v0"},
             "steps":5,"replicates":1,"parent":"v0","candidate":"v1"}
    raw=json.dumps(payload).encode()
    assert len(raw) < 65536
    with pytest.raises(ValueError, match="node count exceeds limit"):
        read_request(io.BytesIO(raw))
    payload["config"]={"wide":[0,1,2]}
    assert read_request(io.BytesIO(json.dumps(payload).encode())) == payload

def test_worker_rejects_wide_json_request_in_subprocess():
    from protected_verifier.worker import MAX_REQUEST_NODES
    payload={"config":{"wide":[0] * (MAX_REQUEST_NODES + 1)},
             "modification":{"target":"learning_rate","operation":"multiply",
                             "parameters":{"value":1.1},"parent_version":"v0"},
             "steps":5,"replicates":1,"parent":"v0","candidate":"v1"}
    result=subprocess.run([sys.executable,'-m','protected_verifier.worker'],
                          input=json.dumps(payload),text=True,capture_output=True,timeout=25)
    assert result.returncode != 0
    assert json.loads(result.stdout) == {'ok':False,'error':'ValueError: Evaluator request node count exceeds limit'}


def test_worker_rejects_self_parented_candidate_before_evaluation():
    import io, pytest
    from protected_verifier.worker import read_request
    payload={"config": {}, "modification": {"target":"learning_rate", "operation":"multiply", "parameters":{"value":1.1}, "parent_version":"v0"}, "steps":5, "replicates":1, "parent":"v0", "candidate":"v0"}
    with pytest.raises(ValueError, match="Candidate version must differ from parent"):
        read_request(io.BytesIO(json.dumps(payload).encode()))

def test_worker_subprocess_rejects_self_parented_candidate():
    from aura.config import Config
    payload={"config":Config().model_dump(), "modification":{"target":"learning_rate","operation":"multiply","parameters":{"value":1.1},"parent_version":"v0"},"steps":5,"replicates":1,"parent":"v0","candidate":"v0"}
    result=subprocess.run([sys.executable,"-m","protected_verifier.worker"], input=json.dumps(payload),capture_output=True,text=True,timeout=25)
    assert result.returncode != 0
    assert json.loads(result.stdout)=={"ok":False,"error":"ValueError: Candidate version must differ from parent"}


def test_worker_requires_strict_utf8_wire_encoding():
    import io, pytest
    from protected_verifier.worker import read_request
    payload={"config":{},"modification":{"target":"learning_rate","operation":"multiply",
             "parameters":{"value":1.1},"parent_version":"v0"},
             "steps":5,"replicates":1,"parent":"v0","candidate":"v1"}
    encoded=json.dumps(payload).encode("utf-8")
    assert read_request(io.BytesIO(encoded)) == payload
    for invalid in (encoded.decode().encode("utf-16"), encoded.decode().encode("utf-32"),
                    b"\xef\xbb\xbf" + encoded, encoded + b"\xff"):
        with pytest.raises(ValueError):
            read_request(io.BytesIO(invalid))

def test_worker_rejects_alternate_encoding_in_real_subprocess():
    from aura.config import Config
    payload={"config":Config().model_dump(),
             "modification":{"target":"learning_rate","operation":"multiply",
             "parameters":{"value":1.1},"parent_version":"v0"},
             "steps":5,"replicates":1,"parent":"v0","candidate":"v1"}
    encoded=json.dumps(payload).encode("utf-16")
    result=subprocess.run([sys.executable,"-m","protected_verifier.worker"],
                          input=encoded,capture_output=True,timeout=25)
    assert result.returncode != 0
    assert json.loads(result.stdout)["ok"] is False
    assert "UTF-8" in json.loads(result.stdout)["error"]


def test_worker_rejects_unpaired_surrogates_in_keys_and_values():
    import io, pytest
    from protected_verifier.worker import read_request
    payload={"config":{},"modification":{"target":"learning_rate","operation":"multiply",
             "parameters":{"value":1.1},"parent_version":"v0"},
             "steps":5,"replicates":1,"parent":"v0","candidate":"v1"}
    for config in ({"bad\ud800":1}, {"nested":["bad\udfff"]}):
        payload["config"]=config
        with pytest.raises(ValueError, match="unpaired Unicode surrogate"):
            read_request(io.BytesIO(json.dumps(payload).encode()))

def test_worker_subprocess_rejects_unpaired_surrogate():
    payload={"config":{"bad":"\ud800"},"modification":{"target":"learning_rate",
             "operation":"multiply","parameters":{"value":1.1},"parent_version":"v0"},
             "steps":5,"replicates":1,"parent":"v0","candidate":"v1"}
    result=subprocess.run([sys.executable,"-m","protected_verifier.worker"],
                          input=json.dumps(payload),capture_output=True,text=True,timeout=25)
    assert result.returncode != 0
    assert json.loads(result.stdout)=={"ok":False,"error":"ValueError: Evaluator request contains unpaired Unicode surrogate"}


def test_seccomp_denies_legacy_creat_syscall_without_touching_sentinel(tmp_path):
    """Real kernel test of legacy creat(2), a write path not covered by unlink rules."""
    import os
    target = tmp_path / 'sentinel'
    target.write_bytes(b'protected')
    probe = r'''
import ctypes, errno, os, sys
from protected_verifier.network_limits import enforce_network_limits
libc = ctypes.CDLL(None, use_errno=True)
libc.creat.argtypes = [ctypes.c_char_p, ctypes.c_uint]
libc.creat.restype = ctypes.c_int
enforce_network_limits()
fd = libc.creat(os.fsencode(sys.argv[1]), 0o600)
assert fd == -1 and ctypes.get_errno() == errno.EPERM, (fd, ctypes.get_errno())
print('creat denied')
'''
    result = subprocess.run([sys.executable, '-c', probe, str(target)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'creat denied'
    assert target.read_bytes() == b'protected'


def test_seccomp_denies_write_capable_open_and_openat_preserves_read(tmp_path):
    """Exercise real kernel enforcement rather than inspecting the filter source."""
    import os
    path = tmp_path / 'sentinel'
    path.write_text('unchanged')
    probe = '''
import ctypes, errno, json, os
from protected_verifier.network_limits import enforce_network_limits
p=os.environ['SENTINEL']
enforce_network_limits()
results={}
for name, flags in [('write', os.O_WRONLY), ('readwrite', os.O_RDWR),
                    ('create', os.O_RDONLY | os.O_CREAT),
                    ('truncate', os.O_RDONLY | os.O_TRUNC),
                    ('append', os.O_RDONLY | os.O_APPEND)]:
    try:
        fd=os.open(p, flags); os.close(fd); results[name]='ALLOWED'
    except OSError as exc:
        results[name]=exc.errno
with open(p, 'r') as stream:
    results['read']=stream.read()
print(json.dumps(results))
'''
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True,
                            env={**os.environ, 'SENTINEL': str(path)}, timeout=15)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert all(data[key] == 1 for key in ('write','readwrite','create','truncate','append'))
    assert data['read'] == 'unchanged'
    assert path.read_text() == 'unchanged'



def test_seccomp_denies_memfd_creation_in_real_subprocess():
    """A candidate cannot create an anonymous writable in-memory file."""
    probe = """
import ctypes, errno, json, os
from protected_verifier.network_limits import enforce_network_limits
enforce_network_limits()
try:
    fd=os.memfd_create('candidate-buffer', 0)
except OSError as exc:
    print(json.dumps({'errno': exc.errno}))
else:
    os.close(fd)
    print(json.dumps({'errno': 0}))
"""
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'errno': 1}


def test_seccomp_denies_openat2_in_real_subprocess(tmp_path):
    """openat2 uses pointer-based flags, so deny it entirely under this policy."""
    sentinel = tmp_path / 'sentinel'
    sentinel.write_bytes(b'protected')
    probe = r'''
import ctypes as C, errno, os, sys
from protected_verifier.network_limits import enforce_network_limits
lib = C.CDLL('libseccomp.so.2')
lib.seccomp_syscall_resolve_name.argtypes = [C.c_char_p]
lib.seccomp_syscall_resolve_name.restype = C.c_int
number = lib.seccomp_syscall_resolve_name(b'openat2')
assert number >= 0
libc = C.CDLL(None, use_errno=True)
libc.syscall.restype = C.c_long
how = (C.c_uint64 * 3)(os.O_WRONLY, 0, 0)
enforce_network_limits()
fd = libc.syscall(C.c_long(number), C.c_int(-100), C.c_char_p(os.fsencode(sys.argv[1])), C.byref(how), C.c_size_t(C.sizeof(how)))
assert fd == -1 and C.get_errno() == errno.EPERM, (fd, C.get_errno())
assert open(sys.argv[1], 'rb').read() == b'protected'
'''
    result = subprocess.run([sys.executable, '-c', probe, str(sentinel)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == b'protected'


def test_seccomp_denies_xattr_mutation_preserves_sentinel(tmp_path):
    """Filesystem metadata must not be mutable under the read-only policy."""
    import os, errno, pytest
    sentinel = tmp_path / 'sentinel'
    sentinel.write_bytes(b'protected')
    try:
        os.setxattr(sentinel, 'user.aura', b'original')
    except OSError as exc:
        if exc.errno in (errno.ENOTSUP, errno.EOPNOTSUPP):
            pytest.skip('filesystem lacks user extended attributes')
        raise
    probe = r'''
import errno, os, sys
from protected_verifier.network_limits import enforce_network_limits
p = sys.argv[1]
enforce_network_limits()
for action in (lambda: os.setxattr(p, 'user.aura', b'changed'),
               lambda: os.removexattr(p, 'user.aura')):
    try:
        action()
    except OSError as exc:
        assert exc.errno == errno.EPERM, exc
    else:
        raise AssertionError('metadata mutation allowed')
assert os.getxattr(p, 'user.aura') == b'original'
'''
    result = subprocess.run([sys.executable, '-c', probe, str(sentinel)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert os.getxattr(sentinel, 'user.aura') == b'original'
    assert sentinel.read_bytes() == b'protected'


def test_runner_rejects_invalid_success_identity_and_acceptance(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    import pytest
    from aura.self_improvement import runner
    report = asdict(decide('v1', 'v0', [.01]*5, [.1]*5, [.09]*5,
                           {'trained_model_steps': 50, 'replicates': 5}))
    for changes in ({'candidate_version': 'other'}, {'baseline_version': 'other'},
                    {'accepted': 'false'}, {'accepted': True, 'reason': 'NO_VALIDATED_GAIN_OR_REGRESSION'}):
        payload = {'ok': True, 'report': dict(report, **changes)}
        monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps(payload), stderr='', returncode=0))
        with pytest.raises(RuntimeError, match='Invalid verifier response'):
            runner.evaluate_candidate(Config(), Modification('learning_rate','multiply',{'value':1.1},'v0'),steps=5,replicates=5)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout=json.dumps({'ok': True, 'report': report}), stderr='', returncode=0))
    assert runner.evaluate_candidate(Config(), Modification('learning_rate','multiply',{'value':1.1},'v0'),steps=5,replicates=5).accepted is True


def test_runner_rejects_inconsistent_report_budget_and_evidence(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    import pytest
    from aura.self_improvement import runner
    report = asdict(decide('v1', 'v0', [.01]*5, [.1]*5, [.09]*5,
                           {'trained_model_steps': 50, 'replicates': 5}))
    mod = Modification('learning_rate','multiply',{'value':1.1},'v0')
    for changes in (
        {'total_cost': {'trained_model_steps': 49, 'replicates': 5}},
        {'total_cost': {'trained_model_steps': 50, 'replicates': True}},
        {'metrics': dict(report['metrics'], mean_prediction_gain=float('nan'))},
        {'metrics': dict(report['metrics'], baseline_mean_loss=-.1)},
        {'metrics': dict(report['metrics'], mean_prediction_gain=.5)},
        {'confidence_intervals': {'prediction_gain_ci95': [.02, .01]}},
        {'confidence_intervals': {'prediction_gain_ci95': [-.01, .01]}},
        {'regressions': {'max_seed_regression': .02}},
    ):
        payload = {'ok': True, 'report': dict(report, **changes)}
        monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps(payload), stderr='', returncode=0))
        with pytest.raises(RuntimeError, match='Invalid verifier response'):
            runner.evaluate_candidate(Config(), mod, steps=5, replicates=5)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout=json.dumps({'ok': True, 'report': report}), stderr='', returncode=0))
    assert runner.evaluate_candidate(Config(), mod, steps=5, replicates=5).accepted


def test_runner_rejects_ambiguous_response_framing(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    import pytest
    from aura.self_improvement import runner
    report = asdict(decide('v1', 'v0', [.01]*5, [.1]*5, [.09]*5,
                           {'trained_model_steps': 50, 'replicates': 5}))
    valid = json.dumps({'ok': True, 'report': report})
    mod = Modification('learning_rate','multiply',{'value':1.1},'v0')
    def invoke(stdout):
        monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
            stdout=stdout, stderr='', returncode=0))
        return runner.evaluate_candidate(Config(), mod, steps=5, replicates=5)
    for malformed in (valid.replace('"ok": true', '"ok": false, "ok": true'),
                      valid.replace('"accepted": true', '"accepted": false, "accepted": true'),
                      '{"ok":false}\n'+valid, valid+'\n'+valid,
                      valid[:-1]+', "extra": 1}', '[]', '{',
                      valid.replace('"accepted": true', '"accepted": NaN')):
        with pytest.raises(RuntimeError, match='Invalid verifier response'):
            invoke(malformed)
    assert invoke(valid+'\n').accepted is True


def test_runner_rejects_impossible_bootstrap_summary(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    import pytest
    from aura.self_improvement import runner
    report = asdict(decide('v1', 'v0', [.01]*5, [.1]*5, [.09]*5,
                           {'trained_model_steps': 50, 'replicates': 5}))
    mod = Modification('learning_rate','multiply',{'value':1.1},'v0')
    def invoke(changes):
        monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps({'ok': True, 'report': dict(report, **changes)}), stderr='', returncode=0))
        return runner.evaluate_candidate(Config(), mod, steps=5, replicates=5)
    # Every resampled mean is bounded by the minimum and maximum paired gain.
    # With five nonnegative baseline losses totaling .5, no gain can exceed .5.
    for changes in ({'confidence_intervals': {'prediction_gain_ci95': [.6, .7]}},
                    {'regressions': {'max_seed_regression': 10**400}},
                    {'metrics': dict(report['metrics'], baseline_mean_loss=10**400)}):
        with pytest.raises(RuntimeError, match='Invalid verifier response'):
            invoke(changes)
    assert invoke({}).accepted


def test_runner_preserves_small_sample_bootstrap_sentinel(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    from aura.self_improvement import runner
    report = asdict(decide('v1', 'v0', [.01]*2, [.1]*2, [.09]*2,
                           {'trained_model_steps': 20, 'replicates': 2}))
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout=json.dumps({'ok': True, 'report': report}), stderr='', returncode=0))
    assert not runner.evaluate_candidate(Config(), Modification('learning_rate','multiply',{'value':1.1},'v0'),steps=5,replicates=2).accepted
