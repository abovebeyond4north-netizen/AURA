"""Controlled namespace experiment; never used to authorize verifier execution."""
import ctypes
import errno
import resource
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import sysconfig


def _inside(root, sentinel, parent_pid_namespace, inherited_fd):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from protected_verifier.worker import close_inherited_descriptors
    inherited_fd = int(inherited_fd)
    if os.pread(inherited_fd, 9, 0) != b'protected':
        raise RuntimeError('inherited sentinel descriptor missing before isolation')
    subprocess.run(['mount', '--make-rprivate', '/'], check=True)
    subprocess.run(['mount', '-t', 'tmpfs', '-o', 'size=4m,nosuid,nodev,noexec', 'tmpfs', root], check=True)
    # Expose only the trusted standard library, not site-packages or host root.
    stdlib = sysconfig.get_path('stdlib')
    runtime = Path(root) / 'runtime'
    runtime.mkdir()
    subprocess.run(['mount', '--bind', stdlib, str(runtime)], check=True)
    subprocess.run(['mount', '-o', 'remount,bind,ro,nosuid,nodev,noexec', str(runtime)], check=True)
    # Narrow executable mounts: only the trusted SQLite extension and its library.
    import importlib.util
    native = Path(importlib.util.find_spec('_sqlite3').origin)
    library = Path('/lib') / sysconfig.get_config_var('MULTIARCH') / 'libsqlite3.so.0'
    for source, relative in ((native, Path('native') / native.name),
                             (library, Path(str(library).lstrip('/')))):
        target = Path(root) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
        subprocess.run(['mount', '--bind', str(source), str(target)], check=True)
        subprocess.run(['mount', '-o', 'remount,bind,ro,nosuid,nodev,exec', str(target)], check=True)
    if os.getpid() != 1 or os.readlink('/proc/self/ns/pid') == parent_pid_namespace:
        raise RuntimeError('private PID namespace not established')
    libc = ctypes.CDLL(None, use_errno=True)
    class Header(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]
    os.chroot(root)
    os.chdir('/')
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.capset(ctypes.byref(Header(0x20080522, 0)), (Data * 2)()) != 0:
        raise OSError(ctypes.get_errno(), 'capability drop failed')
    if os.path.exists('/proc') or os.getppid() != 0:
        raise RuntimeError('host procfs or parent visible')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, 64), hard))
    close_inherited_descriptors()
    try:
        os.fstat(inherited_fd)
    except OSError as exc:
        if exc.errno != errno.EBADF:
            raise
    else:
        raise RuntimeError('inherited sentinel descriptor remains open')
    for fd in (0, 1, 2):
        os.fstat(fd)
    # Trusted probe only; no candidate execution. Force a new source import.
    import importlib
    if 'colorsys' in sys.modules:
        raise RuntimeError('runtime probe module already loaded')
    sys.path[:] = ['/runtime']
    sys.dont_write_bytecode = True
    importlib.invalidate_caches()
    import colorsys
    if colorsys.__file__ != '/runtime/colorsys.py' or colorsys.rgb_to_hsv(1, 0, 0) != (0.0, 1.0, 1):
        raise RuntimeError('fresh runtime import failed')
    try:
        fd = os.open('/runtime/colorsys.py', os.O_WRONLY)
    except OSError as exc:
        if exc.errno != errno.EROFS:
            raise
    else:
        os.close(fd)
        raise RuntimeError('runtime mount writable')
    if '_sqlite3' in sys.modules or 'sqlite3' in sys.modules:
        raise RuntimeError('native probe module already loaded')
    sys.path.insert(0, '/native')
    import sqlite3
    import _sqlite3
    if _sqlite3.__file__ != '/native/' + native.name:
        raise RuntimeError('native module origin mismatch')
    with sqlite3.connect(':memory:') as connection:
        if connection.execute('SELECT 6 * 7').fetchone() != (42,):
            raise RuntimeError('native computation failed')
    connection.close()
    for path in (_sqlite3.__file__, str(library)):
        try:
            fd = os.open(path, os.O_WRONLY)
        except OSError as exc:
            if exc.errno != errno.EROFS:
                raise
        else:
            os.close(fd)
            raise RuntimeError('native runtime writable')
    os.mkdir('/scratch')
    with open('/scratch/check', 'w') as f:
        f.write('private')
    with open('/scratch/check') as f:
        assert f.read() == 'private'
    try:
        with open(sentinel, 'rb'):
            pass
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError('host sentinel visible')
    try:
        os.chroot('/')
    except PermissionError:
        pass
    else:
        raise RuntimeError('chroot capability retained')
    print('{"scratch_write":true,"host_sentinel_hidden":true,"chroot_denied":true,"pid_namespace_private":true,"host_proc_hidden":true,"inherited_descriptors_closed":true,"fresh_stdlib_import":true,"runtime_write_denied":true,"fresh_native_import":true,"native_write_denied":true}')


def probe():
    with tempfile.TemporaryDirectory(prefix='aura-namespace-probe-') as directory:
        root = Path(directory) / 'root'
        root.mkdir()
        sentinel = Path(directory) / 'sentinel'
        sentinel.write_bytes(b'protected')
        with sentinel.open('rb') as handle:
            import fcntl
            high_fd = fcntl.fcntl(handle.fileno(), fcntl.F_DUPFD, 220)
            try:
                child = subprocess.run(['unshare', '-Urnmp', '--fork', '--kill-child', sys.executable, str(Path(__file__).resolve()), '--inside', str(root), str(sentinel), os.readlink('/proc/self/ns/pid'), str(high_fd)],
                                       pass_fds=(high_fd,), capture_output=True, text=True, timeout=20, check=True)
            finally:
                os.close(high_fd)
        result = json.loads(child.stdout)
        if sentinel.read_bytes() != b'protected' or list(root.iterdir()):
            raise RuntimeError('host state changed')
        return result


if __name__ == '__main__':
    if len(sys.argv) == 6 and sys.argv[1] == '--inside':
        _inside(*sys.argv[2:])
    else:
        print(json.dumps(probe(), sort_keys=True))
