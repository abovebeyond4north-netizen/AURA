"""Linux seccomp network and filesystem mutation syscall denial.

Not a complete hostile-code sandbox: other filesystem channels, existing
descriptors and memory mappings are not comprehensively blocked.
"""
import ctypes as C
import errno
import os
import sys


def enforce_network_limits():
    """Install an inherited, irreversible filter before evaluator imports.

    Deny socket creation, selected filesystem mutations and io_uring setup.
    No candidate-supplied policy or permissive fallback is supported.
    """
    if sys.platform != 'linux':
        raise RuntimeError('Network containment requires Linux and libseccomp')
    lib = C.CDLL('libseccomp.so.2')
    lib.seccomp_init.argtypes = [C.c_uint32]
    lib.seccomp_init.restype = C.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [C.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = C.c_int
    lib.seccomp_rule_add.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_uint]
    lib.seccomp_rule_add.restype = C.c_int
    lib.seccomp_load.argtypes = [C.c_void_p]
    lib.seccomp_load.restype = C.c_int
    lib.seccomp_release.argtypes = [C.c_void_p]
    lib.seccomp_release.restype = None
    class ArgCompare(C.Structure):
        _fields_ = [('arg', C.c_uint), ('op', C.c_uint32),
                    ('datum_a', C.c_uint64), ('datum_b', C.c_uint64)]
    lib.seccomp_rule_add_array.argtypes = [C.c_void_p, C.c_uint32, C.c_int,
                                           C.c_uint, C.POINTER(ArgCompare)]
    lib.seccomp_rule_add_array.restype = C.c_int
    ctx = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError('Unable to initialize seccomp')
    try:
        deny = 0x00050000 | errno.EPERM  # SCMP_ACT_ERRNO
        for name in ('socket', 'socketpair', 'connect', 'bind', 'listen',
                     'accept', 'accept4', 'sendto', 'sendmsg', 'sendmmsg',
                     'recvfrom', 'recvmsg', 'recvmmsg', 'io_uring_setup',
                     'io_uring_enter', 'io_uring_register', 'memfd_create', 'openat2',
                     'unlink', 'unlinkat', 'rename', 'renameat', 'renameat2',
                     'mkdir', 'mkdirat', 'rmdir', 'symlink', 'symlinkat',
                     'link', 'linkat', 'truncate', 'ftruncate', 'creat',
                     'ptrace', 'process_vm_readv', 'process_vm_writev',
                     'pidfd_getfd', 'open_by_handle_at',
                     'chmod', 'fchmod', 'fchmodat', 'chown', 'fchown',
                     'fchownat', 'lchown', 'setxattr', 'lsetxattr', 'fsetxattr',
                     'removexattr', 'lremovexattr', 'fremovexattr'):
            number = lib.seccomp_syscall_resolve_name(name.encode('ascii'))
            if number < 0 or lib.seccomp_rule_add(ctx, deny, number, 0) != 0:
                raise RuntimeError(f'Unable to restrict syscall {name}')
        # Reject write-capable opens, including O_RDONLY|O_CREAT and O_TMPFILE.
        # SCMP_CMP_MASKED_EQ (7) matches flags & mask == value.
        write_bits = (os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC,
                      os.O_APPEND, getattr(os, 'O_TMPFILE', 0))
        for name, flag_arg in (('open', 1), ('openat', 2)):
            number = lib.seccomp_syscall_resolve_name(name.encode('ascii'))
            if number < 0:
                raise RuntimeError(f'Unable to restrict syscall {name}')
            for bit in write_bits:
                if not bit:
                    continue
                cmp = ArgCompare(flag_arg, 7, bit, bit)
                if lib.seccomp_rule_add_array(ctx, deny, number, 1, C.byref(cmp)) != 0:
                    raise RuntimeError(f'Unable to restrict {name} write flags')
        if lib.seccomp_load(ctx) != 0:
            raise RuntimeError('Unable to load seccomp network filter')
    finally:
        lib.seccomp_release(ctx)
