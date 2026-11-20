"""Kernel-enforced per-process limits, not a filesystem/network sandbox.

No candidate-controlled values are accepted. Existing stricter limits are kept.
"""
CPU_SECONDS = 600
FILE_BYTES = 16 * 1024 * 1024
OPEN_FILES = 256


def enforce_worker_limits():
    import resource  # unsupported platforms fail closed

    for kind, ceiling in (
        (resource.RLIMIT_CPU, CPU_SECONDS),
        (resource.RLIMIT_FSIZE, FILE_BYTES),
        (resource.RLIMIT_NOFILE, OPEN_FILES),
        (resource.RLIMIT_CORE, 0),
    ):
        soft, hard = resource.getrlimit(kind)
        effective = min([ceiling] + [v for v in (soft, hard) if v != resource.RLIM_INFINITY])
        resource.setrlimit(kind, (effective, effective))
