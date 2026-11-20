"""Opt-in integration test: unsupported hosts must not report a pass."""
import os
import pytest
from scripts.probe_private_scratch import probe


@pytest.mark.skipif(os.environ.get('AURA_TEST_NAMESPACES') != '1', reason='requires explicitly enabled Linux namespace integration')
def test_private_scratch_namespace_probe():
    assert probe() == {'scratch_write': True, 'host_sentinel_hidden': True, 'chroot_denied': True, 'pid_namespace_private': True, 'host_proc_hidden': True, 'inherited_descriptors_closed': True, 'fresh_stdlib_import': True, 'runtime_write_denied': True, 'fresh_native_import': True, 'native_write_denied': True}


@pytest.mark.skipif(os.environ.get('AURA_TEST_NAMESPACES') != '1', reason='requires explicitly enabled Linux namespace integration')
def test_private_pid_namespace_probe():
    result = probe()
    assert result['pid_namespace_private'] is True
    assert result['host_proc_hidden'] is True


@pytest.mark.skipif(os.environ.get('AURA_TEST_NAMESPACES') != '1', reason='requires explicitly enabled Linux namespace integration')
def test_private_namespace_closes_inherited_descriptors():
    assert probe()['inherited_descriptors_closed'] is True


@pytest.mark.skipif(os.environ.get('AURA_TEST_NAMESPACES') != '1', reason='requires explicitly enabled Linux namespace integration')
def test_private_namespace_readonly_runtime_import():
    result = probe()
    assert result['fresh_stdlib_import'] is True
    assert result['runtime_write_denied'] is True


@pytest.mark.skipif(os.environ.get('AURA_TEST_NAMESPACES') != '1', reason='requires explicitly enabled Linux namespace integration')
def test_private_namespace_native_extension():
    result = probe()
    assert result['fresh_native_import'] is True
    assert result['native_write_denied'] is True
