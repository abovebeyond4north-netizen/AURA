from pathlib import Path
import pytest
from tools.verify_external_eval import verify

def test_unsigned_external_claim_rejected(tmp_path):
    a = tmp_path / 'claim.json'
    a.write_text('{"independent":true}')
    with pytest.raises(ValueError, match='Missing required'):
        verify(a, a, tmp_path/'missing.sig', tmp_path/'missing.asc', 'A'*40)

def test_invalid_fingerprint_rejected_before_authenticating(tmp_path):
    f = tmp_path/'claim.json'
    f.write_text('{}')
    with pytest.raises(ValueError, match='fingerprint'):
        verify(f, f, f, f, 'self-asserted')
