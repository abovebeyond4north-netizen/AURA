"""Lossless, compressed, append-only observations and outcomes for reproducibility.

Every stored image is verified against a SHA-256 digest when read back. The DB
is local and the hash chain is tamper evident, not externally signed.
"""
import hashlib
import zlib
import base64
import numpy as np
from aura.types import Experience, Source

def _encode(image):
    if image.dtype != np.uint8: raise ValueError('Images must be uint8')
    return {'shape':list(image.shape),'dtype':'uint8',
            'data_zlib_base64':base64.b64encode(zlib.compress(image.tobytes())).decode('ascii'),
            'sha256':hashlib.sha256(image.tobytes()).hexdigest()}

def _decode(blob):
    if blob.get('dtype')!='uint8' or blob.get('shape')!=[3,7,7]: raise ValueError('Invalid observation schema')
    raw=zlib.decompress(base64.b64decode(blob['data_zlib_base64']))
    if hashlib.sha256(raw).hexdigest()!=blob['sha256']: raise ValueError('Observation digest mismatch')
    return np.frombuffer(raw,dtype=np.uint8).reshape(blob['shape']).copy()

class EpisodicMemory:
    def __init__(self, ledger): self.ledger=ledger
    def add(self, exp: Experience):
        return self.ledger.append(exp.evidence_id,'experience',Source.OBSERVED,{
            'episode_id':exp.episode_id,'seed':exp.seed,'action':exp.action,
            'reward':exp.reward,'terminated':exp.terminated,'truncated':exp.truncated,
            'context':exp.context,
            'observation':_encode(exp.observation),
            'next_observation':_encode(exp.next_observation)})
    def reconstruct(self,record):
        if record['kind']!='experience' or record['source']!=Source.OBSERVED.value:
            raise ValueError('Only observed experiences can be reconstructed')
        p=record['payload']
        return Experience(_decode(p['observation']),p['action'],_decode(p['next_observation']),
                          p['reward'],p['terminated'],p['truncated'],p['episode_id'],
                          record['record_id'],Source.OBSERVED,p['seed'],p['context'])
