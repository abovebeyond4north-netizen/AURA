"""Append-only, hash-chained SQLite evidence; tamper *evident*, not tamper-proof.
Any learner code has no mutating SQL connection to this process-owned store.
"""
from __future__ import annotations
import sqlite3, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from aura.types import Source

SCHEMA = '''
CREATE TABLE IF NOT EXISTS ledger(
 sequence INTEGER PRIMARY KEY AUTOINCREMENT,
 record_id TEXT NOT NULL UNIQUE,
 kind TEXT NOT NULL,
 source TEXT NOT NULL,
 payload TEXT NOT NULL,
 previous_hash TEXT NOT NULL,
 record_hash TEXT NOT NULL,
 recorded_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger BEGIN SELECT RAISE(ABORT,'append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger BEGIN SELECT RAISE(ABORT,'append-only'); END;
'''

class EvidenceLedger:
    def __init__(self,path=':memory:'):
        self.path=str(path)
        self.con=sqlite3.connect(self.path)
        self.con.executescript(SCHEMA)
    def append(self,record_id,kind,source: Source,payload: dict):
        if not isinstance(source,Source): raise ValueError('Source enum required')
        if kind in ('experience', 'door_observation') and source != Source.OBSERVED: raise ValueError('Imagined data cannot be evidence of actual experience')
        encoded=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
        with self.con:
            previous=self.con.execute('SELECT record_hash FROM ledger ORDER BY sequence DESC LIMIT 1').fetchone()
            prev=previous[0] if previous else 'GENESIS'
            digest=hashlib.sha256(f'{record_id}|{kind}|{source.value}|{encoded}|{prev}'.encode()).hexdigest()
            self.con.execute('INSERT INTO ledger(record_id,kind,source,payload,previous_hash,record_hash,recorded_at) VALUES(?,?,?,?,?,?,?)',
                             (record_id,kind,source.value,encoded,prev,digest,datetime.now(timezone.utc).isoformat()))
        return digest
    def records(self,kind=None):
        query='SELECT record_id,kind,source,payload FROM ledger'
        rows=self.con.execute(query+(' WHERE kind=?' if kind else '')+' ORDER BY sequence', (kind,) if kind else ())
        return [{'record_id':r[0],'kind':r[1],'source':r[2],'payload':json.loads(r[3])} for r in rows]
    def verify(self):
        prev='GENESIS'
        for rid,kind,src,payload,ph,h in self.con.execute('SELECT record_id,kind,source,payload,previous_hash,record_hash FROM ledger ORDER BY sequence'):
            expected=hashlib.sha256(f'{rid}|{kind}|{src}|{payload}|{prev}'.encode()).hexdigest()
            if ph!=prev or h!=expected: return False
            prev=h
        return True
    def close(self): self.con.close()
