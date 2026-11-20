"""Export and verify a ledger prefix against an independently retained receipt."""
import argparse
from contextlib import closing
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3


def canonical(value):
    return (json.dumps(value, ensure_ascii=True, separators=(',', ':'), sort_keys=True,
                       allow_nan=False) + '\n').encode('ascii')


def checkpoint(db, length=None):
    """Commit to all original columns, including sequence and timestamp."""
    count = 0
    tip = hashlib.sha256(b'aura-evidence-prefix-v1\n')
    with closing(sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True)) as ledger:
        rows = ledger.execute(
            'SELECT sequence, record_id, kind, source, payload, previous_hash, record_hash, recorded_at '
            'FROM ledger ORDER BY sequence')
        for row in rows:
            if length is not None and count == length:
                break
            tip.update(canonical(row))
            count += 1
    return {'format': 'aura-evidence-prefix-v1', 'rows': count, 'sha256': tip.hexdigest()}


def verify(db, receipt_bytes, pinned_sha256):
    if not hmac.compare_digest(hashlib.sha256(receipt_bytes).hexdigest(), pinned_sha256):
        return False
    receipt = json.loads(receipt_bytes)
    if not isinstance(receipt, dict) or type(receipt.get('rows')) is not int or receipt['rows'] <= 0:
        return False
    return checkpoint(db, receipt['rows']) == receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    check = commands.add_parser('verify')
    for command in (export, check):
        command.add_argument('ledger', type=Path)
        command.add_argument('receipt', type=Path)
    check.add_argument('--pin', required=True)
    args = parser.parse_args()
    if args.command == 'export':
        value = checkpoint(args.ledger)
        if not value['rows']:
            parser.error('cannot anchor an empty ledger')
        raw = canonical(value)
        with args.receipt.open('xb') as f:
            f.write(raw)
        print(hashlib.sha256(raw).hexdigest())
    elif verify(args.ledger, args.receipt.read_bytes(), args.pin):
        print('Evidence prefix verified')
    else:
        parser.exit(1, 'Evidence prefix verification failed\n')


if __name__ == '__main__':
    main()
