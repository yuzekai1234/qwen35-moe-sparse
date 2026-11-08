"""Shared paths, atomic records, and cross-agent GPU leases (standard library only)."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import time

HERE = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = Path('/data/quark-7db0d9a22ffb')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash(root=HERE):
    h = hashlib.sha256()
    root = Path(root)
    for p in sorted(root.rglob('*')):
        if (p.is_file() and '__pycache__' not in p.parts
                and not any(part.startswith('._') for part in p.relative_to(root).parts)
                and p.suffix in ('.py', '.sh', '.json')):
            h.update(str(p.relative_to(root)).encode() + b'\0' + p.read_bytes())
    return h.hexdigest()


def state_root(workspace):
    return Path(workspace).resolve() / 'b-round-20261109'


def db(workspace):
    root = state_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / 'coordination.sqlite', timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''CREATE TABLE IF NOT EXISTS tasks
        (id TEXT PRIMARY KEY, owner TEXT, status TEXT, host TEXT, pid INTEGER,
         updated REAL, detail TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS events
        (time REAL, task TEXT, owner TEXT, status TEXT, detail TEXT)''')
    conn.commit()
    return conn


def claim(workspace, task, owner, resume=False):
    with contextlib.closing(db(workspace)) as c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute('SELECT * FROM tasks WHERE id=?', (task,)).fetchone()
        if row:
            # Never silently reclaim another agent's live or stale registration.
            if row['status'] in ('running', 'waiting', 'training', 'evaluating'):
                stale_owned = False
                if resume and row['owner'] == owner and row['host'] == socket.gethostname():
                    try:
                        os.kill(row['pid'], 0)
                    except ProcessLookupError:
                        stale_owned = True
                if not stale_owned:
                    raise RuntimeError('Task already claimed: ' + json.dumps(dict(row), ensure_ascii=False))
            if row['status'] == 'complete' or not resume:
                raise RuntimeError('Use a new run id, or --resume for a failed/interrupted task: ' + task)
        c.execute('INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?)',
                  (task, owner, 'running', socket.gethostname(), os.getpid(), time.time(), '{}'))
        c.execute('INSERT INTO events VALUES (?,?,?,?,?)', (time.time(), task, owner, 'claimed', '{}'))
        c.commit()


def record(workspace, task, owner, status, detail):
    encoded = json.dumps(detail, ensure_ascii=False)
    with contextlib.closing(db(workspace)) as c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute('SELECT owner,pid FROM tasks WHERE id=?', (task,)).fetchone()
        if not row or row['owner'] != owner or row['pid'] != os.getpid():
            raise RuntimeError('Task ownership changed; refusing to update it')
        c.execute('UPDATE tasks SET status=?,updated=?,detail=? WHERE id=?',
                  (status, time.time(), encoded, task))
        c.execute('INSERT INTO events VALUES (?,?,?,?,?)', (time.time(), task, owner, status, encoded))
        c.commit()


def inventory():
    import csv
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(i): dict(uuid=u.strip(), total_mib=int(t), used_mib=int(m), utilization=int(v))
            for i, u, t, m, v in csv.reader(text.splitlines())}


class Lease:
    """Kernel locks plus physical-process checks. No force/unlock/kill-other operation."""
    def __init__(self, workspace, cards, owner, task):
        if not cards or len(set(cards)) != len(cards):
            raise ValueError('Supply distinct GPU indices')
        self.root = state_root(workspace) / 'gpu-locks'
        self.cards = list(cards)
        self.owner, self.task, self.handles, self.gpus = owner, task, {}, {}

    def acquire(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            inv = inventory()
            for card in sorted(self.cards):
                h = (self.root / ('gpu-' + str(card) + '.lock')).open('a+')
                self.handles[card] = h
                fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                g = inv[card]
                if g['used_mib'] > 512 or g['utilization'] > 0:
                    raise RuntimeError('GPU busy: ' + str(card) + ' ' + json.dumps(g))
                self.gpus[card] = g
            physical = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                 '--format=csv,noheader'], text=True)
            for g in self.gpus.values():
                if any(line.split(',')[0].strip() == g['uuid'] for line in physical.splitlines()):
                    raise RuntimeError('GPU has a compute process: ' + g['uuid'])
            write(self.root.parent / 'running' / (self.task.replace(':', '-') + '.json'),
                  dict(owner=self.owner, task=self.task, pid=os.getpid(), host=socket.gethostname(),
                       gpus=self.gpus, started=time.time()))
            return self
        except BaseException:
            self.release()
            raise

    def release(self):
        for h in self.handles.values():
            h.close()
        self.handles.clear()
        p = self.root.parent / 'running' / (self.task.replace(':', '-') + '.json')
        if p.exists() and read(p).get('pid') == os.getpid():
            p.unlink()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()


def show_status(workspace):
    with contextlib.closing(db(workspace)) as c:
        return [dict(r) | {'detail': json.loads(r['detail'])} for r in c.execute('SELECT * FROM tasks ORDER BY id')]
