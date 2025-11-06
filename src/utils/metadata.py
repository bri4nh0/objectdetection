import json
import os
import subprocess
import sys
import hashlib
from datetime import datetime


def git_sha(repo_root):
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_root, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def pip_freeze():
    try:
        out = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], stderr=subprocess.DEVNULL)
        return out.decode().splitlines()
    except Exception:
        return []


def file_checksum(path):
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def save_metadata(run_id, repo_root, config=None, models=None, out_path='results'):
    os.makedirs(out_path, exist_ok=True)
    meta = {
        'run_id': run_id,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'git_sha': git_sha(repo_root),
        'python': sys.version,
        'pip_freeze': pip_freeze(),
        'config': config or {},
        'models': {}
    }
    if models:
        for name, path in models.items():
            meta['models'][name] = {'path': path, 'sha256': file_checksum(path)}

    out_file = os.path.join(out_path, f'{run_id}_metadata.json')
    with open(out_file, 'w') as f:
        json.dump(meta, f, indent=2)
    return out_file
