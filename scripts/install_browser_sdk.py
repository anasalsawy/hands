"""Reproduce the edited Browser-Use SDK tested on Railway; never install stock silently."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
vendor = root / 'vendor/browser-use'
source = json.loads((vendor / 'SOURCE.json').read_text())
destination = root / '.browser-use-sdk'

def run(*args):
    subprocess.run(args, check=True)

for name, expected in source['patches'].items():
    if hashlib.sha256((vendor / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Browser SDK patch integrity failure: {name}')
if destination.exists():
    raise SystemExit(f'{destination} already exists; use a clean build directory')
run('git', 'clone', '--filter=blob:none', '--no-checkout', 'https://github.com/browser-use/browser-use.git', str(destination))
run('git', '-C', str(destination), 'checkout', '--detach', source['upstream_commit'])
for name in source['patches']:
    run('git', '-C', str(destination), 'apply', '--check', str(vendor / name))
    run('git', '-C', str(destination), 'apply', str(vendor / name))
run(sys.executable, '-m', 'pip', 'install', '--no-cache-dir', str(root / 'vendor/persona-lanes'), str(destination))
run(sys.executable, '-c', 'from browser_use.persona.agent import PersonaBrowserAgent; assert hasattr(PersonaBrowserAgent, "_start_persona_prefetch")')
