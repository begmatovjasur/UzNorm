"""Audit only Git-visible release files; never prints matched secret values."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DENIED_SUFFIXES = {'.safetensors', '.pt', '.pth', '.bin', '.onnx', '.zip', '.jsonl', '.csv', '.tsv', '.log', '.pem', '.key'}
DENIED_FOLDERS = {'evaluation-data', 'evaluations', 'logs', 'wandb', 'build', '.venv', 'venv', '__pycache__'}
PATTERNS = {
    'credential': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'),
    'private_cloud_link': re.compile(r'https://(?:drive\.google\.com|drive\.usercontent\.google\.com|wandb\.ai)/'),
    'personal_windows_path': re.compile(r'[A-Za-z]:[/\\]+Users[/\\]+[^/\\\s]+'),
}


def files():
    result = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], check=True, capture_output=True)
    return sorted(set(name.decode('utf-8') for name in result.stdout.split(b'\0') if name))


def main():
    errors, inventory = [], {}
    for name in files():
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            errors.append((name, 'non-regular-file')); continue
        if path.suffix in DENIED_SUFFIXES or any(p in DENIED_FOLDERS for p in Path(name).parts) or (name.startswith('models/') and name != 'models/README.md'):
            errors.append((name, 'private-or-large-artifact'))
        if path.name.startswith('.env') and path.name != '.env.vscode.example':
            errors.append((name, 'environment-file'))
        content = path.read_bytes()
        if len(content) > 5 * 1024**2:
            errors.append((name, 'file-over-5-MiB')); continue
        try:
            text = content.decode('utf-8-sig')
        except UnicodeError:
            errors.append((name, 'unexpected-binary')); continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append((name, label))
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=name)
            except SyntaxError:
                errors.append((name, 'invalid-python'))
        if path.suffix in ('.json', '.ipynb', '.code-workspace'):
            try:
                value = json.loads(text)
                if path.suffix == '.ipynb':
                    if any(c.get('outputs') or c.get('execution_count') is not None or c.get('attachments') for c in value['cells']):
                        errors.append((name, 'notebook-private-output'))
            except (ValueError, KeyError):
                errors.append((name, 'invalid-json'))
        if path.suffix == '.md':
            for raw in re.findall(r'\]\(([^)]+)\)', text):
                target = raw.split('#', 1)[0].strip('<>')
                if target and not re.match(r'\w+://|mailto:', target) and not (path.parent / target).exists():
                    errors.append((name, 'broken-relative-link'))
        if name != 'PUBLICATION_MANIFEST.json':
            # Git may normalize Windows line endings; attest canonical text bytes.
            canonical = text.replace('\r\n', '\n').encode('utf-8')
            inventory[name] = hashlib.sha256(canonical).hexdigest()
    if errors:
        print(json.dumps({'passed':False, 'issues':errors}, ensure_ascii=False, indent=2))
        return 1
    manifest_path = ROOT / 'PUBLICATION_MANIFEST.json'
    if '--seal' in sys.argv:
        if manifest_path.exists():
            raise RuntimeError('Manifest exists; refusing to overwrite it')
        with manifest_path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump({'schema':1, 'scope':'public-code-only', 'hash_encoding':'UTF-8 with LF line endings', 'files':inventory}, stream, indent=2)
            stream.write('\n')
    elif manifest_path.exists():
        if json.loads(manifest_path.read_text())['files'] != inventory:
            print('PUBLIC_AUDIT_FAILED: release differs from publication manifest')
            return 1
    print(f'PUBLIC_AUDIT_OK: {len(inventory)} files; no blocked artifacts/patterns, valid code/notebooks/links.')
    print('Pattern checks are not a substitute for human privacy and provenance review.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
