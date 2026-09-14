"""Visible subprocess output + local log; never prints API credentials."""
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time


def run_visible(command, env, log_path):
    log_path = Path(log_path)
    if log_path.exists():
        raise RuntimeError('Choose a new log filename; existing logs are not overwritten.')
    secrets = [value for name, value in env.items() if ('KEY' in name or 'TOKEN' in name) and len(value) >= 8]
    def clean(value):
        for secret in secrets:
            value = value.replace(secret, '[REDACTED]')
        return value
    messages = queue.Queue()
    with log_path.open('x', encoding='utf-8', buffering=1) as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        def read():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        started = time.monotonic()
        try:
            while True:
                try:
                    line = messages.get(timeout=30)
                except queue.Empty:
                    print(f'Jarayon ishlayapti: {(time.monotonic()-started)/60:.1f} min. '
                          'Yuklash/tekshirish/baholashda loglar oralig‘i uzun bo‘lishi mumkin.', flush=True)
                    continue
                if line is None:
                    break
                line = clean(line)
                print(line, end='', flush=True)
                log.write(line)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print('Jarayon hali to‘xtamadi. Yangi run boshlamang yoki runtime reset qilmang.', flush=True)
            raise
        finally:
            if process.poll() is not None:
                reader.join(timeout=2)
                process.stdout.close()
    print('Trening exit code:', code, 'Log:', log_path, flush=True)
    if code:
        raise RuntimeError('Trening to‘xtadi. Yuqoridagi asl sababni yuboring; run/checkpointlarni o‘chirmang.')
    return code
