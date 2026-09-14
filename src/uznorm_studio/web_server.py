"""Loopback-only web UI, same-origin ephemeral token, one serialized CPU worker.

Not an Internet-facing production server. Never exposes arbitrary filesystem paths.
"""
import difflib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit

from .artifacts import ModelError
from .config import policy
from .service import Corrector, validate_text

ASSETS = Path(__file__).with_name('web') / 'dist'
MAX_BODY = 8192


class Application:
    def __init__(self, settings):
        self.settings = settings
        self.engine = Corrector(settings)
        self.token = secrets.token_urlsafe(32)
        self.guard = threading.Lock()
        self.busy = False
        self.job = 0
        self.kind = ''
        self.message = 'Model diskda. Birinchi so‘rovda xotiraga yuklanadi.'
        self.error = None
        self.result = None

    def progress(self, message):
        with self.guard:
            self.message = message

    def snapshot(self):
        with self.guard:
            return dict(busy=self.busy, ready=self.engine.ready, job=self.job, kind=self.kind,
                message=self.message, error=self.error, result=self.result)

    def start(self, kind, work):
        with self.guard:
            if self.busy:
                raise ModelError('Hozir boshqa amal bajarilmoqda. Tugashini kuting.')
            self.busy, self.kind, self.error, self.result = True, kind, None, None
            self.message = 'Model tayyorlanmoqda…'
            self.job += 1
            job = self.job
        def worker():
            try:
                self.engine.load(self.progress)
                result = work()
                with self.guard:
                    self.result = result
                    self.message = 'Tayyor.'
            except Exception as exc:
                logging.getLogger('uznorm_studio').error('web_job_failed type=%s', type(exc).__name__)
                with self.guard:
                    self.error = str(exc)
                    self.message = 'Amal bajarilmadi.'
            finally:
                with self.guard:
                    self.busy = False
        threading.Thread(target=worker, daemon=True, name='uznorm-web-worker').start()
        return job

    def correct(self, text):
        result = self.engine.correct(text).to_dict()
        result['diff'] = [dict(kind=op, before=text[a:b], after=result['output'][c:d])
            for op, a, b, c, d in difflib.SequenceMatcher(None, text, result['output'], autojunk=False).get_opcodes()]
        return result


def make_handler(app, port):
    origin = f'http://127.0.0.1:{port}'
    class Handler(BaseHTTPRequestHandler):
        server_version = 'UzNormLocal/1.1'

        def log_message(self, fmt, *args):
            pass  # Never log URLs, bodies, model inputs, or outputs.

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def send(self, status, body, content_type='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            for key, value in {
                'Content-Type': content_type, 'Content-Length': str(len(body)), 'Cache-Control': 'no-store',
                'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', 'X-Frame-Options': 'DENY',
                'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
            }.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def same_origin(self):
            return (self.headers.get('Host') == f'127.0.0.1:{port}'
                and self.headers.get('Origin', origin) == origin
                and self.headers.get('Sec-Fetch-Site', 'none') in ('same-origin', 'none'))

        def authorized(self):
            return self.same_origin() and secrets.compare_digest(self.headers.get('X-UzNorm-Session', ''), app.token)

        def do_GET(self):
            if not self.same_origin():
                return self.send(403, {'error': 'Faqat lokal saytning o‘zidan foydalaning.'})
            path = urlsplit(self.path).path
            if path == '/api/session':
                return self.send(200, dict(token=app.token, model=policy()['label'], run_id=policy()['binding']['run_id'], step=157, threads=app.settings.threads))
            if path.startswith('/api/'):
                if not self.authorized():
                    return self.send(403, {'error': 'Sessiya tekshiruvi o‘tmadi. Sahifani yangilang.'})
                if path == '/api/status':
                    return self.send(200, app.snapshot())
                return self.send(404, {'error': 'Topilmadi.'})
            allowed = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'application/javascript; charset=utf-8'),
                       '/style.css': ('style.css', 'text/css; charset=utf-8'), '/favicon.svg': ('favicon.svg', 'image/svg+xml')}
            if path not in allowed:
                return self.send(404, {'error': 'Topilmadi.'})
            filename, content_type = allowed[path]
            return self.send(200, (ASSETS / filename).read_bytes(), content_type)

        def do_POST(self):
            if not self.authorized():
                return self.send(403, {'error': 'Sessiya yoki manba tekshiruvi o‘tmadi.'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.send(415, {'error': 'JSON talab qilinadi.'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_BODY:
                    return self.send(413, {'error': 'So‘rov hajmi noto‘g‘ri.'})
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('JSON obyekt kerak.')
                path = urlsplit(self.path).path
                if path == '/api/correct':
                    if set(data) != {'text'}:
                        raise ValueError('Faqat text maydoni qabul qilinadi.')
                    text = data.get('text')
                    validate_text(text)
                    job = app.start('correct', lambda: app.correct(text))
                elif path == '/api/load':
                    job = app.start('load', lambda: {'loaded': True})
                else:
                    return self.send(404, {'error': 'Topilmadi.'})
                return self.send(202, dict(job=job))
            except ModelError as exc:
                return self.send(409, {'error': str(exc)})
            except (ValueError, TypeError, OSError) as exc:
                return self.send(400, {'error': str(exc)})

        def do_OPTIONS(self):
            self.send(403, {'error': 'Cross-origin so‘rovlar taqiqlangan.'})
    return Handler


def serve(settings, port=8765):
    if not 1024 <= port <= 65535:
        raise ValueError('Port 1024–65535 oralig‘ida bo‘lsin.')
    app = Application(settings)
    server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(app, port))
    server.daemon_threads = True
    print(f'UZNorm LOCAL: http://127.0.0.1:{port}', flush=True)
    print('Faqat lokal. Tugatish: Ctrl+C. Model internetga ulanmaydi.', flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print('Lokal sayt yopilmoqda…', flush=True)
    finally:
        server.server_close()
        app.engine.close()
