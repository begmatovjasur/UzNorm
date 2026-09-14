"""Local HTTP contract tests. Fake model only; no external network or real weights."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer

from uznorm_studio.config import Settings
from uznorm_studio.service import Prediction
from uznorm_studio.web_server import Application, make_handler, ASSETS
from uznorm_studio.evaluation import comparison


class FakeEngine:
    ready = False
    def load(self, progress):
        self.ready = True
    def correct(self, text):
        from types import SimpleNamespace
        return SimpleNamespace(to_dict=lambda: dict(input=text, output=text, seconds=0.01,
                                                    ended_with_eos=True, step=157))
    def close(self):
        pass


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch('uznorm_studio.web_server.Corrector', return_value=FakeEngine()):
            self.app = Application(Settings.default(Path(self.temp.name)))
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), lambda *args: None)
        self.port = self.server.server_address[1]
        self.server.RequestHandlerClass = make_handler(self.app, self.port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, data=None, headers=None, auth=True):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        h = {'Content-Type':'application/json'}
        if auth:
            h['X-UzNorm-Session'] = self.app.token
        h.update(headers or {})
        conn.request(method,path,body=None if data is None else json.dumps(data),headers=h)
        response = conn.getresponse()
        code, body = response.status, response.read()
        conn.close()
        return code, body

    def test_site_has_only_manual_correction(self):
        code, page = self.request('GET','/')
        self.assertEqual(code,200)
        html = page.decode()
        self.assertIn('Sizning matningiz',html)
        for forbidden in ('view-report','id="evaluate"','id="reference"','manual-metrics','WER','CER'):
            self.assertNotIn(forbidden,html)
        js = (ASSETS/'app.js').read_text(encoding='utf-8')
        self.assertNotIn("api('report')",js)
        self.assertNotIn("api('datasets')",js)

    def test_report_routes_removed(self):
        for path in ('/api/report','/api/datasets'):
            self.assertEqual(self.request('GET',path)[0],404)
        self.assertEqual(self.request('POST','/api/evaluate',{})[0],404)
        self.assertNotIn('report_available',self.app.snapshot())

    def test_csrf_and_host_checks(self):
        self.assertEqual(self.request('GET','/api/status',auth=False)[0],403)
        self.assertEqual(self.request('GET','/api/session',headers={'Host':'evil.invalid'})[0],403)
        self.assertEqual(self.request('POST','/api/correct',{'text':'Salom'},headers={'Origin':'https://evil.invalid'})[0],403)

    def test_rejects_bad_or_reference_input(self):
        for value in ('', 'a'*512, None, 123):
            self.assertEqual(self.request('POST','/api/correct',{'text':value})[0],400)
        self.assertEqual(self.request('POST','/api/correct',{'text':'Salom','reference':'Salom'})[0],400)

    def test_correction_is_async_and_raw(self):
        text='<b>Salom</b>'
        code,body=self.request('POST','/api/correct',{'text':text})
        self.assertEqual(code,202)
        limit=time.monotonic()+3
        while self.app.busy and time.monotonic()<limit:
            time.sleep(.01)
        state=self.app.snapshot()
        self.assertFalse(state['busy'])
        self.assertEqual(state['result']['output'],text)
        self.assertNotIn('comparison',state['result'])

    def test_busy_rejected(self):
        self.app.busy=True
        self.assertEqual(self.request('POST','/api/correct',{'text':'Salom'})[0],409)


class MetricTests(unittest.TestCase):
    def test_paired_known_edit_distance(self):
        r=comparison([{'input':'cat','target':'cot'}],['cot'])
        self.assertAlmostEqual(r['before']['raw_cer_pct'],100/3)
        self.assertEqual(r['after']['raw_cer_pct'],0)
        self.assertEqual(r['transitions']['fully_fixed'],1)
        self.assertEqual(r['before']['word_wer_pct'],100)

    def test_apostrophe_is_evaluation_only(self):
        row={'input':"O'zbek",'target':'O‘zbek'}
        raw=comparison([row],[row['input']])
        normalized=comparison([row],[row['input']],True)
        self.assertGreater(raw['after']['raw_cer_pct'],0)
        self.assertEqual(normalized['after']['raw_cer_pct'],0)
        self.assertEqual(row['input'],"O'zbek")
