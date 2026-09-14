"""GUI and terminal entry points. Merely importing this module loads no model."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from . import __version__
from .artifacts import ModelError, import_archive, inspect_model
from .config import Settings
from .logging_setup import configure
from .service import Corrector


def doctor(settings):
    print(f"UzNorm {__version__} | Python {sys.version.split()[0]}")
    print("Python:", sys.executable)
    print("Loyiha:", settings.home)
    print("Model:", settings.model_dir)
    print(f"Rejim: CPU FP32 / {settings.threads} threads / offline")
    missing = []
    for package in ("torch", "transformers", "safetensors"):
        try:
            print(package + ":", importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
            print(package + ": O‘RNATILMAGAN")
    from .tk_runtime import prepare_tk
    try:
        prepare_tk()
        import tkinter
        print("Tcl:", tkinter.Tcl().eval("info patchlevel"))
    except Exception as exc:
        print("GUI muhiti tayyor emas:", type(exc).__name__)
        print("Python o‘rnatishda Tcl/Tk komponentini tanlang. CLI bundan mustaqil.")
    print("Model papkasi:", "mavjud (hash hali tekshirilmadi)" if settings.model_dir.exists() else "ZIP IMPORT KUTILMOQDA")
    print("Model yuklanmadi; API chaqirilmadi.")
    return 2 if missing else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="UzNorm — mahalliy ByT5 matn tuzatish")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--home", type=Path, help="Model va loglar saqlanadigan loyiha papkasi")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads: 1–8 (standart: 4)")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("gui", help="Desktop matn oynasi")
    web = commands.add_parser("web", help="Faqat localhost’da ishlaydigan sayt")
    web.add_argument("--port", type=int, default=8765)
    commands.add_parser("evaluate", help="Muzlatilgan korpusda input va modelni bir xil etalon bilan baholash")
    commands.add_parser("doctor", help="Muhitni tekshirish; model yuklanmaydi")
    commands.add_parser("verify-model", help="Mahalliy modelning SHA-256 tekshiruvi")
    importer = commands.add_parser("import-model", help="Yangi 5k FINAL ZIPni bir marta import qilish")
    importer.add_argument("archive", type=Path)
    correct = commands.add_parser("correct", help="Bitta matn yoki interaktiv sessiya")
    correct.add_argument("--text", help="Kiritilmasa, interaktiv sessiya ochiladi")
    correct.add_argument("--json", action="store_true", help="Natijani JSON shaklida chiqarish")
    args = parser.parse_args(argv)
    try:
        settings = Settings.default(args.home, args.threads)
        if args.command == "doctor":
            return doctor(settings)
        configure(settings.log_dir)
        progress = lambda message: print(message, file=sys.stderr, flush=True)
        if args.command == "import-model":
            import_archive(args.archive, settings.model_dir, progress)
        elif args.command == "web":
            from .web_server import serve
            serve(settings, args.port)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            service = Corrector(settings)
            try:
                report = evaluate(service, settings.home, progress)
                print(json.dumps(dict(n=report['n'], step=report['step'], suites=[s['title'] for s in report['suites']]), ensure_ascii=False))
            finally:
                service.close()
        elif args.command == "verify-model":
            meta = inspect_model(settings.model_dir, progress)
            print(f"MODEL_OK run={meta['run_id']} step={meta['step']}")
        elif args.command == "correct":
            service = Corrector(settings)
            try:
                service.load(progress)
                def predict(text):
                    result = service.correct(text)
                    if args.json:
                        print(json.dumps(result.to_dict(), ensure_ascii=False))
                    else:
                        print(result.output)
                        progress(f"CPU: {result.seconds:.2f} s | qadam {result.step} | qo‘shimcha tuzatishsiz")
                    if not result.ended_with_eos:
                        progress("OGOHLANTIRISH: javob uzunlik chegarasida to‘xtagan bo‘lishi mumkin.")
                if args.text is not None:
                    predict(args.text)
                else:
                    progress("Matn kiriting. Chiqish: /exit yoki Ctrl+C. Model sessiyada bir marta yuklanadi.")
                    while True:
                        try:
                            text = input("Matn > ")
                        except EOFError:
                            break
                        if text.strip() == "/exit":
                            break
                        try:
                            predict(text)
                        except ValueError as exc:
                            progress(str(exc))
            finally:
                service.close()
        else:
            from .gui import launch
            launch(settings)
        return 0
    except KeyboardInterrupt:
        print("Sessiya yopildi. Model fayllari o‘zgarmadi.", file=sys.stderr)
        return 130
    except (ModelError, ValueError, OSError, ImportError) as exc:
        print(f"XATO: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"XATO ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
