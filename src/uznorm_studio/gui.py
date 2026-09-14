"""Native desktop view. One worker at a time; widgets are updated on the UI thread."""
import difflib
import logging
import queue
import threading

from . import __version__
from .artifacts import import_archive, write_new
from .config import policy
from .service import Corrector, validate_text
from .tk_runtime import prepare_tk


class Studio:
    def __init__(self, root, settings):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root, self.settings = root, settings
        self.service = Corrector(settings)
        self.events, self.busy, self.prediction = queue.Queue(), False, None
        root.title("UzNorm · Mahalliy ByT5")
        root.geometry("1120x760")
        root.minsize(820, 620)
        root.configure(bg="#f3f5f8")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f5f8")
        style.configure("TLabel", background="#f3f5f8", foreground="#172b43", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(14, 9))
        style.configure("Accent.TButton", background="#087f8c", foreground="white", borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#096a77"), ("disabled", "#bdc9ce")])
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 12))
        style.configure("Muted.TLabel", foreground="#5e6d7e")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        header = tk.Frame(root, bg="#152b43", padx=26, pady=22)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        tk.Label(header, text="UzNorm", font=("Segoe UI Semibold", 24), fg="white", bg="#152b43").grid(row=0, column=0, sticky="w")
        tk.Label(header, text="O‘zbekcha matnni tuzatish · ByT5 / Quality 5k / 157", font=("Segoe UI", 11), fg="#b7cbdc", bg="#152b43").grid(row=1, column=0, sticky="w", pady=(5, 0))
        tk.Label(header, text=f"OFFLINE\nCPU · {settings.threads} threads", font=("Segoe UI Semibold", 10), fg="#74d6cb", bg="#152b43", justify="right").grid(row=0, column=1, rowspan=2, sticky="e")

        toolbar = ttk.Frame(root, padding=(26, 18, 26, 12))
        toolbar.grid(row=1, column=0, sticky="ew")
        toolbar.columnconfigure(2, weight=1)
        self.import_button = ttk.Button(toolbar, text="1. Model ZIPini ulash", command=self.import_model)
        self.import_button.grid(row=0, column=0, padx=(0, 8))
        self.load_button = ttk.Button(toolbar, text="2. Modelni yuklash", command=self.load_model)
        self.load_button.grid(row=0, column=1)
        self.model_status = tk.StringVar(value="Model RAMga yuklanmagan")
        ttk.Label(toolbar, textvariable=self.model_status, style="Muted.TLabel").grid(row=0, column=2, sticky="e", padx=(10, 0))

        editors = ttk.Frame(root, padding=(26, 0, 26, 0))
        editors.grid(row=2, column=0, sticky="nsew")
        editors.columnconfigure((0, 1), weight=1, uniform="panes")
        editors.rowconfigure(0, weight=1)
        self.input_box = self.make_editor(editors, 0, "Siz yozgan matn", "Imlo xatolari bor matn yoki oddiy izoh kiriting.")
        self.output_box = self.make_editor(editors, 1, "Modelning javobi", "Yashil — qo‘shilgan/almashtirilgan; qizil — o‘zgargan kirish qismi.")
        self.output_box.configure(state="disabled")
        self.input_box.tag_configure("changed", background="#ffe7e7", foreground="#8b2830")
        self.output_box.tag_configure("changed", background="#d6f2e7", foreground="#155947")
        self.input_box.bind("<<Modified>>", self.input_changed)
        self.input_box.bind("<Control-Return>", lambda event: self.correct() or "break")

        actions = ttk.Frame(root, padding=(26, 14, 26, 14))
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(1, weight=1)
        self.correct_button = ttk.Button(actions, text="Matnni tuzatish  ·  Ctrl+Enter", style="Accent.TButton", command=self.correct)
        self.correct_button.grid(row=0, column=0, sticky="w")
        self.byte_label = tk.StringVar(value="0 / 511 UTF-8 bayt")
        ttk.Label(actions, textvariable=self.byte_label, style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=16)
        self.copy_button = ttk.Button(actions, text="Nusxa olish", command=self.copy)
        self.copy_button.grid(row=0, column=2, padx=4)
        self.save_button = ttk.Button(actions, text="JSON saqlash", command=self.save)
        self.save_button.grid(row=0, column=3, padx=4)
        self.clear_button = ttk.Button(actions, text="Tozalash", command=self.clear)
        self.clear_button.grid(row=0, column=4, padx=(4, 0))

        footer = ttk.Frame(root, padding=(26, 0, 26, 18))
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.status = tk.StringVar(value="Avval ZIPni import qiling, so‘ng modelni yuklang. Import faqat bir marta kerak.")
        ttk.Label(footer, textvariable=self.status, wraplength=1000).grid(row=0, column=0, sticky="w")
        self.progress_bar = ttk.Progressbar(footer, mode="indeterminate")
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        ttk.Label(footer, text="Matn internetga yuborilmaydi va logga yozilmaydi. Javob — qo‘shimcha qoidasiz xom model chiqishi.\nModel xato qilishi mumkin: ma’no, raqam va nomlarni tekshiring. Bu oynada trening bajarilmaydi.", style="Muted.TLabel", wraplength=1000).grid(row=2, column=0, sticky="w")
        ttk.Label(footer, text=f"v{__version__}  /  {policy()['binding']['run_id']}", style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(8, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)
        self.refresh()

    def make_editor(self, parent, column, title, subtitle):
        panel = self.ttk.Frame(parent, padding=(0 if column == 0 else 10, 0, 10 if column == 0 else 0, 0))
        panel.grid(row=0, column=column, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)
        self.ttk.Label(panel, text=title, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.ttk.Label(panel, text=subtitle, style="Muted.TLabel", wraplength=480).grid(row=1, column=0, sticky="w", pady=(6, 12))
        text = self.tk.Text(panel, wrap="word", width=1, height=8, font=("Segoe UI", 14), padx=18, pady=16,
                            bg="white", fg="#172b43", insertbackground="#087f8c", relief="flat", undo=column == 0,
                            highlightthickness=1, highlightbackground="#d8dfe7", highlightcolor="#087f8c", spacing3=5)
        text.grid(row=2, column=0, sticky="nsew")
        scroll = self.ttk.Scrollbar(panel, orient="vertical", command=text.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)
        return text

    def input_changed(self, event=None):
        if self.input_box.edit_modified():
            value = self.input_box.get("1.0", "end-1c")
            self.byte_label.set(f"{len(value.encode('utf-8', errors='replace'))} / 511 UTF-8 bayt")
            self.input_box.tag_remove("changed", "1.0", "end")
            self.input_box.edit_modified(False)
            if self.prediction is not None and value != self.prediction.input:
                self.status.set("Kirish matni o‘zgardi. O‘ngdagi javob oldingi matnga tegishli; qayta tekshiring.")

    def refresh(self):
        enabled = not self.busy
        for button in (self.import_button, self.load_button, self.clear_button):
            button.configure(state="normal" if enabled else "disabled")
        if self.service.ready:
            self.import_button.configure(state="disabled")
            self.load_button.configure(state="disabled")
            self.model_status.set("TAYYOR · Quality 5k / 157")
        elif self.settings.model_dir.exists():
            self.model_status.set("Model diskda · yuklash kerak")
        self.correct_button.configure(state="normal" if enabled and self.service.ready else "disabled")
        for button in (self.copy_button, self.save_button):
            button.configure(state="normal" if enabled and self.prediction else "disabled")
        self.input_box.configure(state="normal" if enabled else "disabled")

    def start_job(self, label, function):
        if self.busy:
            return
        self.busy = True
        self.status.set(label)
        self.progress_bar.start(12)
        self.refresh()
        def work():
            try:
                self.events.put(("done", function()))
            except Exception as exc:
                logging.getLogger("uznorm_studio").error("operation_failed type=%s", type(exc).__name__)
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        threading.Thread(target=work, daemon=True, name="uznorm-worker").start()

    def progress(self, message):
        self.events.put(("progress", message))

    def poll(self):
        from tkinter import messagebox
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    self.status.set(value)
                    continue
                self.busy = False
                self.progress_bar.stop()
                if kind == "error":
                    self.status.set("Amal bajarilmadi. Oldingi model fayllari o‘zgarmadi.")
                    messagebox.showerror("UzNorm", value, parent=self.root)
                elif value is not None and hasattr(value, "output"):
                    self.show_prediction(value)
                else:
                    self.status.set("Model tayyor. Matn kiriting." if self.service.ready else "ZIP tekshirildi va import qilindi. Endi «Modelni yuklash»ni bosing.")
                self.refresh()
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def import_model(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(parent=self.root, title="Quality 5k · 157 FINAL ZIP", filetypes=[("ZIP arxiv", "*.zip")])
        if filename:
            self.start_job("Arxiv tekshirilmoqda…", lambda: import_archive(filename, self.settings.model_dir, self.progress))

    def load_model(self):
        self.start_job("Model tekshirilmoqda va RAMga yuklanmoqda…", lambda: self.service.load(self.progress))

    def correct(self):
        from tkinter import messagebox
        if self.busy or not self.service.ready:
            return
        text = self.input_box.get("1.0", "end-1c")
        try:
            validate_text(text)
        except ValueError as exc:
            messagebox.showwarning("Matnni tekshiring", str(exc), parent=self.root)
            return
        self.start_job("Model matnni tekshirmoqda… CPU tezligi matn uzunligiga bog‘liq.", lambda: self.service.correct(text))

    def show_prediction(self, result):
        self.prediction = result
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", "end")
        self.output_box.insert("1.0", result.output)
        self.input_box.tag_remove("changed", "1.0", "end")
        for operation, a, b, c, d in difflib.SequenceMatcher(None, result.input, result.output, autojunk=False).get_opcodes():
            if operation != "equal":
                # Tk 8.6 counts supplementary Unicode characters as two UTF-16 units.
                def index(text, offset):
                    units = len(text[:offset].encode("utf-16-le")) // 2
                    return f"1.0+{units}c"
                self.input_box.tag_add("changed", index(result.input, a), index(result.input, b))
                self.output_box.tag_add("changed", index(result.output, c), index(result.output, d))
        self.output_box.configure(state="disabled")
        warning = " · DIQQAT: javob uzunlik chegarasida tugagan bo‘lishi mumkin" if not result.ended_with_eos else ""
        self.status.set(f"Tayyor · {result.seconds:.2f} soniya · CPU FP32 · qadam {result.step}{warning}")

    def copy(self):
        if self.prediction:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.prediction.output)
            self.status.set("Model javobi nusxalandi.")

    def save(self):
        from tkinter import filedialog, messagebox
        if not self.prediction:
            return
        filename = filedialog.asksaveasfilename(parent=self.root, title="Matn va javobni saqlash", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if filename:
            try:
                write_new(filename, self.prediction.to_dict())
                self.status.set("Kirish matni va model javobi tanlangan JSON fayliga saqlandi.")
            except OSError as exc:
                messagebox.showerror("Saqlanmadi", f"Yangi fayl nomini tanlang. {exc}", parent=self.root)

    def clear(self):
        self.prediction = None
        self.input_box.delete("1.0", "end")
        self.output_box.configure(state="normal")
        self.output_box.delete("1.0", "end")
        self.output_box.configure(state="disabled")
        self.status.set("Yangi matn kiriting.")
        self.refresh()

    def close(self):
        from tkinter import messagebox
        if self.busy:
            messagebox.showinfo("Amal davom etmoqda", "Import yoki joriy javob tugashini kuting; oynani keyin yoping.", parent=self.root)
            return
        self.service.close()
        self.root.destroy()


def launch(settings):
    prepare_tk()
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        raise RuntimeError("Desktop uchun Tcl/Tk ishga tushmadi. README’dagi Python muhiti bo‘limini ko‘ring; CLI mustaqil ishlaydi.") from None
    Studio(root, settings)
    root.mainloop()
