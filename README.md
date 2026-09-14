# UzNorm

O‘zbekcha matnni tuzatish uchun lokal AI yordamchi. **ByT5-base**, to‘rtta ketma-ket fine-tuning bosqichi va sodda brauzer interfeysi.

Matn kompyuterning o‘zida qayta ishlanadi. Inference uchun tashqi API, Gemini yoki W&B talab qilinmaydi. Sayt faqat matn kiritish va model taklifini ko‘rish uchun; metrikalar [conclusion report](reports/CONCLUSION_REPORT.md)da.

> **Public code-only release:** model vaznlari, xom izohlar, datasetlar, shaxsiy Drive havolalari va eski notebook chiqishlari repoga qo‘shilmagan. Ishlaydigan inference uchun alohida, tekshirilgan **Quality5k-157 FINAL** model arxivi kerak. Reponi klonlashning o‘zi modelni yuklamaydi.

## Qisqa natijalar

Ajratilgan **62 misollik Silver sinovda**, bir xil etalonga nisbatan input → model javobi:

| Mezon | Input | Model |
|---|---:|---:|
| CER ↓ | 8.13% | 4.61% |
| Mazmuniy WER ↓ | 31.69% | 22.89% |
| Punctuation F1 ↑ | 43.61% | 67.75% |
| Apostrof/tutuq F1 ↑ | 64.41% | 89.19% |

CER nisbatan **43.4% kamaygan**; ushbu sinovdagi 13 ta toza nazorat matni saqlangan. Bu kichik Silver to‘plamdagi natija, umumiy o‘zbek tili accuracy da’vosi emas. To‘liq natijalar, taqqoslash chegaralari va boshqa bosqichlar reportda saqlangan.

## Windows / VS Code’da ishga tushirish

Python **3.12 x64**, Git hamda VS Code Python/debugpy kengaytmalari tavsiya etiladi. Buyruqlarni ushbu loyiha papkasida bajaring:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

O‘rnatish internet talab qiladi. Model import qilingach, oddiy matn tuzatish lokal ishlaydi.

Model arxivi sizda bo‘lsa, bir marta import qiling:

```powershell
.\run.cmd import-model "D:\Models\quality5k-final.zip"
.\run.cmd verify-model
.\run.cmd web
```

`D:\Models\quality5k-final.zip` — misol yo‘li, haqiqiy arxivingiz yo‘li bilan almashtiring. Importer aynan shu modelning imzosi va fayl hashlarini tekshiradi; boshqa checkpointni jim tanlamaydi. [Model o‘rnatish](models/README.md).

Brauzerda **http://127.0.0.1:8765/** ni oching. Keyingi safar `web.cmd`ni ikki marta bosish yetarli. Terminalni ochiq qoldiring; to‘xtatish uchun Ctrl+C.

VS Code’da **UzNorm.code-workspace → Python: Select Interpreter → .venv → Run and Debug → UzNorm: lokal sayt → F5**.

## Namoyish xaritasi

| Bo‘lim | Mazmuni |
|---|---|
| [Arxitektura](docs/ARCHITECTURE.md) | ByT5 encoder–decoder va lokal ilova |
| [Trening tarixi](training/README.md) | Katta korpus → 10k → real izohlar → 5k |
| [Conclusion report](reports/CONCLUSION_REPORT.md) | Metrikalar va taqqoslash metodologiyasi |
| [Nashr chegaralari](docs/PUBLICATION.md) | Nimalar kiritilgan va nimalar chiqarilgan |
| [Xavfsizlik](SECURITY.md) | Lokal server, matn va model himoyasi |

```text
UzNorm/
├── src/uznorm_studio/   # CPU inference, CLI, desktop GUI, lokal web server
│   └── web/dist/        # HTML, CSS, JavaScript
├── tests/              # fake/tiny model bilan dastur testlari
├── training/           # 4 bosqich: tarixiy kod, tozalangan notebook, numeric history
├── reports/            # umumlashtirilgan conclusion report
├── docs/               # arxitektura va public nashr doirasi
├── scripts/            # public fayllarni tekshirish
├── models/             # vaznlar lokal saqlanadi; Git ularni e’tiborsiz qoldiradi
└── web.cmd             # Windows’da saytni boshlash
```

## Testlar

```powershell
.\run.cmd doctor
.\run.cmd test
.\.venv\Scripts\python.exe -B scripts\audit_public.py
```

Testlar haqiqiy vaznlarni yuklamaydi, tashqi API chaqirmaydi va trening boshlamaydi. HTTP testlari faqat loopback’da vaqtinchalik port ishlatadi. `training/**/source/` tarixiy yozuvdir; yuqoridagi test buyrug‘i faqat lokal ilova testlarini bajaradi.

## Model va foydalanish

Faol model: **quality5k-157**; parent zanjiri **3551 → 313 → 392 → 157**. Qadamlar har yangi bosqichda yangidan sanaladi. Asosiy model arxitekturasi Google tomonidan yaratilgan; UzNorm uni o‘zbekcha tuzatishga moslashtiradi.

Inference: CPU FP32, greedy, qo‘shimcha tuzatish qoidalarisiz. Chegara: **511 UTF-8 bayt**, 511 so‘z emas. Input yashirin kesilmaydi. Model tahrir taklif qiladi; ma’no, nomlar va raqamlarni foydalanuvchi tekshiradi.

## Trening va baholashni takrorlash

Public notebooklar tarixiy kod namoyishi uchun; **Run All qilmang**. Ular to‘liq, mustaqil Colab release paketi emas: xom train/test materiallari va shaxsiy bulut sozlamalari kiritilmagan. Model baholash kodi mavjud, ammo original muzlatilgan korpus alohida kerak; korpussiz tarixiy raqamlarni qayta o‘lchagan deb da’vo qilinmaydi.

## Mualliflik va uchinchi tomonlar

Model, kutubxonalar va dataset manbalari haqida [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)ni ko‘ring. Ushbu nashrga loyiha uchun alohida qayta foydalanish litsenziyasi hali tanlanmagan. Public ko‘rinish model/datasetlarni qayta tarqatish uchun alohida ruxsat o‘rnini bosmaydi.
