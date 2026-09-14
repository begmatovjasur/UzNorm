# Trening laboratoriyasi

**google/byt5-base → 3551 → 313 → 392 → 157**. Qadamlar bosqichlar kesimida yangidan sanaladi; 157 eskiroq model degani emas.

| Bosqich | Train | Parent → checkpoint | Bajarilgan epoch | LR | Effektiv batch |
|---|---:|---|---:|---:|---:|
| [Katta korpus](01_base_200k/README.md) | 171 845 | Google → 3551 | 0.6612 / reja 3 | 5e-5 | 32 |
| [Correction 10k](02_correction_10k/README.md) | 10 000 | 3551 → 313 | 1 | 2e-5 | 32 |
| [Real izohlar](03_real_reviews/README.md) | 12 541 | 313 → 392 | 1 | 1e-5 | 32 |
| [Quality 5k](04_quality_5k/README.md) | 5 000 | 392 → 157 | 1 | 1e-5 | 32 |

Hamma bosqich full fine-tuning: Adafactor, L4 GPU, BF16. Keyingi dataset bosqichlari oldingi vaznlar bilan, yangi optimizer/scheduler bilan boshlangan. Shu bosqichning uzilgan runini davom ettirishda esa to‘liq checkpoint holati tiklangan.

`colab.ipynb` fayllari kodni ko‘rsatish uchun; public nusxada output va shaxsiy havolalar olib tashlangan. Birinchi ikki notebook asl dossierda ham shablon edi. `training-history.json` saqlangan trainer_state’dan olingan haqiqiy numeric yozuvlar; yetishmayotgan loglar uydirilmagan.

`source/` — tarixiy source code, to‘liq qayta o‘qitish paketi emas. Dataset va original manifestlar alohida; **Run All qilmang**. Taqsimot, metrikalar va metodologiya [conclusion report](../reports/CONCLUSION_REPORT.md)da.
