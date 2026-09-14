# 03 · Real izohlar

Uzum, Commeta va Google Play manbalariga moslashtirish. 13 955 manba juftligi: 12 541 train, 688 validation, 705 test va 21 karantin.

Train: Uzum 5 316, Commeta 4 005, Google Play 3 220. Xom foydalanuvchi izohlari repoga joylanmagan.

Parent **10k-313 → real-392**, 1 epoch. LR 1e-5, micro batch 2, accumulation 16, effective batch 32; warmup 5%; Adafactor; BF16.

- [Outputlari tozalangan tarixiy notebook](colab.ipynb)
- [Trening sozlamalari](config.json)
- [Saqlangan numeric history](training-history.json)

272 monitor (144 real + 128 synthetic) ishlatilgan. To‘liq final comparison mahalliy checkpoint arxivida mavjud emasligi [conclusion report](../../reports/CONCLUSION_REPORT.md)da aniq ko‘rsatilgan; yetishmayotgan natijalar yaratilmagan. Source/ mustaqil retraining kit emas.
