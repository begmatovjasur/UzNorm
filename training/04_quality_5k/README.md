# 04 · Quality 5k

Parent **real-392 → quality5k-157**, 1 epoch. LR 1e-5, micro batch 2, accumulation 16, effective batch 32; warmup 5%; Adafactor; BF16; har 25 qadamda tasdiqlanadigan cloud backup.

| Kategoriya | Juftlik |
|---|---:|
| Lexical | 2 000 |
| Chat/mixed | 1 000 |
| Format | 750 |
| Identity | 1 000 |
| Protected entities | 250 |

60 Silver development misolida oldin/keyin taqqoslash; final test trening notebookida ishlatilmagan. 5k: 1 963 publisher MCQ moslashuvi, 1 769 qayta tekshirilgan real-train, 1 018 reviewed targetdan sintetik input va 250 nazoratli entity shabloni. Barchasi yangi tabiiy izoh yoki Gold emas.

- [Outputlari tozalangan tarixiy notebook](colab.ipynb)
- [Trening sozlamalari](config.json)
- [Aggregate dataset pasporti](dataset.json)
- [Saqlangan numeric history](training-history.json)

Natijalar [conclusion report](../../reports/CONCLUSION_REPORT.md)da. Source/ va notebook namoyish uchun, dataset/model/manifestlarsiz mustaqil Colab release emas.
