# 01 · Katta korpus

Korpus jami 206 016 juftlik: 171 845 train, 16 708 validation, 17 463 test. Aralash sintetik va identity misollar.

Reja 3 epoch bo‘lgan, tanlangan parent checkpoint esa **3551 / epoch 0.661247**. Bu bosqich to‘liq uch epoch o‘qitilgan deb ko‘rsatilmaydi. Keyingi fine-tuning shu tekshirilgan modeldan boshlangan.

LR 5e-5; physical batch 1, accumulation 32; effective batch 32; warmup 5%; Adafactor; BF16.

- [Tarixiy notebook shabloni](colab.ipynb)
- [Trening sozlamalari](config.json)
- [Saqlangan numeric history](training-history.json)

Source va notebook — namoyish uchun. Xom dataset, private manifest, model va cloud hisob sozlamalari bu repoda yo‘q.
