> Public aggregate report. References to private `evidence/` files identify local audit sources; those raw artifacts are intentionally not distributed.

# UzNorm — Conclusion Report

**UzNorm — o‘zbekcha matnni tuzatishga mo‘ljallangan lokal AI yordamchi.** Loyiha to‘rtta ketma-ket fine-tuning bosqichini sodda, kompyuterning o‘zida ishlaydigan ilova bilan birlashtiradi.

## Qisqa natijalar

Ajratilgan **62 misollik Silver sinovda**, bir xil etalonga nisbatan kiritilgan matn → yakuniy model javobi:

- **Belgilar xatosi (CER): 8.13% → 4.61%** — nisbatan **43.4% kamayish**.
- **Mazmuniy so‘z xatosi (WER): 31.69% → 22.89%.**
- **Tinish belgilari F1: 43.61% → 67.75%; apostrof/tutuq F1: 64.41% → 89.19%.**
- Shu sinovdagi **13 ta toza nazorat matnining barchasi saqlangan**.

Amaliy afzalliklar: **tashqi API talab qilinmaydi**, matn lokal kompyuterda qayta ishlanadi, model bir marta yuklanadi va natija sodda interfeysda ko‘rsatiladi. Bu raqamlar aynan ko‘rsatilgan sinovga tegishli; boshqa to‘plamlar bo‘yicha to‘liq natijalar quyida alohida berilgan.

## 1. Maqsad va hisoblash muhiti

O‘zbekcha xato matnni ma’noni saqlagan holda tuzatish: imlo, qisqartma, aralash yozuv, tinish belgisi, registr, o‘/g‘ va tutuq belgisi. Backbone: `google/byt5-base` (~581.7 mln parametr). Trening: Colab NVIDIA L4, BF16; laptopdagi inference: CPU FP32, greedy, qo‘shimcha qoidasiz. VS Code kod va hisobotni ko‘rsatadi; brauzer lokal interfeysni ochadi.

## 2. To‘rtta bosqichning tekshirilgan tarixi

| Bosqich | Trening juftligi | Boshlanish → saqlangan natija | Epoch va holat | Learning rate / samarali batch |
|---|---:|---|---|---|
| “200 minglik” aralash korpus | 171 845 (jami 206 016) | google/byt5-base → 3551 | 0.6612 / rejalangan 3; tugallanmagan | 5e-5 / 32 |
| Correction 10k | 10 000 | 3551 → 313 | 1; tugallangan | 2e-5 / 32 |
| Real izohlar | 12 541 | 313 → 392 | 1; tugallangan | 1e-5 / 32 |
| Quality 5k | 5 000 | 392 → 157 | 1; tugallangan | 1e-5 / 32 |

Asl korpus: 171 845 train, 16 708 validation, 17 463 test. Taxminan 14 minglik real bosqich uch manbani birlashtiradi: 13 955 manba juftligi; train: Uzum 5 316, Commeta 4 005, Google Play 3 220. Split: 12 541 train, 688 validation, 705 test, 21 karantin. 10k avvalgi korpusdan tanlangan; 5k ichida 1 769 real-train juftligi qayta tekshirib ishlatilgan. Hajmlar bosqichlar kesimida berilgan; ularning yig‘indisi noyob misollar soni emas.

Quality 5k tarkibi: 2 000 lexical, 1 000 chat/mixed, 750 format, 1 000 identity, 250 protected entities. Manba turi: 1 963 nashriyot MCQ moslashuvi, 1 769 qayta tekshirilgan real-train, 1 018 tekshirilgan targetdan sintetik input, 250 shablonli entity misoli. Silver; to‘liq inson Gold tasdig‘i yo‘q.

Colab uzilishlari paytida mounted Drive’dan qayta o‘qish bulutga yetib borganini kafolatlamagan. Keyin alohida tasdiqlanadigan cloud arxivlar qo‘shilgan. Eski logda 4750 ko‘rinishi yangi zanjir 4750 dan boshlanganini anglatmaydi: arxivdagi tanlangan parent **3551**. `checkpoint-3000` best bog‘liqligi bor, ammo keyingi bosqichning parenti 3000 emas.

## 3. Natijalarni qanday solishtiramiz?

**Taqqoslash usuli:** har bir oldin/keyin juftligi bir xil to‘plamda o‘lchangan. 128 synthetic monitor, 128 UzLiB, 60 development va 62 final — alohida to‘plamlar; ularning foizlari yagona accuracy chizig‘iga birlashtirilmaydi. CER/WER kamroq, F1/casing/exact match ko‘proq bo‘lsa yaxshi. WER bu yerda mazmuniy token WER; standart whitespace WER yakuniy JSONda ham bor. Punctuation macro F1 — faol sinflar makro-o‘rtachasi; tarixiy hisobotlarda faol sinflar farqlanishi mumkin.

### 3.1 Katta korpus: tarixiy monitor

3551 arxivining eng so‘nggi saqlangan validation yozuvi **3500-qadamda**, n=1024: exact match 86.1328%, CER 0.1684%, WER 0.4322%, punctuation F1 97.6417%. Ushbu tarixiy monitor keyingi moslashtirish bosqichlari uchun boshlang‘ich tayanchni hujjatlashtiradi. Bu 3551 ning yangi testi emas; boshlang‘ich google/byt5-base uchun shu monitor o‘lchovi mavjud emas.

### 3.2 3551 → 10k-313: bir xil synthetic monitor, n=128

| Ko‘rsatkich | Oldingi model | Keyingi model |
|---|---:|---:|
| To‘liq moslik ↑ | 89.8438% | 90.6250% |
| CER ↓ | 0.1289% | 0.1117% |
| Mazmuniy WER ↓ | 0.2102% | 0.1402% |
| Imloviy CER ↓ | 0.0295% | 0.0197% |
| Punctuation macro F1 ↑ | 99.1873% | 94.0851% |
| Apostrof/tutuq F1 ↑ | 99.4048% | 99.7033% |
| Casing ↑ | 99.9497% | 99.9698% |
| Toza matnni o‘zgartirish ↓ | 0.0000% | 0.0000% |

10k bosqichida imloviy CER va mazmuniy WER taxminan uchdan birga kamaygan, to‘liq moslik oshgan va toza nazorat matnlari saqlangan. Format ko‘rsatkichlari ham to‘liq jadvalda berilgan; natija ushbu synthetic monitorga tegishli. Manba: `training/02_correction_10k/evidence/comparison.json`.

### 3.3 Real-392: tabiiy izohlarga moslashtirish

Real izohlar bosqichi 392/392 qadam va 1 epoch bilan yakunlangan; Colab logida `REAL_STAGE_COMPLETE_CLOUD_VERIFIED` belgisi saqlangan. 272 ta monitor (144 real + 128 synthetic) bajarilgan. Notebook W&B xulosasida real-apostrof F1 **77.0538% → 88.60759%** (after qiymati yaxlitlangan). Mahalliy checkpoint arxivida baseline bor, to‘liq final `comparison.json` esa mavjud emas; shu sababli bu qism saqlangan loglar va quyidagi alohida lokal baholashga tayanadi.

Quyidagi **CPU FP32, aynan bir xil development-60** taqqoslash real modelga o‘tishdagi farqni ko‘rsatadi; bu 313 → 392 ta’sirini alohida ajratmaydi, chunki oldingi model 3551:

| Ko‘rsatkich | Oldingi model | Keyingi model |
|---|---:|---:|
| To‘liq moslik ↑ | 11.6667% | 6.6667% |
| CER ↓ | 8.8449% | 7.5399% |
| Mazmuniy WER ↓ | 27.2059% | 22.4265% |
| Imloviy CER ↓ | 4.5708% | 4.3478% |
| Punctuation macro F1 ↑ | 13.8022% | 31.1159% |
| Apostrof/tutuq F1 ↑ | 72.7273% | 87.3239% |
| Casing ↑ | 95.1567% | 95.8974% |
| Toza matnni o‘zgartirish ↓ | 58.3333% | 100.0000% |

Real izohlarga moslashtirishdan keyin CER, WER, apostrof va punctuation ko‘rsatkichlarida yaxshilanish kuzatilgan. Toza matn formatini saqlash keyingi Quality 5k bosqichining alohida yo‘nalishi bo‘lgan. Manba: `reports/evidence/development_cpu/`. Ushbu development to‘plami keyingi pilotni baholashda ham ishlatilgan; yashirin test emas.

### 3.4 Real-392 → Quality5k-157: Colab BF16, development-60

| Ko‘rsatkich | Oldingi model | Keyingi model |
|---|---:|---:|
| To‘liq moslik ↑ | 6.6667% | 31.6667% |
| CER ↓ | 7.5882% | 5.1716% |
| Mazmuniy WER ↓ | 22.7941% | 22.4265% |
| Imloviy CER ↓ | 4.4036% | 3.8462% |
| Punctuation macro F1 ↑ | 31.1159% | 39.8214% |
| Apostrof/tutuq F1 ↑ | 87.3239% | 88.8889% |
| Casing ↑ | 95.8974% | 96.0684% |
| Toza matnni o‘zgartirish ↓ | 100.0000% | 0.0000% |

Quality 5k bosqichining asosiy yutug‘i — **toza matnni saqlash va to‘liq moslikning yaxshilanishi**. Shu development to‘plamida exact match 4/60 dan 19/60 ga, toza nazoratni saqlash 0/12 dan 12/12 ga yetgan. Imloviy CER va punctuation F1 ham yaxshilangan. CPU va BF16 natijalari alohida o‘lchov sifatida ko‘rsatilgan. Manba: `training/04_quality_5k/evidence/comparison.json`.

## 4. Yakuniy lokal baholash: input → quality5k-157

**Bu oldingi model → yangi model emas**, xato input va model javobi aynan bir xil target bilan taqqoslanadi. Apostrof gliflari tenglashtirilgan; barcha boshqa farqlar saqlanadi.

### 4.1 Ajratilgan sinov, n=62

| Ko‘rsatkich | Kiritilgan matn | Model javobi |
|---|---:|---:|
| To‘liq moslik ↑ | 20.9677% | 27.4194% |
| CER ↓ | 8.1333% | 4.6056% |
| Mazmuniy WER ↓ | 31.6901% | 22.8873% |
| Imloviy CER ↓ | 5.5302% | 4.0479% |
| Punctuation macro F1 ↑ | 43.6061% | 67.7511% |
| Apostrof/tutuq F1 ↑ | 64.4068% | 89.1892% |
| Casing ↑ | 95.2187% | 96.4431% |
| Toza matnni o‘zgartirish ↓ | 0.0000% | 0.0000% |

Etalonga mos bo‘lmagan matn: 49/62 → 45/62. 4 to‘liq tuzatildi, 15 qisman yaxshilandi, 1 yomonlashdi, 13 toza saqlandi. CER nisbiy kamayishi 43.4%. 62 ning 22 tasi real manbali qayta tekshirilgan izoh; 30 MCQ moslashuvi va 10 toza nashriyot misoli ham bor. Barchasi tabiiy izoh emas.

### 4.2 Avval ko‘rilgan UzLiB diagnostikasi, n=128

| Ko‘rsatkich | Kiritilgan matn | Model javobi |
|---|---:|---:|
| To‘liq moslik ↑ | 25.0000% | 35.1562% |
| CER ↓ | 4.9433% | 4.8526% |
| Mazmuniy WER ↓ | 26.9103% | 23.5880% |
| Imloviy CER ↓ | 3.5859% | 3.2323% |
| Punctuation macro F1 ↑ | 85.0000% | 81.5634% |
| Apostrof/tutuq F1 ↑ | 74.6667% | 82.9268% |
| Casing ↑ | 96.8459% | 96.9493% |
| Toza matnni o‘zgartirish ↓ | 0.0000% | 6.2500% |

Etalonga mos bo‘lmagan matn: 96/128 → 83/128. 32 toza nazoratdan 2 tasi o‘zgargan. Bu nashriyot savollaridan moslashtirilgan diagnostika, rasmiy UzLiB MCQ balli yoki umumiy tabiiy chat accuracy emas. Manba: `reports/evidence/final_cpu/REPORT.json`, qatorli xom javoblar mahalliy dalil arxivida, public repoda emas.

## 5. Amaliy qiymat va keyingi rivojlanish

**UzNormning asosiy natijasi — o‘zbekcha matnni tuzatish uchun lokal, ishlaydigan va natijalari hujjatlashtirilgan yechim.** Ketma-ket fine-tuning imlo, apostrof va matn formatini saqlashni rivojlantirgan. Ajratilgan 62 misolda CER nisbatan 43.4% kamaygan; Quality 5k development sinovida toza nazorat matnlarini saqlash yaxshilangan. Sodda interfeys va lokal hisoblash modelni qo‘lda matn tekshirish uchun qulay qiladi.

**Qo‘llanish doirasi:** sifat to‘plam va matn turiga bog‘liq; barcha ko‘rsatkichlar bir xil yo‘nalishda o‘zgarmagan (to‘liq jadvallar yuqorida). UzNorm tahrir taklifini beradi; yakuniy matn ma’nosi, nomlar va raqamlarni foydalanuvchi tekshiradi.

**Keyingi ilmiy bosqich:** mustaqil inson tekshirgan kattaroq Gold test va to‘rtta checkpointni bir xil mezonda taqqoslash. Hozirgi etalonlar Silver; corpus overlap tekshiruvi butun matn, 8-gram va manba identifikatorini qamraydi, pretraining tarkibini emas. 62 misollik test ko‘rilgani sababli navbatdagi tuning uchun yangi tegilmagan test ajratiladi. 313 uchun umumiy benchmark o‘lchovi hozircha mavjud emas.

## 6. Namoyish qilish tartibi

1. VS Code’da README va `training/README.md`: maqsad, 4 bosqich va model zanjiri.
2. Har bosqich `colab.ipynb`: trening kodi; public nusxada eski chiqishlar olib tashlangan. `training-history.json`: haqiqiy qadam/loss yozuvlari.
3. Ushbu conclusion report: avval qisqa yutuqlar, keyin bosqichlar bo‘yicha dalilli taqqoslash va rivojlanish rejasi.
4. `UzNorm: lokal sayt` F5: matn kiriting va faqat model taklifini ko‘rsating. Saytda baholash yo‘q.

Bu public nusxa umumlashtirilgan hisobotdir. Original dalillar va ularning SHA-256 manifesti mahalliy loyihada saqlanadi; xom izohlar bu repoga joylanmagan.
