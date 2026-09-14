# Modelni alohida o‘rnatish

Model vaznlari GitHub repoga kiritilmagan. Bu loyiha aynan **Quality5k-157 FINAL** arxivini yoki undan oldin tekshirib import qilingan modelni ishlatadi. `google/byt5-base`ni o‘zi yuklash fine-tuned UzNorm modelini bermaydi.

Loyiha egasidan tegishli model arxiviga ruxsat oling; shaxsiy Google Drive havolalari public repoda tarqatilmaydi. Arxiv mavjud bo‘lgach:

```powershell
.\run.cmd import-model "D:\Models\quality5k-final.zip"
.\run.cmd verify-model
.\run.cmd web
```

Importer barcha arxiv fayllarini hash orqali tekshiradi, kerakli inference fayllarini `models/quality5k-157/`ga chiqaradi. Optimizer/pickle fayllarini inference uchun yuklamaydi. Eski mavjud modelni bosib yozmaydi. Tekshiruvni aylanib o‘tish uchun `model-policy.json` hashlarini o‘zgartirmang.

Oldindan import qilingan `quality5k-157` papkasini alohida ko‘chirish ham mumkin: `LOCAL_MODEL.json` va barcha inference fayllari birga bo‘lishi kerak. Keyin `verify-model` bajaring.

Git bu papkadagi vaznlarni e’tiborsiz qoldiradi. `git add -f` orqali ularni majburan qo‘shmang.
