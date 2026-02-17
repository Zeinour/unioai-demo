# UnioAI Demo (Streamlit)

Bu repo, **UnioAI: Bankacılıkta Akıllı Borç Yönetimi ve Tahsilat Analitiği** sunumunuz için hazırlanmış, GitHub'a koyabileceğiniz **çalışır demo** örneğidir.

## Özellikler
- Sentetik veri üretimi **veya** CSV yükleme
- AI tabanlı **tahsilat skoru** (RandomForest / LogisticRegression)
- Skora göre **dinamik önceliklendirme** + kanal/zaman önerisi + mesaj şablonu
- Basit **simülasyon** (eşik & kapasite etkisi)
- Demo amaçlı **analitik asistan** (dış API yok)
- Power BI/Excel için **CSV dışa aktarım**

## Hızlı Başlangıç

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## CSV Şeması
Beklenen sütunlar:

- `musteri_id` (int)
- `yas` (int)
- `gelir_duzeyi` (Dusuk/Orta/Yuksek)
- `gecikme_sayisi` (int)
- `toplam_borc` (float)
- `gecikme_gunu` (int)
- `son_odeme_uzerinden_gun` (int)
- `kanal_tercihi` (SMS/Arama/E-posta)
- `geri_odeme` (0/1)

## Demo Notu
Uygulama **simülasyon** amaçlıdır. Gerçek banka verisi içermez.
