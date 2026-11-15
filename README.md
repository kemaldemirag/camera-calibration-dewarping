# Tek kameralı saha kalibrasyonu

Bu ürün kökü, U01 için ilk uygulama dilimini yürütür. İlk dilim `U01-T1`
perspektif temel ölçümüdür: her 7 x 5 hedef görünümünün dört dış köşesiyle
bir düzlemsel dönüşüm kurulur, kalan 31 nokta ölçümden ayrılarak piksel
hatası hesaplanır.

## Kapsam

Bu sürüm aşağıdaki çıktıları üretir:

- görünüm başına kontrol noktası sayısı, RMS hata ve en büyük hata içeren
  `perspektif.csv`;
- girdi SHA-256 değerlerini, birleşik metrikleri, prosedürü ve iddia sınırını
  içeren `ozet.json`.

`TEKRARLANDI` durumu yalnız `U01-T1` sayısal tekrarını ifade eder. Lens
bozulması, milimetre doğruluğu, nesne takibi ve müşteriye özgü tolerans
uyumu bu sonuçla kabul edilmiş sayılmaz. Bunlar sırasıyla `U01-T2` ve
`U01-T3` için ayrı veri ve önceden belirlenmiş eşik gerektirir.

## Çalıştırma

Python 3.12 ve NumPy 2.3.5 ile:

```powershell
$env:PYTHONPATH = "src"
python -m camera_calibration_dewarping.cli `
  "..\..\..\003-Proje Kayıtları\deneyler\15.11.2026\source-package\veri" `
  --output-dir "artifacts\u01-t1"
```

Girdi dizini `chess_corners1.dat` ile `chess_corners7.dat` arasındaki sabit
OpenCV kayıtlarını içermelidir. Her dosya 7 satır, 5 sütun ve 70 sonlu sayı
taşımalıdır.

## Testler

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Golden test, çalışma alanındaki doğrulanmış K03 girdilerini kullanır. Girdiler
henüz indirilmemişse yalnız bu test atlanır; bir teslim doğrulamasında atlanan
golden test kabul edilmez.

