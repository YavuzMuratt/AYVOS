# Assignment 11: YOLOv8 ile İnsan Tespiti ve Sayımı

## Görev Açıklaması
Bu assignment'te YOLOv8 modeli kullanarak insan tespiti ve sayımı yapılacak:
- YOLOv8 modelinin kurulumu ve konfigürasyonu
- Video ve görüntü dosyalarından insan tespiti
- Gerçek zamanlı insan sayımı
- Şık ve kullanıcı dostu arayüz
- Tespit sonuçlarının görselleştirilmesi

## Öğrenme Hedefleri
1. YOLOv8 modelinin kurulumu ve kullanımı
2. Gerçek zamanlı nesne tespiti
3. Video işleme ve analizi
4. Kullanıcı arayüzü tasarımı
5. İnsan sayımı algoritması
6. Performans optimizasyonu

## Dosya Yapısı
```
Assignment11/
├── README.md                                    # Bu dosya
├── human_detection_app.py                       # Ana insan tespiti uygulaması
├── requirements.txt                              # Gerekli kütüphaneler
├── model/                                       # Eğitilmiş model klasörü
│   └── best.pt                                 # Kendi eğitilmiş YOLOv8 modeli
├── videos/                                      # Test videoları
├── images/                                      # Test görüntüleri
└── results/                                     # Çıktı dosyaları
    ├── detections/
    ├── counts/
    └── analysis/
```

## Kullanılan Kütüphaneler
- **ultralytics** - YOLOv8 modeli
- **OpenCV (cv2)** - Video ve görüntü işleme
- **NumPy** - Matematiksel hesaplamalar
- **Matplotlib** - Görselleştirme
- **Pillow** - Görüntü işleme
- **tkinter** - Kullanıcı arayüzü

## Çalıştırma
```bash
pip install -r requirements.txt
python human_detection_app.py
```

## Özellikler
1. **Gerçek Zamanlı Tespit**: Video akışından canlı insan tespiti
2. **Dosya İşleme**: Video ve görüntü dosyalarından tespit
3. **İnsan Sayımı**: Tespit edilen insanların sayısını tutma
4. **Şık Arayüz**: Kullanıcı dostu grafik arayüz
5. **Sonuç Kaydetme**: Tespit sonuçlarını kaydetme
6. **Performans Analizi**: FPS ve doğruluk analizi

## YOLOv8 Modeli
- **Öncelik**: Kendi eğitilmiş model (`model/best.pt`)
- **Yedek**: YOLOv8n (nano) - Eğer özel model yoksa
- **Sınıf**: Person (insan) tespiti
- **Güven Eşiği**: 0.5 (ayarlanabilir)
- **NMS**: Non-Maximum Suppression aktif

### Model Kullanımı
Uygulama önce `model/best.pt` dosyasını arar. Eğer bu dosya mevcutsa, kendi eğitilmiş modelinizi kullanır. Aksi takdirde varsayılan YOLOv8n modelini indirir ve kullanır.

## Çıktı Dosyaları
- `human_count_{timestamp}.txt` - İnsan sayımı sonuçları
- `detection_analysis_{timestamp}.json` - Detaylı analiz verileri
- `visualization_{timestamp}.png` - Ana görselleştirme grafikleri
- `detailed_analysis_{timestamp}.png` - Detaylı analiz grafikleri

### Grafik Türleri:
1. **Zaman Serisi**: İnsan sayısı değişimi
2. **Histogram**: İnsan sayısı dağılımı
3. **Box Plot**: İstatistiksel özet
4. **Kümülatif Grafik**: Toplam benzersiz insan
5. **Yoğunluk Analizi**: İnsan yoğunluğu
6. **Değişim Hızı**: Hareket analizi
7. **İstatistiksel Özet**: Maksimum, minimum, ortalama
8. **Zaman Dilimi**: Bölüm bazlı analiz

## Sonuçlar
Bu assignment sonunda YOLOv8 modelini kullanarak insan tespiti yapabilecek, gerçek zamanlı sayım gerçekleştirebilecek bir arayüz tasarlamak hedeflendi. 
