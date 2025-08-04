# Assignment 3: OpenCV Temelleri

## Görev Açıklaması
Bu assignment'te OpenCV'nin temel görüntü işleme fonksiyonlarını öğreniyoruz:
- `cv2.imread()` - Görüntü okuma
- `cv2.imshow()` - Görüntü gösterme  
- `cv2.resize()` - Görüntü boyutlandırma

## Öğrenme Hedefleri
1. Farklı görüntü formatlarını okuma
2. Görüntüleri ekranda gösterme ve pencere yönetimi
3. Görüntü boyutlandırma teknikleri
4. Interpolasyon yöntemlerinin farklarını anlama

## Dosya Yapısı
```
Assignment3/
├── README.md                    # Bu dosya
├── opencv_basics.py            # Ana demo kodu
├── requirements.txt             # Gerekli kütüphaneler
├── images/                     # Test görüntüleri
│   ├── sample.jpg
│   └── sample.png
└── results/                    # Çıktı görüntüleri
    ├── resized_images/
    └── processed_images/
```

## Kullanılan Kütüphaneler
- OpenCV (cv2) - Görüntü işleme
- NumPy - Dizi işlemleri
- Matplotlib - Görselleştirme

## Çalıştırma
```bash
pip install -r requirements.txt
python opencv_basics.py
```

## Öğrenilen Kavramlar
1. **Görüntü Okuma**: Farklı formatlar ve renk uzayları
2. **Görüntü Gösterme**: Pencere yönetimi ve kullanıcı etkileşimi
3. **Boyutlandırma**: Interpolasyon yöntemleri ve kalite farkları
4. **Piksel İşlemleri**: Görüntü matrisi manipülasyonu
5. **Renk Kanalları**: BGR formatında kanal ayrıştırma, birleştirme ve renk efektleri

## Sonuçlar
Bu assignment sonunda OpenCV'nin temel fonksiyonlarını kullanarak görüntü işleme yapabilir hale gelmek hedeflenmiştir. 