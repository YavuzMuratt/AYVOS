# Assignment 5: Kendi Gaussian Blur Implementasyonum

## Görev Açıklaması
Bu assignment'te Gaussian kernel'ı elle oluşturarak bir görüntü üzerinde filtre uygulamak amaçlandı:
- Gaussian kernel'ını matematiksel formülle oluşturma
- OpenCV ile hız ve kalite karşılaştırması yapma
- Farklı kernel boyutları ve sigma değerleri ile denemeler

## Öğrenme Hedefleri
1. Gaussian dağılımının matematiksel formülünü anlama
2. Kernel oluşturma algoritmasını yazma
3. Filtreleme işlemini manuel olarak gerçekleştirme
4. OpenCV ile performans karşılaştırması
5. Hız ve kalite analizi

## Dosya Yapısı
```
Assignment5/
├── README.md                                    # Bu dosya
├── custom_gaussian_blur.py                     # Kendi Gaussian Blur implementasyonu
├── visualize_gaussian_filters_3d.py            # 3D Gaussian filter görselleştirme
├── requirements.txt                             # Gerekli kütüphaneler
├── images/                                     # Test görüntüleri
└── results/                                    # Çıktı görüntüleri
    ├── custom_implementations/
    ├── comparisons/
    ├── performance_analysis/
    └── 3d_visualization/                       # 3D görselleştirme sonuçları
```

## Kullanılan Kütüphaneler
- OpenCV (cv2) - Karşılaştırma için
- NumPy - Matematiksel hesaplamalar
- Matplotlib - Görselleştirme
- Time - Performans ölçümü

## Çalıştırma
```bash
pip install -r requirements.txt
python custom_gaussian_blur.py
python visualize_gaussian_filters_3d.py
```

## Öğrenilen Kavramlar
1. **Gaussian Kernel Oluşturma**: Matematiksel formül uygulama
2. **Manuel Filtreleme**: Kernel ile konvolüsyon işlemi
3. **Separebilite (Separability)**: 2D kernel'ı 1D kernel'lara ayırma
4. **Otomatik Sigma Hesaplama**: Kernel boyutuna göre sigma belirleme
5. **Performans Analizi**: Hız ve kalite karşılaştırması
6. **Optimizasyon**: Algoritma iyileştirme teknikleri
7. **Doğrulama**: OpenCV sonuçları ile karşılaştırma
8. **3D Görselleştirme**: Gaussian filtrelerinin 3 boyutlu analizi

## Gaussian Kernel Formülü
```
G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
```
- **σ (sigma)**: Standart sapma
- **x,y**: Kernel koordinatları
- **Normalizasyon**: Toplam ağırlık 1 olacak şekilde

## Separebilite (Separability)
```
G(x,y) = G(x) * G(y) = (1/(√2πσ)) * e^(-x²/(2σ²)) * (1/(√2πσ)) * e^(-y²/(2σ²))
```
- **Avantaj**: 2D kernel yerine 2 adet 1D kernel kullanma
- **Hız**: O(n²) yerine O(2n) karmaşıklık
- **Bellek**: Daha az bellek kullanımı

## Otomatik Sigma Hesaplama
```
σ = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
```
- **OpenCV Formülü**: Kernel boyutuna göre optimal sigma
- **Kullanım**: Manuel sigma belirtmek yerine otomatik hesaplama

## 3D Görselleştirme Özellikleri

### `visualize_gaussian_filters_3d.py` Script'i:
1. **Tek Gaussian Filtreleri**: Farklı kernel boyutları ve sigma değerleri için 3D görselleştirme
2. **Separable Gaussian**: 1D kernel'ların 3D gösterimi ve 2D kernel oluşumu
3. **Sigma Etkisi**: Farklı sigma değerlerinin Gaussian şekline etkisi
4. **Kernel Boyutu Etkisi**: Farklı kernel boyutlarının görsel analizi
5. **Animasyonlu Oluşum**: Gaussian filtresinin gelişim aşamaları
6. **Karşılaştırmalı Analiz**: 2D heatmap ve 3D surface plot'lar

### Çıktı Dosyaları:
- `gaussian_3x3_sigma_0.5.jpg` - Küçük kernel, düşük sigma
- `gaussian_5x5_sigma_1.0.jpg` - Orta kernel, orta sigma
- `gaussian_7x7_sigma_1.5.jpg` - Büyük kernel, yüksek sigma
- `gaussian_9x9_sigma_2.0.jpg` - Çok büyük kernel, çok yüksek sigma
- `gaussian_11x11_sigma_2.5.jpg` - Ekstra büyük kernel, ekstra yüksek sigma
- `separable_gaussian_3d.jpg` - Separable Gaussian analizi
- `sigma_effect_3d.jpg` - Sigma etkisi karşılaştırması
- `kernel_size_effect_3d.jpg` - Kernel boyutu etkisi
- `animated_gaussian_formation.jpg` - Animasyonlu oluşum
- `gaussian_kernels_comparison_2d.jpg` - 2D karşılaştırma

## Sonuçlar
Bu assignment sonunda Gaussian Blur'ın nasıl çalıştığını derinlemesine anlamak hedeflenmiştir.
