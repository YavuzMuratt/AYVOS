# Assignment 6: Geleneksel Görüntü İşleme Yöntemleri

## Görev Açıklaması
Bu assignment'te klasik görüntü işleme yöntemlerini öğrenme ve uygulama:
- **Thresholding**: Farklı thresholding teknikleri ve uygulamaları
- **Kenar Bulma**: Canny, Sobel, Laplacian algoritmaları
- **Morfolojik İşlemler**: Erode, Dilate, Opening, Closing

## Öğrenme Hedefleri
1. **Thresholding Teknikleri**: Binary, Adaptive, Otsu thresholding
2. **Kenar Bulma Algoritmaları**: Canny, Sobel, Laplacian, Prewitt
3. **Morfolojik İşlemler**: Erode, Dilate, Opening, Closing, Gradient
4. **Kombinasyon Teknikleri**: Birden fazla yöntemi birleştirme

## Dosya Yapısı
```
Assignment6/
├── README.md                    # Bu dosya
├── traditional_image_processing.py  # Ana demo kodu
├── requirements.txt             # Gerekli kütüphaneler
├── images/                     # Test görüntüleri
└── results/                    # Çıktı dosyaları
    ├── thresholding/
    ├── edge_detection/
    ├── morphological/
    ├── combinations/
    └── real_world_applications/
```

## Kullanılan Kütüphaneler
- **OpenCV (cv2)**: Ana görüntü işleme kütüphanesi
- **NumPy**: Matematiksel işlemler
- **Matplotlib**: Görselleştirme
- **SciPy**: Bilimsel hesaplamalar
- **Pillow**: Görüntü işleme

## Çalıştırma
```bash
pip install -r requirements.txt
python traditional_image_processing.py
```

## Öğrenilen Kavramlar

### 1. Thresholding (Eşikleme)
- **Binary Thresholding**: Sabit eşik değeri ile ikili görüntü oluşturma
- **Adaptive Thresholding**: Yerel bölgeye göre dinamik eşikleme
- **Otsu Thresholding**: Otomatik optimal eşik değeri belirleme
- **Multi-level Thresholding**: Çoklu eşik değerleri

### 2. Kenar Bulma (Edge Detection)
- **Canny Edge Detection**: En popüler kenar bulma algoritması
- **Sobel Operator**: Gradient tabanlı kenar bulma
- **Laplacian Operator**: İkinci türev tabanlı kenar bulma
- **Prewitt Operator**: Basit gradient operatörü

### 3. Morfolojik İşlemler
- **Erosion (Aşınma)**: Nesneleri küçültme, gürültüyü azaltma
- **Dilation (Genişleme)**: Nesneleri büyütme, boşlukları doldurma
- **Opening**: Erosion + Dilation (gürültü temizleme)
- **Closing**: Dilation + Erosion (boşluk doldurma)
- **Gradient**: Kenar çıkarma

### 4. Kombinasyon Teknikleri
- **Thresholding + Morphology**: İkili görüntü iyileştirme
- **Edge Detection + Morphology**: Kenar temizleme
- **Multi-stage Processing**: Çok aşamalı işleme

### 5. Gerçek Dünya Uygulamaları
- **Belge İşleme**: OCR öncesi görüntü temizleme
- **Medikal Görüntüleme**: Tümör tespiti, kemik analizi
- **Endüstriyel Kontrol**: Kalite kontrol, nesne sayma
- **Güvenlik Sistemleri**: Hareket tespiti, nesne takibi

## Matematiksel Temeller

### Thresholding
```
Binary: dst(x,y) = maxval if src(x,y) > thresh else 0
Adaptive: dst(x,y) = maxval if src(x,y) > mean(neighborhood) - C else 0
```

### Kenar Bulma
```
Sobel: G = √(Gx² + Gy²) where Gx = [-1 0 1; -2 0 2; -1 0 1]
Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y²
Canny: 1. Gaussian Blur → 2. Gradient → 3. Non-maximum suppression → 4. Hysteresis
```

### Morfolojik İşlemler
```
Erosion: A ⊖ B = {z | Bz ⊆ A}
Dilation: A ⊕ B = {z | Bz ∩ A ≠ ∅}
Opening: A ○ B = (A ⊖ B) ⊕ B
Closing: A ● B = (A ⊕ B) ⊖ B
```

## Sonuçlar
Bu assignment sonunda geleneksel görüntü işleme yöntemlerini derinlemesine anlayacak ve gerçek dünya problemlerine uygulayabilecek kapasiteye gelmek amaçlanmıştır. Modern deep learning yöntemlerinin temelini oluşturan bu klasik teknikler, görüntü işleme alanında hala kritik öneme sahiptir. 