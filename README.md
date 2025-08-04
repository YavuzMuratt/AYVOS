# AYVOS - Computer Vision Projects

Bu repository, bilgisayarlı görü ve derin öğrenme projelerini içermektedir. Her assignment farklı bir konuya odaklanmıştır.

## 📋 İçindekiler

- [Assignment 3: OpenCV Temelleri](#assignment-3-opencv-temelleri)
- [Assignment 4: Gaussian Blur](#assignment-4-gaussian-blur)
- [Assignment 5: Özel Gaussian Blur Implementasyonu](#assignment-5-özel-gaussian-blur-implementasyonu)
- [Assignment 6: Geleneksel Görüntü İşleme](#assignment-6-geleneksel-görüntü-işleme)
- [Assignment 7: Keypoint ve Feature Extraction](#assignment-7-keypoint-ve-feature-extraction)
- [Assignment 11: YOLOv8 İnsan Tespiti](#assignment-11-yolov8-insan-tespiti)

## 🚀 Kurulum

### Gereksinimler
- Python 3.8+
- pip (Python paket yöneticisi)

### Ortam Kurulumu
```bash
# Repository'yi klonlayın
git clone https://github.com/KULLANICI_ADINIZ/REPOSITORY_ADINIZ.git
cd AYVOS

# Her assignment için ayrı sanal ortam oluşturmanız önerilir
python -m venv venv
# Windows için:
venv\Scripts\activate
# Linux/Mac için:
source venv/bin/activate
```

---

## Assignment 3: OpenCV Temelleri

**Konu:** OpenCV kütüphanesi ile temel görüntü işleme işlemleri

### Kurulum
```bash
cd Assignment3
pip install -r requirements.txt
```

### Çalıştırma
```bash
python opencv_basics.py
```

### Ne Yapar?
- Görüntü okuma ve yazma
- Renk kanallarını ayırma (BGR, RGB)
- Görüntü boyutlandırma ve yeniden boyutlandırma
- Farklı interpolasyon yöntemleri
- Piksel manipülasyonu

### Sonuçlar
Sonuçlar `results/` klasöründe saklanır:
- `processed_images/`: İşlenmiş görüntüler
- `resized_images/`: Boyutlandırılmış görüntüler

---

## Assignment 4: Gaussian Blur

**Konu:** Gaussian blur filtreleri ve farklı parametrelerin etkisi

### Kurulum
```bash
cd Assignment4
pip install -r requirements.txt
```

### Çalıştırma
```bash
python gaussian_blur_demo.py
```

### Ne Yapar?
- Farklı kernel boyutları ile Gaussian blur
- Sigma parametresinin etkisi
- Kernel görselleştirme
- Diğer blur yöntemleri ile karşılaştırma

### Parametreler
- **Kernel Boyutları:** 3x3, 5x5, 7x7, 9x9, 15x15, 21x21
- **Sigma Değerleri:** 0.5, 1.0, 1.5, 2.0, 3.0, 5.0

### Sonuçlar
- `blur_effects/`: Farklı parametrelerle bulanıklaştırılmış görüntüler
- `comparisons/`: Karşılaştırma görüntüleri
- `kernel_visualizations/`: Kernel görselleştirmeleri

---

## Assignment 5: Özel Gaussian Blur Implementasyonu

**Konu:** Gaussian blur'un manuel implementasyonu ve optimizasyon

### Kurulum
```bash
cd Assignment5
pip install -r requirements.txt
```

### Çalıştırma
```bash
# Ana uygulama
python custom_gaussian_blur.py

# 3D görselleştirme (isteğe bağlı)
python visualize_gaussian_filters_3d.py
```

### Ne Yapar?
- Gaussian kernel'in manuel oluşturulması
- 2D ve separable implementasyonlar
- Performans karşılaştırması
- 3D kernel görselleştirme

### Özellikler
- Manuel 2D Gaussian blur
- Separable Gaussian blur
- Otomatik sigma hesaplama
- Performans analizi

### Sonuçlar
- `custom_implementations/`: Manuel implementasyon sonuçları
- `comparisons/`: OpenCV ile karşılaştırmalar
- `3d_visualization/`: 3D kernel görselleştirmeleri
- `performance_analysis/`: Performans grafikleri

---

## Assignment 6: Geleneksel Görüntü İşleme

**Konu:** Kenar tespiti, morfolojik işlemler ve eşikleme

### Kurulum
```bash
cd Assignment6
pip install -r requirements.txt
```

### Çalıştırma
```bash
python traditional_image_processing.py
```

### Ne Yapar?
- **Kenar Tespiti:** Sobel, Prewitt, Laplacian, Canny
- **Morfolojik İşlemler:** Erosion, dilation, opening, closing
- **Eşikleme:** Binary, Otsu, adaptive thresholding
- **Kombinasyonlar:** Farklı işlemlerin birleştirilmesi

### Sonuçlar
- `edge_detection/`: Kenar tespiti sonuçları
- `morphological/`: Morfolojik işlem sonuçları
- `thresholding/`: Eşikleme sonuçları
- `combinations/`: Kombinasyon sonuçları

---

## Assignment 7: Keypoint ve Feature Extraction

**Konu:** Özellik noktaları tespiti ve eşleştirme

### Kurulum
```bash
cd Assignment7
pip install -r requirements.txt
```

### Çalıştırma
```bash
python keypoint_feature_extraction.py
```

### Ne Yapar?
- **Keypoint Tespiti:** SIFT, ORB, FAST, Harris
- **Feature Matching:** FLANN, brute force matching
- **Uygulamalar:** Homography, object detection

### Algoritmalar
- **SIFT:** Scale-Invariant Feature Transform
- **ORB:** Oriented FAST and Rotated BRIEF
- **FAST:** Features from Accelerated Segment Test
- **Harris:** Harris corner detection

### Sonuçlar
- `keypoints/`: Keypoint tespiti sonuçları
- `matching/`: Feature matching sonuçları
- `applications/`: Uygulama sonuçları

---

## Assignment 11: YOLOv8 İnsan Tespiti

**Konu:** YOLOv8 ile gerçek zamanlı insan tespiti ve sayımı

### Kurulum
```bash
cd Assignment11
pip install -r requirements.txt
```

### Model Kurulumu
```bash
# YOLOv8 modelini indirin (eğer yoksa)
# Model dosyası model/best.pt konumunda olmalı
```

### Çalıştırma
```bash
python human_detection_app.py
```

### Ne Yapar?
- **Video İşleme:** MP4, AVI dosyaları
- **Görüntü İşleme:** JPG, PNG dosyaları
- **Gerçek Zamanlı:** Kamera akışı
- **İnsan Sayımı:** Tracking ile benzersiz insan sayısı
- **Analiz:** Detaylı istatistikler ve görselleştirmeler

### Özellikler
- Şık GUI arayüzü
- Çoklu dosya formatı desteği
- Gerçek zamanlı tracking
- Sonuç kaydetme ve analiz
- Performans metrikleri

### Kullanım
1. Uygulamayı başlatın
2. "Video Seç" veya "Görüntü Seç" butonuna tıklayın
3. Dosyayı seçin
4. "Tespiti Başlat" butonuna tıklayın
5. Sonuçları "Sonuçları Kaydet" ile kaydedin

### Sonuçlar
- `results/counts/`: İnsan sayımı sonuçları
- `results/analysis/`: Detaylı analiz raporları
- `results/analysis/detection_analysis_*.json`: JSON formatında analiz
- `results/analysis/visualization_*.png`: Görselleştirmeler

---

## 📁 Proje Yapısı

```
AYVOS/
├── Assignment3/          # OpenCV Temelleri
├── Assignment4/          # Gaussian Blur
├── Assignment5/          # Özel Gaussian Blur
├── Assignment6/          # Geleneksel Görüntü İşleme
├── Assignment7/          # Keypoint ve Feature Extraction
├── Assignment11/         # YOLOv8 İnsan Tespiti
└── README.md            # Bu dosya
```


---

**Not:** Her assignment'ı çalıştırmadan önce ilgili klasöre gidip `pip install -r requirements.txt` komutunu çalıştırmayı unutmayın. 
