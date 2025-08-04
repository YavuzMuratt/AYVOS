# Assignment 7: Keypoint ve Özellik Çıkarımı

## 📋 Görev Açıklaması
SIFT, ORB gibi algoritmalar ile görüntüdeki anlamlı noktaların tespiti, eşleştirme ve uygulamaları.

## 🎯 Öğrenme Hedefleri
- **Keypoint Detection**: Görüntüdeki önemli noktaların tespiti
- **Feature Extraction**: Bu noktalardan özellik vektörlerinin çıkarılması
- **Feature Matching**: Farklı görüntüler arasında özellik eşleştirme
- **Homography Estimation**: Perspektif dönüşüm hesaplama
- **Real-world Applications**: Panorama, nesne tespiti, görüntü birleştirme

## 📁 Dosya Yapısı
```
Assignment7/
├── README.md
├── requirements.txt
├── keypoint_feature_extraction.py
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── sample_images/
└── results/
    ├── keypoints/
    ├── features/
    ├── matching/
    └── applications/
```

## 📚 Kullanılan Kütüphaneler
- **OpenCV (cv2)**: Keypoint detection, feature extraction, matching
- **NumPy**: Matematiksel işlemler
- **Matplotlib**: Görselleştirme
- **Pillow**: Görüntü işleme

## 🚀 Çalıştırma
```bash
cd Assignment7
pip install -r requirements.txt
python keypoint_feature_extraction.py
```

## 📖 Öğrenilen Kavramlar

### 1. Keypoint Detection (Anahtar Nokta Tespiti)
- **Harris Corner Detection**: Köşe noktalarının tespiti
- **SIFT (Scale-Invariant Feature Transform)**: Ölçek değişmez özellik dönüşümü
- **SURF (Speeded Up Robust Features)**: Hızlandırılmış sağlam özellikler
- **ORB (Oriented FAST and Rotated BRIEF)**: Hızlı ve döndürülmüş özellikler
- **FAST (Features from Accelerated Segment Test)**: Hızlandırılmış segment testi

### 2. Feature Extraction (Özellik Çıkarımı)
- **Descriptor Calculation**: Özellik vektörlerinin hesaplanması
- **Binary Descriptors**: ORB, BRIEF
- **Floating-point Descriptors**: SIFT, SURF
- **Descriptor Matching**: Brute Force, FLANN

### 3. Feature Matching (Özellik Eşleştirme)
- **Brute Force Matching**: Tüm olası eşleştirmeleri deneme
- **FLANN Matching**: Hızlı yakın komşu arama
- **Ratio Test**: Lowe'un oran testi
- **RANSAC**: Sağlam eşleştirmeleri filtreleme

### 4. Homography and Applications
- **Homography Matrix**: Perspektif dönüşüm matrisi
- **Image Stitching**: Görüntü birleştirme
- **Object Detection**: Nesne tespiti
- **Image Registration**: Görüntü kayıt

## 🔬 Matematiksel Temeller

### SIFT Algoritması
```
1. Scale-space Extrema Detection
   - Gaussian Pyramid
   - Difference of Gaussians (DoG)
   
2. Keypoint Localization
   - Taylor expansion
   - Hessian matrix
   
3. Orientation Assignment
   - Gradient magnitude
   - Orientation histogram
   
4. Descriptor Generation
   - 16x16 patches
   - 128-dimensional vector
```

### ORB Algoritması
```
1. FAST Keypoint Detection
   - 9-point circle test
   - Non-maximum suppression
   
2. Orientation Assignment
   - Intensity centroid
   - Orientation vector
   
3. BRIEF Descriptor
   - Binary test patterns
   - 256-bit descriptor
```

## 🎯 Gerçek Dünya Uygulamaları
- **Panorama Oluşturma**: Birden fazla görüntüyü birleştirme
- **Nesne Tespiti**: Belirli nesneleri görüntüde bulma
- **Görüntü Kayıt**: Farklı açılardan çekilmiş görüntüleri hizalama
- **Augmented Reality**: Gerçek dünya üzerine sanal nesneler ekleme
- **Image Stitching**: Geniş açılı görüntüler oluşturma

## 📊 Sonuçlar
Bu assignment sonunda keypoint detection, feature extraction ve matching algoritmalarını derinlemesine anlayacak ve gerçek dünya problemlerine uygulayabilecek kapasiteye gelmek amaçlanmıştır. 