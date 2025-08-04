# Staj Görevleri Yol Haritası

## Genel Yaklaşım

Bu doküman, staj görevlerinin nasıl sistematik ve profesyonel bir şekilde yapıldığını açıklar. Her görev için aynı yaklaşım kullanılır.

## 📋 Görev Yapma Stratejisi

### 1. Görev Analizi
- Görev dokümanını detaylıca oku
- Hangi kavramların öğrenilmesi gerektiğini belirle
- Seviyeyi değerlendir (Başlangıç/Orta/İleri)

### 2. Klasör Yapısı Oluşturma
```
Assignment{N}/
├── README.md                    # Görev açıklaması ve öğrenme hedefleri
├── {görev_adı}.py             # Ana demo kodu
├── requirements.txt             # Gerekli kütüphaneler
├── images/                     # Test görüntüleri
└── results/                    # Çıktı dosyaları
    ├── resized_images/
    └── processed_images/
```

### 3. Kod Yazma Stratejisi
- **Sınıf Tabanlı Yaklaşım**: Her görev için ayrı demo sınıfı
- **Detaylı Açıklamalar**: Her fonksiyon için docstring
- **Görsel Çıktılar**: Sonuçları kaydetme ve görselleştirme
- **Kullanıcı Etkileşimi**: Interaktif demo'lar

### 4. Dokümantasyon
- README.md ile görev açıklaması
- Öğrenilen kavramların listesi
- Çalıştırma talimatları
- Sonuçların açıklaması

## 🎯 Tamamlanan Görevler

### Assignment 3: OpenCV Temelleri ✅
**Görev**: cv2.imread, cv2.imshow, cv2.resize fonksiyonlarını öğrenme

**Yapılan İşler**:
1. **Görüntü Okuma Demo'su**
   - Farklı okuma modları (BGR, Gri, Orijinal)
   - Hata yönetimi
   - Görüntü bilgilerini yazdırma

2. **Görüntü Gösterme Demo'su**
   - Pencere yönetimi
   - Kullanıcı etkileşimi ('q' ile kapatma, 's' ile kaydetme)
   - Çoklu pencere gösterimi

3. **Boyutlandırma Demo'su**
   - Farklı boyutlandırma örnekleri
   - Interpolasyon yöntemleri karşılaştırması
   - Temiz çıktı klasörü (her interpolasyon için ayrı dosya değil)

4. **Piksel İşlemleri Demo'su**
   - Piksel değerlerini inceleme
   - Piksel değiştirme
   - Renk kanalları ayrıştırma
   - **8 farklı renk kanalı kombinasyonu**:
     - blue_green_only.jpg
     - blue_red_only.jpg
     - green_red_only.jpg
     - blue_only.jpg
     - green_only.jpg
     - red_only.jpg
     - bgr_to_rgb.jpg
     - enhanced_blue.jpg

**Öğrenilen Kavramlar**:
- BGR renk uzayı ve kanal yapısı
- Interpolasyon yöntemlerinin farkları
- Piksel seviyesinde görüntü manipülasyonu
- Renk kanallarının matematiksel işlemleri

### Assignment 4: Gaussian Blur ✅
**Görev**: cv2.GaussianBlur() fonksiyonunu öğrenme

**Yapılan İşler**:
1. **Gaussian Blur Teorisi**
   - Matematiksel formül açıklaması
   - Kernel ve sigma parametreleri
   - Gaussian dağılımı teorisi

2. **Kernel Görselleştirme**
   - Farklı kernel boyutları (3x3, 5x5, 7x7, 9x9)
   - Farklı sigma değerleri (1.0, 1.5, 2.0)
   - Kernel değerlerinin görselleştirilmesi

3. **Temel Gaussian Blur İşlemleri**
   - Farklı kernel boyutları ile denemeler
   - Otomatik sigma hesaplama
   - Görüntü bulanıklaştırma örnekleri

4. **Sigma Etkisi**
   - Sabit kernel boyutu, farklı sigma değerleri
   - Bulanıklaştırma yoğunluğunun analizi

5. **Parametre Kombinasyonları**
   - Hafif, orta, güçlü, çok güçlü bulanıklaştırma
   - Aşırı ve maksimum bulanıklaştırma

6. **Kenar Koruma Özelliği**
   - Gaussian vs Ortalama filtre karşılaştırması
   - Gaussian vs Median filtre karşılaştırması
   - Kenar detaylarının korunması analizi

**Öğrenilen Kavramlar**:
- Gaussian dağılımı ve matematiksel formülü
- Kernel boyutu ve sigma parametrelerinin etkisi
- Otomatik sigma hesaplama formülü
- Kenar koruma özelliği
- Farklı filtre türlerinin karşılaştırması

### Assignment 5: Kendi Gaussian Blur Implementasyonu ✅
**Görev**: Gaussian kernel'ı elle oluşturarak filtre uygulama

**Yapılan İşler**:
1. **Gaussian Kernel Oluşturma**
   - Matematiksel formül uygulama
   - Farklı parametrelerle kernel testleri
   - Kernel görselleştirme

2. **Manuel Gaussian Blur**
   - 2D kernel ile manuel filtreleme
   - Farklı kernel boyutları ve sigma değerleri
   - İşlem süresi ölçümü

3. **Verimli Implementasyonlar**
   - **Separebilite (Separability)**: 2D kernel'ı 1D kernel'lara ayırma
   - **Otomatik Sigma Hesaplama**: OpenCV formülü ile optimal sigma
   - Manuel separebilite implementasyonu
   - OpenCV separebilite implementasyonu

4. **Performans Karşılaştırması**
   - Manuel 2D vs Manuel Separebilite
   - Manuel vs OpenCV Separebilite
   - OpenCV Separebilite vs OpenCV GaussianBlur
   - Hız ve kalite analizi

5. **Otomatik Sigma Hesaplama**
   - Kernel boyutuna göre sigma hesaplama
   - Farklı kernel boyutları ile test
   - Sigma değerlerinin görselleştirilmesi

**Öğrenilen Kavramlar**:
- Gaussian kernel'ın matematiksel oluşturulması
- Separebilite optimizasyonu (O(n²) → O(2n))
- Otomatik sigma hesaplama formülü
- Performans analizi ve optimizasyon
- OpenCV ile karşılaştırma ve doğrulama

### Assignment 6: Geleneksel Görüntü İşleme Yöntemleri ✅
**Görev**: Thresholding, Kenar Bulma, Morfolojik işlemler

**Yapılan İşler**:
1. **Thresholding Teknikleri**
   - Binary Thresholding (farklı eşik değerleri)
   - Adaptive Thresholding (Mean ve Gaussian)
   - Otsu Thresholding (otomatik optimal eşik)
   - Multi-level Thresholding (çoklu seviye)

2. **Kenar Bulma Algoritmaları**
   - **Canny Edge Detection**: Farklı parametrelerle
   - **Sobel Operator**: X ve Y yönlerinde ayrı ayrı
   - **Laplacian Operator**: İkinci türev tabanlı
   - **Prewitt Operator**: Manuel implementasyon

3. **Morfolojik İşlemler**
   - **Erosion (Aşınma)**: Nesneleri küçültme
   - **Dilation (Genişleme)**: Nesneleri büyütme
   - **Opening**: Erosion + Dilation (gürültü temizleme)
   - **Closing**: Dilation + Erosion (boşluk doldurma)
   - **Gradient**: Kenar çıkarma
   - **Top Hat & Black Hat**: Özel morfolojik işlemler

4. **Kombinasyon Teknikleri**
   - Thresholding + Morphology: İkili görüntü iyileştirme
   - Edge Detection + Morphology: Kenar temizleme
   - Multi-stage Processing: Çok aşamalı işleme

5. **Gerçek Dünya Uygulamaları**
   - **Belge İşleme**: OCR öncesi görüntü temizleme
   - **Nesne Sayma**: Kontur tespiti ve sayma
   - **Kenar Tabanlı Nesne Tespiti**: Canny + Morphology
   - **Gürültü Azaltma**: Gaussian, Bilateral, Median filtreler

**Öğrenilen Kavramlar**:
- Thresholding teorisi ve farklı yöntemleri
- Kenar bulma algoritmalarının matematiksel temelleri
- Morfolojik işlemlerin teorisi ve uygulamaları
- Kombinasyon teknikleri ve çok aşamalı işleme
- Gerçek dünya problemlerine uygulama

## 🔄 Sonraki Görevler

### Assignment 7: Keypoint ve Özellik Çıkarımı ✅
**Görev**: SIFT, ORB gibi algoritmalar ile anlamlı noktaların tespiti

**Yapılan İşler**:
1. **Keypoint Detection Algoritmaları**
   - Harris Corner Detection: Köşe noktalarının tespiti
   - SIFT (Scale-Invariant Feature Transform): Ölçek değişmez özellik dönüşümü
   - ORB (Oriented FAST and Rotated BRIEF): Hızlı ve döndürülmüş özellikler
   - FAST (Features from Accelerated Segment Test): Hızlandırılmış segment testi
   - SURF (Speeded Up Robust Features): Hızlandırılmış sağlam özellikler

2. **Feature Matching Teknikleri**
   - Brute Force Matching: Tüm olası eşleştirmeleri deneme
   - FLANN Matching: Hızlı yakın komşu arama
   - Lowe's Ratio Test: Sağlam eşleştirmeleri filtreleme
   - RANSAC: Outlier'ları temizleme

3. **Homography ve Uygulamalar**
   - Homography Matrix: Perspektif dönüşüm hesaplama
   - Image Stitching: Görüntü birleştirme (panorama)
   - Object Detection: Template matching ile nesne tespiti
   - Feature Density Analysis: Keypoint yoğunluk analizi

4. **Performans Karşılaştırması**
   - Detection süreleri karşılaştırması
   - Matching süreleri karşılaştırması
   - Keypoint sayısı analizi
   - Match kalitesi değerlendirmesi

**Öğrenilen Kavramlar**:
- Keypoint detection teorisi ve farklı algoritmaları
- Feature extraction ve descriptor hesaplama
- Feature matching teknikleri ve optimizasyon
- Homography estimation ve perspektif dönüşüm
- Gerçek dünya uygulamaları (panorama, nesne tespiti)

## 🔄 Sonraki Görevler

### Assignment 8: YOLO Nesne Tespiti ✅
**Görev**: YOLO algoritması ile nesne tespiti

### Assignment 9: Pose Estimation ✅
**Görev**: İnsan poz tahmini ve analizi

### Assignment 10: OCR (Optical Character Recognition) ✅
**Görev**: Görüntüden metin çıkarma

### Assignment 11: YOLOv8 İnsan Tespiti ve Sayımı ✅
**Görev**: YOLOv8 modeli ile insan tespiti ve sayımı

**Yapılan İşler**:
1. **YOLOv8 Model Entegrasyonu**
   - ultralytics kütüphanesi ile YOLOv8n modeli
   - Otomatik model indirme ve yükleme
   - Person sınıfı için özel filtreleme

2. **Şık Kullanıcı Arayüzü**
   - Tkinter ile modern GUI tasarımı
   - Video, görüntü ve kamera desteği
   - Gerçek zamanlı FPS ve insan sayısı gösterimi
   - Ayarlanabilir güven eşiği

3. **Çoklu Giriş Desteği**
   - Video dosyası işleme (MP4, AVI, MOV, MKV)
   - Görüntü dosyası işleme (JPG, PNG, BMP)
   - Kamera ile gerçek zamanlı tespit
   - Threading ile performans optimizasyonu

4. **İnsan Sayımı ve Analiz**
   - Frame bazında insan sayısı takibi
   - Toplam ve ortalama istatistikler
   - JSON ve TXT formatında sonuç kaydetme
   - Tespit edilen insanların bounding box'ları

5. **Sonuç Kaydetme ve Raporlama**
   - Tespit edilen görüntülerin kaydedilmesi
   - İnsan sayımı raporları
   - JSON formatında detaylı analiz
   - Performans metrikleri (FPS)

**Öğrenilen Kavramlar**:
- YOLOv8 modeli kullanımı ve konfigürasyonu
- Gerçek zamanlı nesne tespiti
- Video ve görüntü işleme teknikleri
- Kullanıcı arayüzü tasarımı (Tkinter)
- Threading ile performans optimizasyonu
- İnsan sayımı algoritması
- Sonuç kaydetme ve raporlama

---
