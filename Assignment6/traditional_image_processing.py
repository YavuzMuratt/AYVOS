"""
Geleneksel Görüntü İşleme Yöntemleri
Assignment 6: Thresholding, Kenar Bulma, Morfolojik İşlemler

Bu dosya klasik görüntü işleme yöntemlerini öğretmek için hazırlanmıştır.
Thresholding, kenar bulma ve morfolojik işlemleri kapsamlı şekilde ele alır.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class TraditionalImageProcessing:
    """
    Geleneksel görüntü işleme yöntemlerini gösteren sınıf
    """
    
    def __init__(self):
        """Sınıfı başlatır ve gerekli klasörleri oluşturur"""
        self.create_directories()
        self.image_path = "images/image3.jpg"
        self.results_dir = "results"
        
    def create_directories(self):
        """Gerekli klasörleri oluşturur"""
        directories = [
            "images",
            "results",
            "results/thresholding",
            "results/edge_detection", 
            "results/morphological",
            "results/combinations",
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_sample_image(self):
        """
        Demo için örnek görüntü oluşturur
        """
        # 400x400 boyutunda karmaşık bir görüntü oluştur
        image = np.zeros((400, 400, 3), dtype=np.uint8)

        # 1. Farklı renklerde şekiller
        cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Mavi
        cv2.rectangle(image, (200, 50), (300, 150), (0, 255, 0), -1)  # Yeşil
        cv2.circle(image, (350, 100), 50, (0, 0, 255), -1)  # Kırmızı daire

        # 2. İnce çizgiler (kenar tespiti için)
        for i in range(0, 400, 20):
            cv2.line(image, (i, 200), (i+10, 220), (255, 255, 255), 2)

        # 3. Gürültü ekle (thresholding için)
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

        # 4. Metin ekle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Traditional Image Processing", (20, 350), font, 0.8, (255, 255, 255), 2)
        cv2.putText(image, "Assignment 6", (20, 380), font, 0.6, (255, 255, 255), 1)

        # 5. Morfolojik işlemler için küçük nesneler
        for i in range(5):
            x = np.random.randint(50, 350)
            y = np.random.randint(250, 320)
            cv2.circle(image, (x, y), 3, (255, 255, 255), -1)

        # Görüntüyü kaydet
        cv2.imwrite(self.image_path, image)
        print(f"✅ Örnek görüntü oluşturuldu: {self.image_path}")
    
    def demo_thresholding(self):
        """
        Thresholding tekniklerini gösterir
        """
        print("\n" + "="*60)
        print("1. THRESHOLDING TEKNİKLERİ")
        print("="*60)
        
        # Görüntü yoksa oluştur
        if not os.path.exists(self.image_path):
            self.create_sample_image()
        
        # Görüntüyü oku ve gri tonlamaya çevir
        image = cv2.imread(self.image_path)
        if image is None:
            print("❌ Görüntü okunamadı!")
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print(f"\n📖 Orijinal görüntü boyutu: {gray.shape}")
        
        # 1. Binary Thresholding
        print(f"\n🔧 1. Binary Thresholding:")
        thresholds = [50, 100, 150, 200]
        
        for thresh in thresholds:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            save_path = f"{self.results_dir}/thresholding/binary_thresh_{thresh}.jpg"
            cv2.imwrite(save_path, binary)
            print(f"   ✅ Eşik {thresh} -> binary_thresh_{thresh}.jpg")
        
        # 2. Adaptive Thresholding
        print(f"\n🔧 2. Adaptive Thresholding:")
        adaptive_methods = [
            (cv2.ADAPTIVE_THRESH_MEAN_C, "Mean"),
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "Gaussian")
        ]
        
        for method, name in adaptive_methods:
            adaptive = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, 11, 2)
            save_path = f"{self.results_dir}/thresholding/adaptive_{name.lower()}.jpg"
            cv2.imwrite(save_path, adaptive)
            print(f"   ✅ {name} Adaptive -> adaptive_{name.lower()}.jpg")
        
        # 3. Otsu Thresholding
        print(f"\n🔧 3. Otsu Thresholding:")
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        save_path = f"{self.results_dir}/thresholding/otsu_thresholding.jpg"
        cv2.imwrite(save_path, otsu)
        print(f"   ✅ Otsu -> otsu_thresholding.jpg")
        
        # 4. Multi-level Thresholding
        print(f"\n🔧 4. Multi-level Thresholding:")
        # 3 seviyeli thresholding
        multi_level = np.zeros_like(gray)
        multi_level[gray < 85] = 0
        multi_level[(gray >= 85) & (gray < 170)] = 128
        multi_level[gray >= 170] = 255
        
        save_path = f"{self.results_dir}/thresholding/multi_level_thresholding.jpg"
        cv2.imwrite(save_path, multi_level)
        print(f"   ✅ Multi-level -> multi_level_thresholding.jpg")
        
        # Karşılaştırma görüntüsü
        comparison = np.hstack([gray, binary, adaptive, otsu, multi_level])
        cv2.imwrite(f"{self.results_dir}/thresholding/thresholding_comparison.jpg", comparison)
        print(f"   ✅ Karşılaştırma -> thresholding_comparison.jpg")
        
        return gray
    
    def demo_edge_detection(self, gray):
        """
        Kenar bulma algoritmalarını gösterir
        """
        print("\n" + "="*60)
        print("2. KENAR BULMA ALGORİTMALARI")
        print("="*60)
        
        print(f"\n📚 Kenar Bulma Algoritmaları:")
        print(f"   💡 Canny: En popüler, çok aşamalı algoritma")
        print(f"   💡 Sobel: Gradient tabanlı, x ve y yönlerinde")
        print(f"   💡 Laplacian: İkinci türev, gürültüye duyarlı")
        print(f"   💡 Prewitt: Basit gradient operatörü")
        
        # 1. Canny Edge Detection
        print(f"\n🔧 1. Canny Edge Detection:")
        canny_low = cv2.Canny(gray, 50, 150)
        canny_high = cv2.Canny(gray, 100, 200)
        canny_very_high = cv2.Canny(gray, 150, 250)
        
        cv2.imwrite(f"{self.results_dir}/edge_detection/canny_low.jpg", canny_low)
        cv2.imwrite(f"{self.results_dir}/edge_detection/canny_high.jpg", canny_high)
        cv2.imwrite(f"{self.results_dir}/edge_detection/canny_very_high.jpg", canny_very_high)
        
        print(f"   ✅ Canny (50,150) -> canny_low.jpg")
        print(f"   ✅ Canny (100,200) -> canny_high.jpg")
        print(f"   ✅ Canny (150,250) -> canny_very_high.jpg")
        
        # 2. Sobel Operator
        print(f"\n🔧 2. Sobel Operator:")
        # X yönünde
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.uint8(np.absolute(sobel_x))
        
        # Y yönünde
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_y = np.uint8(np.absolute(sobel_y))
        
        # Kombine
        sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        
        cv2.imwrite(f"{self.results_dir}/edge_detection/sobel_x.jpg", sobel_x)
        cv2.imwrite(f"{self.results_dir}/edge_detection/sobel_y.jpg", sobel_y)
        cv2.imwrite(f"{self.results_dir}/edge_detection/sobel_combined.jpg", sobel_combined)
        
        print(f"   ✅ Sobel X -> sobel_x.jpg")
        print(f"   ✅ Sobel Y -> sobel_y.jpg")
        print(f"   ✅ Sobel Combined -> sobel_combined.jpg")
        
        # 3. Laplacian Operator
        print(f"\n🔧 3. Laplacian Operator:")
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        cv2.imwrite(f"{self.results_dir}/edge_detection/laplacian.jpg", laplacian)
        print(f"   ✅ Laplacian -> laplacian.jpg")
        
        # 4. Prewitt Operator (manuel implementasyon)
        print(f"\n🔧 4. Prewitt Operator:")
        # Prewitt kernel'ları
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        prewitt_x = cv2.filter2D(gray, -1, kernel_x)
        prewitt_y = cv2.filter2D(gray, -1, kernel_y)
        prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
        
        cv2.imwrite(f"{self.results_dir}/edge_detection/prewitt_x.jpg", prewitt_x)
        cv2.imwrite(f"{self.results_dir}/edge_detection/prewitt_y.jpg", prewitt_y)
        cv2.imwrite(f"{self.results_dir}/edge_detection/prewitt_combined.jpg", prewitt_combined)
        
        print(f"   ✅ Prewitt X -> prewitt_x.jpg")
        print(f"   ✅ Prewitt Y -> prewitt_y.jpg")
        print(f"   ✅ Prewitt Combined -> prewitt_combined.jpg")
        
        # Karşılaştırma görüntüsü
        comparison = np.hstack([gray, canny_high, sobel_combined, laplacian, prewitt_combined])
        cv2.imwrite(f"{self.results_dir}/edge_detection/edge_detection_comparison.jpg", comparison)
        print(f"   ✅ Karşılaştırma -> edge_detection_comparison.jpg")
        
        return canny_high, sobel_combined, laplacian
    
    def demo_morphological_operations(self, gray):
        """
        Morfolojik işlemleri gösterir
        """
        print("\n" + "="*60)
        print("3. MORFOLOJİK İŞLEMLER")
        print("="*60)
        
        print(f"\n📚 Morfolojik İşlemler:")
        print(f"   💡 Erosion: Nesneleri küçültme, gürültüyü azaltma")
        print(f"   💡 Dilation: Nesneleri büyütme, boşlukları doldurma")
        print(f"   💡 Opening: Erosion + Dilation (gürültü temizleme)")
        print(f"   💡 Closing: Dilation + Erosion (boşluk doldurma)")
        
        # Kernel oluştur
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # 1. Erosion
        print(f"\n🔧 1. Erosion (Aşınma):")
        erosion = cv2.erode(gray, kernel, iterations=1)
        erosion_strong = cv2.erode(gray, kernel, iterations=3)
        
        cv2.imwrite(f"{self.results_dir}/morphological/erosion.jpg", erosion)
        cv2.imwrite(f"{self.results_dir}/morphological/erosion_strong.jpg", erosion_strong)
        
        print(f"   ✅ Erosion (1 iterasyon) -> erosion.jpg")
        print(f"   ✅ Erosion (3 iterasyon) -> erosion_strong.jpg")
        
        # 2. Dilation
        print(f"\n🔧 2. Dilation (Genişleme):")
        dilation = cv2.dilate(gray, kernel, iterations=1)
        dilation_strong = cv2.dilate(gray, kernel, iterations=3)
        
        cv2.imwrite(f"{self.results_dir}/morphological/dilation.jpg", dilation)
        cv2.imwrite(f"{self.results_dir}/morphological/dilation_strong.jpg", dilation_strong)
        
        print(f"   ✅ Dilation (1 iterasyon) -> dilation.jpg")
        print(f"   ✅ Dilation (3 iterasyon) -> dilation_strong.jpg")
        
        # 3. Opening (Erosion + Dilation)
        print(f"\n🔧 3. Opening (Açma):")
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        opening_strong = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
        
        cv2.imwrite(f"{self.results_dir}/morphological/opening.jpg", opening)
        cv2.imwrite(f"{self.results_dir}/morphological/opening_strong.jpg", opening_strong)
        
        print(f"   ✅ Opening -> opening.jpg")
        print(f"   ✅ Opening (2 iterasyon) -> opening_strong.jpg")
        
        # 4. Closing (Dilation + Erosion)
        print(f"\n🔧 4. Closing (Kapatma):")
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        closing_strong = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        cv2.imwrite(f"{self.results_dir}/morphological/closing.jpg", closing)
        cv2.imwrite(f"{self.results_dir}/morphological/closing_strong.jpg", closing_strong)
        
        print(f"   ✅ Closing -> closing.jpg")
        print(f"   ✅ Closing (2 iterasyon) -> closing_strong.jpg")
        
        # 5. Gradient (Dilation - Erosion)
        print(f"\n🔧 5. Gradient (Kenar Çıkarma):")
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        cv2.imwrite(f"{self.results_dir}/morphological/gradient.jpg", gradient)
        print(f"   ✅ Gradient -> gradient.jpg")
        
        # 6. Top Hat (Opening - Original)
        print(f"\n🔧 6. Top Hat (Açık Şapka):")
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        cv2.imwrite(f"{self.results_dir}/morphological/tophat.jpg", tophat)
        print(f"   ✅ Top Hat -> tophat.jpg")
        
        # 7. Black Hat (Closing - Original)
        print(f"\n🔧 7. Black Hat (Siyah Şapka):")
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        cv2.imwrite(f"{self.results_dir}/morphological/blackhat.jpg", blackhat)
        print(f"   ✅ Black Hat -> blackhat.jpg")
        
        # Karşılaştırma görüntüsü
        comparison = np.hstack([gray, erosion, dilation, opening, closing, gradient])
        cv2.imwrite(f"{self.results_dir}/morphological/morphological_comparison.jpg", comparison)
        print(f"   ✅ Karşılaştırma -> morphological_comparison.jpg")
        
        return erosion, dilation, opening, closing, gradient
    
    def demo_combinations(self, gray):
        """
        Farklı yöntemleri birleştirme tekniklerini gösterir
        """
        print("\n" + "="*60)
        print("4. KOMBİNASYON TEKNİKLERİ")
        print("="*60)
        
        print(f"\n📚 Kombinasyon Teknikleri:")
        print(f"   💡 Thresholding + Morphology: İkili görüntü iyileştirme")
        print(f"   💡 Edge Detection + Morphology: Kenar temizleme")
        print(f"   💡 Multi-stage Processing: Çok aşamalı işleme")
        
        # 1. Thresholding + Morphology
        print(f"\n🔧 1. Thresholding + Morphology:")
        # Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary_filled = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, kernel)
        
        cv2.imwrite(f"{self.results_dir}/combinations/binary_original.jpg", binary)
        cv2.imwrite(f"{self.results_dir}/combinations/binary_cleaned.jpg", binary_cleaned)
        cv2.imwrite(f"{self.results_dir}/combinations/binary_filled.jpg", binary_filled)
        
        print(f"   ✅ Binary Original -> binary_original.jpg")
        print(f"   ✅ Binary Cleaned -> binary_cleaned.jpg")
        print(f"   ✅ Binary Filled -> binary_filled.jpg")
        
        # 2. Edge Detection + Morphology
        print(f"\n🔧 2. Edge Detection + Morphology:")
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Morfolojik işlemler ile kenarları temizle
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line)
        edges_thick = cv2.dilate(edges, kernel_line, iterations=1)
        
        cv2.imwrite(f"{self.results_dir}/combinations/edges_original.jpg", edges)
        cv2.imwrite(f"{self.results_dir}/combinations/edges_cleaned.jpg", edges_cleaned)
        cv2.imwrite(f"{self.results_dir}/combinations/edges_thick.jpg", edges_thick)
        
        print(f"   ✅ Edges Original -> edges_original.jpg")
        print(f"   ✅ Edges Cleaned -> edges_cleaned.jpg")
        print(f"   ✅ Edges Thick -> edges_thick.jpg")
        
        # 3. Multi-stage Processing
        print(f"\n🔧 3. Multi-stage Processing:")
        # Aşama 1: Gürültü azaltma
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Aşama 2: Kenar tespiti
        edges_multi = cv2.Canny(blurred, 50, 150)
        
        # Aşama 3: Morfolojik işlemler
        kernel_multi = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_multi_cleaned = cv2.morphologyEx(edges_multi, cv2.MORPH_CLOSE, kernel_multi)
        
        # Aşama 4: Kontur bulma
        contours, _ = cv2.findContours(edges_multi_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Konturları çiz
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        cv2.imwrite(f"{self.results_dir}/combinations/blurred.jpg", blurred)
        cv2.imwrite(f"{self.results_dir}/combinations/edges_multi.jpg", edges_multi)
        cv2.imwrite(f"{self.results_dir}/combinations/edges_multi_cleaned.jpg", edges_multi_cleaned)
        cv2.imwrite(f"{self.results_dir}/combinations/contours_result.jpg", result)
        
        print(f"   ✅ Blurred -> blurred.jpg")
        print(f"   ✅ Edges Multi -> edges_multi.jpg")
        print(f"   ✅ Edges Multi Cleaned -> edges_multi_cleaned.jpg")
        print(f"   ✅ Contours Result -> contours_result.jpg")
        
        # Karşılaştırma görüntüsü - boyut uyumluluğu için
        # Tüm görüntüleri aynı boyuta getir
        gray_3d = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        binary_filled_3d = cv2.cvtColor(binary_filled, cv2.COLOR_GRAY2BGR)
        edges_thick_3d = cv2.cvtColor(edges_thick, cv2.COLOR_GRAY2BGR)
        
        comparison = np.hstack([gray_3d, binary_filled_3d, edges_thick_3d, result])
        cv2.imwrite(f"{self.results_dir}/combinations/combinations_comparison.jpg", comparison)
        print(f"   ✅ Karşılaştırma -> combinations_comparison.jpg")
    
    def run_demo(self):
        """
        Tüm demo'ları çalıştırır
        """
        print("🚀 Geleneksel Görüntü İşleme Demo Başlıyor...")
        print("="*60)
        
        try:
            # 1. Thresholding teknikleri
            gray = self.demo_thresholding()
            if gray is None:
                print("❌ Demo başlatılamadı!")
                return
            
            # 2. Kenar bulma algoritmaları
            canny, sobel, laplacian = self.demo_edge_detection(gray)
            
            # 3. Morfolojik işlemler
            erosion, dilation, opening, closing, gradient = self.demo_morphological_operations(gray)
            
            # 4. Kombinasyon teknikleri
            self.demo_combinations(gray)
            

            
            print("\n" + "="*60)
            print("🎉 Geleneksel Görüntü İşleme Demo tamamlandı!")
            print("📁 Sonuçlar 'results' klasöründe bulunabilir")
            print("📚 README.md dosyasını inceleyerek öğrenilenleri gözden geçirin")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    # Demo'yu başlat
    demo = TraditionalImageProcessing()
    demo.run_demo() 