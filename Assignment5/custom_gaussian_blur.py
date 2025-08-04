"""
Kendi Gaussian Blur Implementasyonu
Assignment 5: Gaussian kernel'ı elle oluşturarak filtre uygulama

Bu dosya Gaussian Blur'ın nasıl çalıştığını derinlemesine öğretmek için hazırlanmıştır.
Matematiksel formül, manuel implementasyon ve OpenCV karşılaştırması içerir.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

class CustomGaussianBlur:
    """
    Kendi Gaussian Blur implementasyonunu gösteren sınıf
    """
    
    def __init__(self):
        """Sınıfı başlatır ve gerekli klasörleri oluşturur"""
        self.create_directories()
        self.image_path = "images/image1.jpg"
        self.results_dir = "results"
        
    def create_directories(self):
        """Gerekli klasörleri oluşturur"""
        directories = [
            "images", 
            "results", 
            "results/custom_implementations", 
            "results/comparisons",
            "results/performance_analysis"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_gaussian_kernel(self, size, sigma):
        """
        Gaussian kernel oluşturur
        
        Args:
            size (int): Kernel boyutu (tek sayı olmalı)
            sigma (float): Standart sapma
            
        Returns:
            numpy.ndarray: Gaussian kernel
        """
        # Merkez noktasını hesapla
        center = size // 2
        
        # Kernel matrisini oluştur
        kernel = np.zeros((size, size))
        
        # Her piksel için Gaussian değerini hesapla
        for i in range(size):
            for j in range(size):
                # Merkeze göre koordinatları hesapla
                x = i - center
                y = j - center
                
                # Gaussian formülü: G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
                exponent = -(x**2 + y**2) / (2 * sigma**2)
                kernel[i, j] = np.exp(exponent)
        
        # Kernel'i normalize et (toplam ağırlık 1 olsun)
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def apply_gaussian_blur_manual(self, image, kernel_size, sigma):
        """
        Manuel olarak Gaussian Blur uygular
        
        Args:
            image (numpy.ndarray): Giriş görüntüsü
            kernel_size (int): Kernel boyutu
            sigma (float): Standart sapma
            
        Returns:
            numpy.ndarray: Bulanıklaştırılmış görüntü
        """
        # Kernel oluştur
        kernel = self.create_gaussian_kernel(kernel_size, sigma)
        
        # Görüntü boyutlarını al
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        # Padding hesapla
        pad = kernel_size // 2
        
        # Çıktı görüntüsünü oluştur
        if channels == 1:
            output = np.zeros((height, width), dtype=np.uint8)
        else:
            output = np.zeros((height, width, channels), dtype=np.uint8)
        
        # Her piksel için filtreleme uygula
        for i in range(pad, height - pad):
            for j in range(pad, width - pad):
                if channels == 1:
                    # Gri tonlama için
                    region = image[i-pad:i+pad+1, j-pad:j+pad+1]
                    output[i, j] = np.sum(region * kernel)
                else:
                    # Renkli görüntü için her kanalı ayrı ayrı işle
                    for c in range(channels):
                        region = image[i-pad:i+pad+1, j-pad:j+pad+1, c]
                        output[i, j, c] = np.sum(region * kernel)
        
        return output
    
    def calculate_sigma_from_kernel_size(self, kernel_size):
        """
        Kernel boyutuna göre sigma değerini hesaplar (OpenCV formülü)
        
        Args:
            kernel_size (int): Kernel boyutu
            
        Returns:
            float: Hesaplanan sigma değeri
        """
        return 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    def apply_gaussian_blur_separable(self, image, kernel_size, sigma):
        """
        Separebilite kullanarak verimli Gaussian Blur uygular
        (OpenCV'nin cv2.filter2D kullanarak)
        
        Args:
            image (numpy.ndarray): Giriş görüntüsü
            kernel_size (int): Kernel boyutu
            sigma (float): Standart sapma
            
        Returns:
            numpy.ndarray: Bulanıklaştırılmış görüntü
        """
        # 1D Gaussian kernel oluştur
        kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
        
        # İlk olarak yatay yönde filtreleme
        temp = cv2.filter2D(image, -1, kernel_1d)
        
        # Sonra dikey yönde filtreleme (transpose ile)
        output = cv2.filter2D(temp, -1, kernel_1d.T)
        
        return output
    
    def apply_gaussian_blur_manual_separable(self, image, kernel_size, sigma):
        """
        Manuel olarak separebilite kullanarak Gaussian Blur uygular
        (2D kernel yerine 1D kernel kullanarak)
        
        Args:
            image (numpy.ndarray): Giriş görüntüsü
            kernel_size (int): Kernel boyutu
            sigma (float): Standart sapma
            
        Returns:
            numpy.ndarray: Bulanıklaştırılmış görüntü
        """
        # 1D Gaussian kernel oluştur
        center = kernel_size // 2
        kernel_1d = np.zeros(kernel_size)
        
        for i in range(kernel_size):
            x = i - center
            kernel_1d[i] = np.exp(-(x**2) / (2 * sigma**2))
        
        # Normalize et
        kernel_1d = kernel_1d / np.sum(kernel_1d)
        
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        pad = kernel_size // 2
        output = np.zeros_like(image)
        
        # Yatay yönde filtreleme
        temp = np.zeros_like(image)
        for i in range(height):
            for j in range(pad, width - pad):
                if channels == 1:
                    region = image[i, j-pad:j+pad+1]
                    temp[i, j] = np.sum(region * kernel_1d)
                else:
                    for c in range(channels):
                        region = image[i, j-pad:j+pad+1, c]
                        temp[i, j, c] = np.sum(region * kernel_1d)
        
        # Dikey yönde filtreleme
        for i in range(pad, height - pad):
            for j in range(width):
                if channels == 1:
                    region = temp[i-pad:i+pad+1, j]
                    output[i, j] = np.sum(region * kernel_1d)
                else:
                    for c in range(channels):
                        region = temp[i-pad:i+pad+1, j, c]
                        output[i, j, c] = np.sum(region * kernel_1d)
        
        return output
    
    def demo_kernel_creation(self):
        """
        Kernel oluşturma sürecini gösterir
        """
        print("\n" + "="*60)
        print("1. GAUSSIAN KERNEL OLUŞTURMA")
        print("="*60)
        
        print("\n📚 Gaussian Kernel Formülü:")
        print("   G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))")
        print("   💡 σ (sigma): Standart sapma")
        print("   💡 x,y: Kernel koordinatları")
        print("   💡 Normalizasyon: Toplam ağırlık = 1")
        
        # Farklı parametrelerle kernel oluştur
        test_cases = [
            (3, 1.0, "Küçük Kernel"),
            (5, 1.5, "Orta Kernel"),
            (7, 2.0, "Büyük Kernel"),
            (9, 2.5, "Çok Büyük Kernel")
        ]
        
        print(f"\n🔧 Kernel Oluşturma Testleri:")
        
        for size, sigma, description in test_cases:
            # Kernel oluştur
            kernel = self.create_gaussian_kernel(size, sigma)
            
            print(f"\n   📐 {description} ({size}x{size}, σ={sigma}):")
            print(f"      📊 Merkez değeri: {kernel[size//2, size//2]:.6f}")
            print(f"      📊 Toplam ağırlık: {np.sum(kernel):.6f}")
            print(f"      📊 Maksimum değer: {np.max(kernel):.6f}")
            print(f"      📊 Minimum değer: {np.min(kernel):.6f}")
            
            # Kernel'i görselleştir
            plt.figure(figsize=(6, 5))
            plt.imshow(kernel, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Ağırlık')
            plt.title(f'Gaussian Kernel: {size}x{size}, σ={sigma}')
            plt.xlabel('X')
            plt.ylabel('Y')
            
            # Kaydet
            save_path = f"{self.results_dir}/custom_implementations/kernel_{size}x{size}_sigma_{sigma}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"      ✅ Görselleştirme -> {save_path}")
    
    def demo_manual_blur(self):
        """
        Manuel Gaussian Blur uygulamasını gösterir
        """
        print("\n" + "="*60)
        print("2. MANUEL GAUSSIAN BLUR UYGULAMASI")
        print("="*60)
        
        # Görüntü yoksa oluştur
        if not os.path.exists(self.image_path):
            self.create_sample_image()
        
        # Görüntüyü oku
        image = cv2.imread(self.image_path)
        if image is None:
            print("❌ Görüntü okunamadı!")
            return None
        
        print(f"\n📖 Orijinal görüntü boyutu: {image.shape}")
        
        # Farklı parametrelerle manuel blur uygula
        test_cases = [
            (3, 0.5, "Hafif Manuel Blur"),
            (5, 1.0, "Orta Manuel Blur"),
            (7, 1.5, "Güçlü Manuel Blur"),
            (9, 2.0, "Çok Güçlü Manuel Blur")
        ]
        
        print(f"\n🔄 Manuel Gaussian Blur Testleri:")
        
        for kernel_size, sigma, description in test_cases:
            print(f"\n   🔧 {description} (Kernel: {kernel_size}x{kernel_size}, σ={sigma}):")
            
            # Zaman ölçümü başlat
            start_time = time.time()
            
            # Manuel blur uygula
            blurred = self.apply_gaussian_blur_manual(image, kernel_size, sigma)
            
            # Zaman ölçümü bitir
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Kaydet
            filename = f"manual_blur_{kernel_size}x{kernel_size}_sigma_{sigma}.jpg"
            save_path = f"{self.results_dir}/custom_implementations/{filename}"
            cv2.imwrite(save_path, blurred)
            
            print(f"      ✅ İşlem tamamlandı -> {save_path}")
            print(f"      ⏱️  İşlem süresi: {processing_time:.4f} saniye")
            
        return image
    
    def demo_opencv_comparison(self, image):
        """
        Kendi implementasyonumuzu OpenCV ile karşılaştırır
        """
        print("\n" + "="*60)
        print("3. OPENCV İLE KARŞILAŞTIRMA")
        print("="*60)
        
        # Test parametreleri
        kernel_size = 5
        sigma = 1.0
        
        print(f"\n🔍 Karşılaştırma Parametreleri:")
        print(f"   📏 Kernel boyutu: {kernel_size}x{kernel_size}")
        print(f"   📊 Sigma: {sigma}")
        
        # Kendi implementasyonumuz
        print(f"\n🔧 Kendi Implementasyonumuz:")
        start_time = time.time()
        custom_blurred = self.apply_gaussian_blur_manual(image, kernel_size, sigma)
        custom_time = time.time() - start_time
        
        # OpenCV implementasyonu
        print(f"🔧 OpenCV Implementasyonu:")
        start_time = time.time()
        opencv_blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        opencv_time = time.time() - start_time
        
        # Sonuçları kaydet
        cv2.imwrite(f"{self.results_dir}/comparisons/custom_implementation.jpg", custom_blurred)
        cv2.imwrite(f"{self.results_dir}/comparisons/opencv_implementation.jpg", opencv_blurred)
        
        # Karşılaştırma görüntüsü oluştur
        comparison = np.hstack([image, custom_blurred, opencv_blurred])
        cv2.imwrite(f"{self.results_dir}/comparisons/side_by_side_comparison.jpg", comparison)
        
        print(f"   ✅ Kendi implementasyon -> custom_implementation.jpg")
        print(f"   ✅ OpenCV implementasyonu -> opencv_implementation.jpg")
        print(f"   ✅ Yan yana karşılaştırma -> side_by_side_comparison.jpg")
        
        # Performans karşılaştırması
        print(f"\n📊 Performans Karşılaştırması:")
        print(f"   ⏱️  Kendi implementasyon süresi: {custom_time:.4f} saniye")
        print(f"   ⏱️  OpenCV süresi: {opencv_time:.4f} saniye")
        print(f"   🚀 Hız farkı: {custom_time/opencv_time:.2f}x daha yavaş")
        
        # Kalite karşılaştırması
        mse = np.mean((custom_blurred.astype(float) - opencv_blurred.astype(float))**2)
        print(f"   📊 Ortalama Kare Hata (MSE): {mse:.6f}")
        
        return custom_blurred, opencv_blurred
    
    def demo_efficient_implementations(self, image):
        """
        Verimli implementasyonları gösterir ve karşılaştırır
        """
        print("\n" + "="*60)
        print("4. VERİMLİ İMPLEMENTASYONLAR")
        print("="*60)
        
        # Test parametreleri
        kernel_size = 9
        sigma = self.calculate_sigma_from_kernel_size(kernel_size)
        
        print(f"\n🔍 Verimli Implementasyon Testi:")
        print(f"   📏 Kernel boyutu: {kernel_size}x{kernel_size}")
        print(f"   📊 Otomatik sigma: {sigma:.3f}")
        
        # 1. Manuel 2D kernel (yavaş)
        print(f"\n🔧 1. Manuel 2D Kernel (Yavaş):")
        start_time = time.time()
        manual_2d = self.apply_gaussian_blur_manual(image, kernel_size, sigma)
        manual_2d_time = time.time() - start_time
        
        # 2. Manuel separebilite (orta hız)
        print(f"🔧 2. Manuel Separebilite (Orta Hız):")
        start_time = time.time()
        manual_separable = self.apply_gaussian_blur_manual_separable(image, kernel_size, sigma)
        manual_separable_time = time.time() - start_time
        
        # 3. OpenCV separebilite (hızlı)
        print(f"🔧 3. OpenCV Separebilite (Hızlı):")
        start_time = time.time()
        opencv_separable = self.apply_gaussian_blur_separable(image, kernel_size, sigma)
        opencv_separable_time = time.time() - start_time
        
        # 4. OpenCV GaussianBlur (en hızlı)
        print(f"🔧 4. OpenCV GaussianBlur (En Hızlı):")
        start_time = time.time()
        opencv_gaussian = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        opencv_gaussian_time = time.time() - start_time
        
        # Sonuçları kaydet
        cv2.imwrite(f"{self.results_dir}/custom_implementations/manual_2d_blur.jpg", manual_2d)
        cv2.imwrite(f"{self.results_dir}/custom_implementations/manual_separable_blur.jpg", manual_separable)
        cv2.imwrite(f"{self.results_dir}/custom_implementations/opencv_separable_blur.jpg", opencv_separable)
        cv2.imwrite(f"{self.results_dir}/custom_implementations/opencv_gaussian_blur.jpg", opencv_gaussian)
        
        print(f"   ✅ Manuel 2D -> manual_2d_blur.jpg")
        print(f"   ✅ Manuel Separebilite -> manual_separable_blur.jpg")
        print(f"   ✅ OpenCV Separebilite -> opencv_separable_blur.jpg")
        print(f"   ✅ OpenCV GaussianBlur -> opencv_gaussian_blur.jpg")
        
        # Performans karşılaştırması
        print(f"\n📊 Performans Karşılaştırması:")
        print(f"   ⏱️  Manuel 2D: {manual_2d_time:.4f} saniye")
        print(f"   ⏱️  Manuel Separebilite: {manual_separable_time:.4f} saniye")
        print(f"   ⏱️  OpenCV Separebilite: {opencv_separable_time:.4f} saniye")
        print(f"   ⏱️  OpenCV GaussianBlur: {opencv_gaussian_time:.4f} saniye")
        
        # Hız iyileştirmeleri
        print(f"\n🚀 Hız İyileştirmeleri:")
        print(f"   📈 Manuel Separebilite vs 2D: {manual_2d_time/manual_separable_time:.2f}x daha hızlı")
        print(f"   📈 OpenCV Separebilite vs Manuel: {manual_separable_time/opencv_separable_time:.2f}x daha hızlı")
        print(f"   📈 OpenCV GaussianBlur vs Manuel: {manual_separable_time/opencv_gaussian_time:.2f}x daha hızlı")
        
        # Kalite karşılaştırması
        mse_manual = np.mean((manual_2d.astype(float) - manual_separable.astype(float))**2)
        mse_opencv = np.mean((manual_separable.astype(float) - opencv_separable.astype(float))**2)
        mse_final = np.mean((opencv_separable.astype(float) - opencv_gaussian.astype(float))**2)
        
        print(f"\n📊 Kalite Karşılaştırması (MSE):")
        print(f"   📊 Manuel 2D vs Separebilite: {mse_manual:.6f}")
        print(f"   📊 Manuel vs OpenCV Separebilite: {mse_opencv:.6f}")
        print(f"   📊 OpenCV Separebilite vs GaussianBlur: {mse_final:.6f}")
        
        # Karşılaştırma görüntüsü
        comparison = np.hstack([image, manual_2d, manual_separable, opencv_gaussian])
        cv2.imwrite(f"{self.results_dir}/comparisons/efficient_implementations_comparison.jpg", comparison)
        print(f"   ✅ Karşılaştırma görüntüsü -> efficient_implementations_comparison.jpg")
    
    def demo_auto_sigma_calculation(self, image):
        """
        Otomatik sigma hesaplama fonksiyonunu gösterir
        """
        print("\n" + "="*60)
        print("5. OTOMATİK SIGMA HESAPLAMA")
        print("="*60)
        
        # Farklı kernel boyutları ile test
        kernel_sizes = [3, 5, 7, 9, 11, 15, 21]
        
        print(f"\n🔍 Otomatik Sigma Hesaplama:")
        print(f"   📏 Test edilen kernel boyutları: {kernel_sizes}")
        
        for size in kernel_sizes:
            # Otomatik sigma hesapla
            auto_sigma = self.calculate_sigma_from_kernel_size(size)
            
            print(f"\n   📐 Kernel {size}x{size}:")
            print(f"      📊 Otomatik sigma: {auto_sigma:.3f}")
            
            # Bu parametrelerle blur uygula
            blurred = self.apply_gaussian_blur_separable(image, size, auto_sigma)
            
            # Kaydet
            filename = f"auto_sigma_kernel_{size}x{size}_sigma_{auto_sigma:.3f}.jpg"
            save_path = f"{self.results_dir}/custom_implementations/{filename}"
            cv2.imwrite(save_path, blurred)
            
            print(f"      ✅ Sonuç -> {filename}")
        
        # Sigma değerlerini görselleştir
        sigmas = [self.calculate_sigma_from_kernel_size(size) for size in kernel_sizes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(kernel_sizes, sigmas, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Kernel Boyutu')
        plt.ylabel('Otomatik Sigma Değeri')
        plt.title('Kernel Boyutuna Göre Otomatik Sigma Hesaplama')
        plt.grid(True, alpha=0.3)
        
        # Değerleri etiketle
        for i, (size, sigma) in enumerate(zip(kernel_sizes, sigmas)):
            plt.annotate(f'{sigma:.2f}', (size, sigma), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/performance_analysis/auto_sigma_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n   ✅ Sigma grafiği -> auto_sigma_plot.png")
    
    def demo_performance_analysis(self, image):
        """
        Performans analizi yapar
        """
        print("\n" + "="*60)
        print("4. PERFORMANS ANALİZİ")
        print("="*60)
        
        # Farklı kernel boyutları ile test
        kernel_sizes = [3, 5, 7, 9, 11]
        sigma = 1.0
        
        print(f"\n📊 Farklı Kernel Boyutları ile Performans Testi:")
        print(f"   📊 Sigma: {sigma}")
        
        custom_times = []
        opencv_times = []
        
        for size in kernel_sizes:
            print(f"\n   🔧 Kernel {size}x{size}:")
            
            # Kendi implementasyon
            start_time = time.time()
            custom_result = self.apply_gaussian_blur_manual(image, size, sigma)
            custom_time = time.time() - start_time
            custom_times.append(custom_time)
            
            # OpenCV implementasyonu
            start_time = time.time()
            opencv_result = cv2.GaussianBlur(image, (size, size), sigma)
            opencv_time = time.time() - start_time
            opencv_times.append(opencv_time)
            
            print(f"      ⏱️  Kendi: {custom_time:.4f}s, OpenCV: {opencv_time:.4f}s")
            print(f"      🚀 Hız oranı: {custom_time/opencv_time:.2f}x")
        
        # Grafik oluştur
        plt.figure(figsize=(12, 5))
        
        # Zaman karşılaştırması
        plt.subplot(1, 2, 1)
        plt.plot(kernel_sizes, custom_times, 'o-', label='Kendi Implementasyon', linewidth=2, markersize=8)
        plt.plot(kernel_sizes, opencv_times, 's-', label='OpenCV', linewidth=2, markersize=8)
        plt.xlabel('Kernel Boyutu')
        plt.ylabel('İşlem Süresi (saniye)')
        plt.title('Performans Karşılaştırması')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Hız oranı
        plt.subplot(1, 2, 2)
        speed_ratios = [custom/opencv for custom, opencv in zip(custom_times, opencv_times)]
        plt.bar(kernel_sizes, speed_ratios, color='orange', alpha=0.7)
        plt.xlabel('Kernel Boyutu')
        plt.ylabel('Hız Oranı (Kendi/OpenCV)')
        plt.title('Hız Oranı Karşılaştırması')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/performance_analysis/performance_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n   ✅ Performans analizi -> performance_comparison.png")
        
        # Sonuçları yazdır
        print(f"\n📈 Performans Sonuçları:")
        print(f"   🚀 En hızlı OpenCV: {min(opencv_times):.4f}s (Kernel: {kernel_sizes[opencv_times.index(min(opencv_times))]}x{kernel_sizes[opencv_times.index(min(opencv_times))]})")
        print(f"   🐌 En yavaş kendi: {max(custom_times):.4f}s (Kernel: {kernel_sizes[custom_times.index(max(custom_times))]}x{kernel_sizes[custom_times.index(max(custom_times))]})")
        print(f"   📊 Ortalama hız oranı: {np.mean(speed_ratios):.2f}x")
    
    
    def run_demo(self):
        """
        Tüm demo'ları çalıştırır
        """
        print("🚀 Kendi Gaussian Blur Implementasyonu Demo Başlıyor...")
        print("="*60)
        
        try:
            # 1. Kernel oluşturma demo'su
            self.demo_kernel_creation()
            
            # 2. Manuel blur uygulaması
            image = self.demo_manual_blur()
            if image is None:
                print("❌ Demo başlatılamadı!")
                return
            
            # 3. OpenCV karşılaştırması
            custom_result, opencv_result = self.demo_opencv_comparison(image)
            
            # 4. Verimli implementasyonlar
            self.demo_efficient_implementations(image)
            
            # 5. Otomatik sigma hesaplama
            self.demo_auto_sigma_calculation(image)
            
            # 6. Performans analizi
            self.demo_performance_analysis(image)
            
            # 7. Optimizasyon ipuçları
            self.demo_optimization_tips()
            
            print("\n" + "="*60)
            print("🎉 Kendi Gaussian Blur Demo tamamlandı!")
            print("📁 Sonuçlar 'results' klasöründe bulunabilir")
            print("📚 README.md dosyasını inceleyerek öğrenilenleri gözden geçirin")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    # Demo'yu başlat
    demo = CustomGaussianBlur()
    demo.run_demo() 