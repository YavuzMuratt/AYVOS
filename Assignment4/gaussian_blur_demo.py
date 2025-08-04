"""
Gaussian Blur Demo
Assignment 4: cv2.GaussianBlur() fonksiyonunun öğrenilmesi

Bu dosya Gaussian Blur'ın nasıl çalıştığını öğretmek için hazırlanmıştır.
Teori, kernel görselleştirme ve farklı parametrelerle denemeler içerir.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from datetime import datetime

class GaussianBlurDemo:
    """
    Gaussian Blur'ı gösteren demo sınıfı
    """
    
    def __init__(self):
        """Demo sınıfını başlatır ve gerekli klasörleri oluşturur"""
        self.create_directories()
        self.image_path = "images/image2.jpg"
        self.results_dir = "results"
        
    def create_directories(self):
        """Gerekli klasörleri oluşturur"""
        directories = [
            "images", 
            "results", 
            "results/blur_effects", 
            "results/kernel_visualizations",
            "results/comparisons"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
    def demo_gaussian_theory(self):
        """
        Gaussian Blur teorisini açıklar
        """
        print("\n" + "="*60)
        print("1. GAUSSIAN BLUR TEORİSİ")
        print("="*60)
        
        print("\n📚 Gaussian Blur Nedir?")
        print("   💡 Gaussian Blur, görüntüdeki gürültüyü azaltmak için kullanılan bir filtreleme yöntemidir.")
        print("   💡 Normal (Gaussian) dağılımına göre ağırlıklandırılmış bir kernel kullanır.")
        print("   💡 Merkezdeki piksel en yüksek ağırlığa sahiptir, kenarlara doğru azalır.")
        
        print("\n🔬 Matematiksel Formül:")
        print("   G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))")
        print("   💡 σ (sigma): Standart sapma - bulanıklaştırma yoğunluğunu belirler")
        print("   💡 x,y: Kernel içindeki koordinatlar")
        
        print("\n⚙️  Parametreler:")
        print("   📏 Kernel Boyutu: Genellikle tek sayılar (3x3, 5x5, 7x7, 9x9)")
        print("   📊 Sigma: Dağılımın standart sapması (0.3*((ksize-1)*0.5-1)+0.8)")
        print("   🎯 Sigma = 0: Otomatik hesaplama")
        
    def demo_kernel_visualization(self):
        """
        Gaussian kernel'larını görselleştirir
        """
        print("\n" + "="*60)
        print("2. GAUSSIAN KERNEL GÖRSELLEŞTİRME")
        print("="*60)
        
        # Farklı kernel boyutları
        kernel_sizes = [3, 5, 7, 9]
        sigma_values = [1.0, 1.5, 2.0]
        
        print(f"\n🔍 Kernel Görselleştirme:")
        print(f"   📏 Kernel boyutları: {kernel_sizes}")
        print(f"   📊 Sigma değerleri: {sigma_values}")
        
        for ksize in kernel_sizes:
            for sigma in sigma_values:
                # OpenCV ile kernel oluştur
                kernel = cv2.getGaussianKernel(ksize, sigma)
                kernel_2d = kernel * kernel.T
                
                # Görselleştir
                plt.figure(figsize=(8, 6))
                plt.imshow(kernel_2d, cmap='viridis', interpolation='nearest')
                plt.colorbar(label='Ağırlık')
                plt.title(f'Gaussian Kernel: {ksize}x{ksize}, σ={sigma}')
                plt.xlabel('X')
                plt.ylabel('Y')
                
                # Kaydet
                save_path = f"{self.results_dir}/kernel_visualizations/kernel_{ksize}x{ksize}_sigma_{sigma}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Kernel {ksize}x{ksize}, σ={sigma} -> {save_path}")
                
                # Kernel değerlerini yazdır
                print(f"      📊 Kernel değerleri (merkez): {kernel_2d[ksize//2, ksize//2]:.4f}")
                print(f"      📊 Toplam ağırlık: {np.sum(kernel_2d):.4f}")
        
    def demo_basic_gaussian_blur(self):
        """
        Temel Gaussian Blur işlemlerini gösterir
        """
        print("\n" + "="*60)
        print("3. TEMEL GAUSSIAN BLUR İŞLEMLERİ")
        print("="*60)
        
        # Görüntü yoksa örnek oluştur
        if not os.path.exists(self.image_path):
            self.create_sample_image()
        
        # Görüntüyü oku
        image = cv2.imread(self.image_path)
        if image is None:
            print("❌ Görüntü okunamadı!")
            return None
        
        print(f"\n📖 Orijinal görüntü boyutu: {image.shape}")
        
        # Farklı kernel boyutları ile deneme
        kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9), (15, 15)]
        
        print(f"\n🔄 Kernel Boyutu Etkisi (σ=0 - otomatik):")
        
        for ksize in kernel_sizes:
            # Gaussian Blur uygula
            blurred = cv2.GaussianBlur(image, ksize, 0)
            
            # Kaydet
            save_path = f"{self.results_dir}/blur_effects/kernel_{ksize[0]}x{ksize[1]}_auto_sigma.jpg"
            cv2.imwrite(save_path, blurred)
            
            print(f"   ✅ Kernel {ksize[0]}x{ksize[1]} -> {save_path}")
        
        return image
    
    def demo_sigma_effects(self, image):
        """
        Sigma değerinin etkisini gösterir
        """
        print("\n" + "="*60)
        print("4. SIGMA DEĞERİNİN ETKİSİ")
        print("="*60)
        
        # Sabit kernel boyutu, farklı sigma değerleri
        ksize = (9, 9)
        sigma_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        
        print(f"\n🔄 Sigma Etkisi (Kernel: {ksize[0]}x{ksize[1]}):")
        
        for sigma in sigma_values:
            # Gaussian Blur uygula
            blurred = cv2.GaussianBlur(image, ksize, sigma)
            
            # Kaydet
            save_path = f"{self.results_dir}/blur_effects/sigma_{sigma}_kernel_{ksize[0]}x{ksize[1]}.jpg"
            cv2.imwrite(save_path, blurred)
            
            print(f"   ✅ Sigma={sigma} -> {save_path}")
        
    def demo_parameter_combinations(self, image):
        """
        Farklı parametre kombinasyonlarını gösterir
        """
        print("\n" + "="*60)
        print("5. PARAMETRE KOMBİNASYONLARI")
        print("="*60)
        
        # Farklı kombinasyonlar
        combinations = [
            ((3, 3), 0.5, "Hafif Bulaniklaştirma"),
            ((5, 5), 1.0, "Orta Bulaniklaştirma"),
            ((7, 7), 1.5, "Güçlü Bulaniklaştirma"),
            ((9, 9), 2.0, "Çok Güçlü Bulaniklaştirma"),
            ((15, 15), 3.0, "Aşırı Bulaniklaştirma"),
            ((21, 21), 5.0, "Maksimum Bulaniklaştirma")
        ]
        
        print(f"\n🔄 Parametre Kombinasyonları:")
        
        for ksize, sigma, description in combinations:
            # Gaussian Blur uygula
            blurred = cv2.GaussianBlur(image, ksize, sigma)
            
            # Kaydet
            filename = f"{description.replace(' ', '_').lower()}_kernel_{ksize[0]}x{ksize[1]}_sigma_{sigma}.jpg"
            save_path = f"{self.results_dir}/blur_effects/{filename}"
            cv2.imwrite(save_path, blurred)
            
            print(f"   ✅ {description} -> {save_path}")
    

    
    def demo_edge_preservation(self, image):
        """
        Kenar koruma özelliğini gösterir
        """
        print("\n" + "="*60)
        print("7. KENAR KORUMA ÖZELLİĞİ")
        print("="*60)
        
        print(f"\n🔍 Gaussian Blur vs Diğer Filtreler:")
        print(f"   💡 Gaussian Blur kenarları daha yumuşak şekilde bulanıklaştırır")
        print(f"   💡 Ortalama filtre tüm pikselleri eşit ağırlıklandırır")
        
        # Gaussian Blur
        gaussian_blur = cv2.GaussianBlur(image, (15, 15), 2.0)
        
        # Ortalama filtre (karşılaştırma için)
        average_blur = cv2.blur(image, (15, 15))
        
        # Median filtre (karşılaştırma için)
        median_blur = cv2.medianBlur(image, 15)
        
        # Kaydet
        cv2.imwrite(f"{self.results_dir}/comparisons/gaussian_vs_average.jpg", np.hstack([gaussian_blur, average_blur]))
        cv2.imwrite(f"{self.results_dir}/comparisons/gaussian_vs_median.jpg", np.hstack([gaussian_blur, median_blur]))
        
        print(f"   ✅ Gaussian vs Ortalama -> gaussian_vs_average.jpg")
        print(f"   ✅ Gaussian vs Median -> gaussian_vs_median.jpg")
        
        # Kenar detaylarını göster
        # Keskin kenarlı bir bölge seç
        roi = image[50:150, 50:150]  # Mavi dikdörtgen bölgesi
        
        # Bu bölgeye farklı filtreler uygula
        roi_gaussian = cv2.GaussianBlur(roi, (15, 15), 2.0)
        roi_average = cv2.blur(roi, (15, 15))
        
        # Karşılaştırma görüntüsü oluştur
        comparison = np.hstack([roi, roi_gaussian, roi_average])
        cv2.imwrite(f"{self.results_dir}/comparisons/edge_preservation_comparison.jpg", comparison)
        
        print(f"   ✅ Kenar koruma karşılaştırması -> edge_preservation_comparison.jpg")
    
    def run_demo(self):
        """
        Tüm demo'ları çalıştırır
        """
        print("🚀 Gaussian Blur Demo Başlıyor...")
        print("="*60)
        
        try:
            # 1. Teori açıklaması
            self.demo_gaussian_theory()
            
            # 2. Kernel görselleştirme
            self.demo_kernel_visualization()
            
            # 3. Temel Gaussian Blur
            image = self.demo_basic_gaussian_blur()
            if image is None:
                print("❌ Demo başlatılamadı!")
                return
            
            # 4. Sigma etkisi
            self.demo_sigma_effects(image)
            
            # 5. Parametre kombinasyonları
            self.demo_parameter_combinations(image)
            
            # 6. Kenar koruma
            self.demo_edge_preservation(image)
            
            print("\n" + "="*60)
            print("🎉 Gaussian Blur Demo tamamlandı!")
            print("📁 Sonuçlar 'results' klasöründe bulunabilir")
            print("📚 README.md dosyasını inceleyerek öğrenilenleri gözden geçirin")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    # Demo'yu başlat
    demo = GaussianBlurDemo()
    demo.run_demo() 