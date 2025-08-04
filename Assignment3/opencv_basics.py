"""
OpenCV Temel Fonksiyonları Demo
Assignment 3: cv2.imread, cv2.imshow, cv2.resize

Bu dosya OpenCV'nin temel görüntü işleme fonksiyonlarını öğretmek için hazırlanmıştır.
Her fonksiyon detaylı açıklamalar ve örneklerle gösterilmektedir.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class OpenCVBasicsDemo:
    """
    OpenCV temel fonksiyonlarını gösteren demo sınıfı
    """
    
    def __init__(self):
        """Demo sınıfını başlatır ve gerekli klasörleri oluşturur"""
        self.create_directories()
        self.image_path = "images/image1.jpg"
        self.results_dir = "results"
        
    def create_directories(self):
        """Gerekli klasörleri oluşturur"""
        directories = ["images", "results", "results/resized_images", "results/processed_images"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
    def demo_imread(self):
        """
        cv2.imread() fonksiyonunu gösterir
        Farklı okuma modları ve hata yönetimi
        """
        print("\n" + "="*50)
        print("1. cv2.imread() DEMO")
        print("="*50)
        
        # 1. Normal okuma (BGR formatında)
        print("\n Görüntü okuma (BGR formatında):")
        image_bgr = cv2.imread(self.image_path)
        if image_bgr is not None:
            print(f"   ✅ Görüntü başarıyla okundu")
            print(f"   📏 Boyut: {image_bgr.shape}")
            print(f"   🎨 Renk kanalları: {image_bgr.shape[2] if len(image_bgr.shape) == 3 else 1}")
            print(f"   📊 Veri tipi: {image_bgr.dtype}")
        else:
            print("   ❌ Görüntü okunamadı!")
            return None
            
        # 2. Gri tonlama olarak okuma
        print("\n📖 Görüntü okuma (Gri tonlama):")
        image_gray = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        print(f"   ✅ Gri tonlama okundu")
        print(f"   📏 Boyut: {image_gray.shape}")
        print(f"   📊 Veri tipi: {image_gray.dtype}")
        
        # 3. Orijinal format olarak okuma
        print("\n📖 Görüntü okuma (Orijinal format):")
        image_original = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        print(f"   ✅ Orijinal format okundu")
        print(f"   📏 Boyut: {image_original.shape}")
        
        return image_bgr, image_gray, image_original
    
    def demo_imshow(self, image_bgr, image_gray):
        """
        cv2.imshow() fonksiyonunu gösterir
        Pencere yönetimi ve kullanıcı etkileşimi
        """
        print("\n" + "="*50)
        print("2. cv2.imshow() DEMO")
        print("="*50)
        
        print("\n🖼️  Görüntüleri gösterme:")
        print("   💡 Pencereler açılacak. Kapatmak için 'q' tuşuna basın.")
        
        # 1. Renkli görüntü gösterme
        cv2.imshow("Renkli Görüntü (BGR)", image_bgr)
        print("   ✅ Renkli görüntü penceresi açıldı")
        
        # 2. Gri tonlama görüntü gösterme
        cv2.imshow("Gri Tonlama Görüntü", image_gray)
        print("   ✅ Gri tonlama penceresi açıldı")
        
        # 3. Görüntü bilgilerini yazdırma
        print(f"\n📊 Görüntü Bilgileri:")
        print(f"   Renkli görüntü boyutu: {image_bgr.shape}")
        print(f"   Gri görüntü boyutu: {image_gray.shape}")
        print(f"   Piksel değeri (0,0): {image_bgr[0,0]}")
        print(f"   Gri piksel değeri (0,0): {image_gray[0,0]}")
        
        # 4. Kullanıcı etkileşimi
        print("\n⌨️  Kullanıcı Etkileşimi:")
        print("   - 'q' tuşu: Tüm pencereleri kapat")
        print("   - 's' tuşu: Görüntüyü kaydet")
        print("   - Diğer tuşlar: Devam et")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("   👋 Pencereler kapatılıyor...")
                break
            elif key == ord('s'):
                # Görüntüyü kaydet
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"{self.results_dir}/saved_image_{timestamp}.jpg"
                cv2.imwrite(save_path, image_bgr)
                print(f"   💾 Görüntü kaydedildi: {save_path}")
        
        cv2.destroyAllWindows()
        print("   ✅ Tüm pencereler kapatıldı")
    
    def demo_resize(self, image_bgr):
        """
        cv2.resize() fonksiyonunu gösterir
        Farklı interpolasyon yöntemleri ve boyutlandırma teknikleri
        """
        print("\n" + "="*50)
        print("3. cv2.resize() DEMO")
        print("="*50)
        
        original_height, original_width = image_bgr.shape[:2]
        print(f"\n📏 Orijinal boyut: {original_width}x{original_height}")
        
        # Farklı boyutlandırma örnekleri
        resize_examples = [
            ("Küçültme (0.5x)", (original_width//2, original_height//2)),
            ("Büyütme (2x)", (original_width*2, original_height*2)),
            ("Kare format", (300, 300)),
            ("Genişletme (2x genişlik)", (original_width*2, original_height)),
            ("Daraltma (0.5x yükseklik)", (original_width, original_height//2))
        ]
        
        print(f"\n🔄 Boyutlandırma örnekleri (INTER_LINEAR ile):")
        
        for example_name, new_size in resize_examples:
            print(f"\n   📐 {example_name} ({new_size[0]}x{new_size[1]}):")
            
            # Sadece INTER_LINEAR ile boyutlandır
            resized = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_LINEAR)
            
            # Sonucu kaydet
            filename = f"{example_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.jpg"
            save_path = f"{self.results_dir}/resized_images/{filename}"
            cv2.imwrite(save_path, resized)
            
            print(f"      ✅ Boyutlandırıldı: {resized.shape} -> {save_path}")
        
        # Interpolasyon yöntemleri karşılaştırması (sadece küçük örnek)
        print(f"\n🔍 Interpolasyon Yöntemleri Karşılaştırması:")
        small_size = (50, 50)
        large_size = (200, 200)
        
        interpolation_methods = [
            ("INTER_NEAREST", cv2.INTER_NEAREST),
            ("INTER_LINEAR", cv2.INTER_LINEAR),
            ("INTER_CUBIC", cv2.INTER_CUBIC),
            ("INTER_AREA", cv2.INTER_AREA)
        ]
        
        # Küçültme ve büyütme karşılaştırması
        for method_name, method in interpolation_methods:
            # Küçültme
            small = cv2.resize(image_bgr, small_size, interpolation=method)
            # Büyütme
            large = cv2.resize(small, large_size, interpolation=method)
            
            # Kaydet
            save_path = f"{self.results_dir}/resized_images/comparison_{method_name.lower()}.jpg"
            cv2.imwrite(save_path, large)
            print(f"   ✅ {method_name}: Küçültme -> Büyütme -> {save_path}")
        
        print(f"\n💡 Interpolasyon Yöntemleri Hakkında:")
        print("   - INTER_NEAREST: En hızlı, en düşük kalite")
        print("   - INTER_LINEAR: Hızlı, orta kalite")
        print("   - INTER_CUBIC: Yavaş, yüksek kalite")
        print("   - INTER_AREA: Küçültme için en iyi")
    
    def demo_pixel_operations(self, image_bgr):
        """
        Piksel seviyesinde işlemler gösterir
        """
        print("\n" + "="*50)
        print("4. PİKSEL İŞLEMLERİ DEMO")
        print("="*50)
        
        print(f"\n🔍 Piksel İnceleme:")
        print(f"   📏 Görüntü boyutu: {image_bgr.shape}")
        print(f"   🎨 Renk kanalları: {image_bgr.shape[2]}")
        
        # Belirli piksel değerleri
        center_y, center_x = image_bgr.shape[0]//2, image_bgr.shape[1]//2
        center_pixel = image_bgr[center_y, center_x]
        print(f"   🎯 Merkez piksel ({center_x}, {center_y}): BGR{center_pixel}")
        
        # Piksel değiştirme
        print(f"\n🎨 Piksel Değiştirme:")
        # Merkezdeki 10x10 piksel alanını kırmızı yap
        image_modified = image_bgr.copy()
        image_modified[center_y-5:center_y+5, center_x-5:center_x+5] = [0, 0, 255]
        
        # Kaydet
        save_path = f"{self.results_dir}/processed_images/pixel_modified.jpg"
        cv2.imwrite(save_path, image_modified)
        print(f"   ✅ Merkezdeki 10x10 alan kırmızı yapıldı -> {save_path}")
        
        # Renk kanallarını ayırma
        print(f"\n🌈 Renk Kanalları Ayrıştırma:")
        print(f"   💡 Her kanal o rengin yoğunluğunu gösterir (0-255)")
        print(f"   💡 Beyaz = O renk çok güçlü, Siyah = O renk zayıf")
        
        blue_channel = image_bgr[:, :, 0]   # BGR'de 0. kanal = Mavi
        green_channel = image_bgr[:, :, 1]  # BGR'de 1. kanal = Yeşil
        red_channel = image_bgr[:, :, 2]    # BGR'de 2. kanal = Kırmızı
        
        # Her kanalı kaydet
        cv2.imwrite(f"{self.results_dir}/processed_images/blue_channel.jpg", blue_channel)
        cv2.imwrite(f"{self.results_dir}/processed_images/green_channel.jpg", green_channel)
        cv2.imwrite(f"{self.results_dir}/processed_images/red_channel.jpg", red_channel)
        print(f"   ✅ Renk kanalları ayrıldı ve kaydedildi")
        print(f"   📁 Sonuçlar: blue_channel.jpg, green_channel.jpg, red_channel.jpg")
        
        # Renk kanallarını birleştirme örnekleri
        print(f"\n🔗 Renk Kanallarını Birleştirme:")
        print(f"   💡 Farklı kanal kombinasyonları ile renk efektleri oluşturma")
        
        # 1. Sadece mavi ve yeşil kanalları
        blue_green_only = np.zeros_like(image_bgr)
        blue_green_only[:, :, 0] = blue_channel  # Mavi
        blue_green_only[:, :, 1] = green_channel # Yeşil
        # Kırmızı kanal 0 kalır (siyah)
        cv2.imwrite(f"{self.results_dir}/processed_images/blue_green_only.jpg", blue_green_only)
        print(f"   ✅ Sadece mavi+yeşil kanalları: blue_green_only.jpg")
        
        # 2. Sadece mavi ve kırmızı kanalları
        blue_red_only = np.zeros_like(image_bgr)
        blue_red_only[:, :, 0] = blue_channel  # Mavi
        blue_red_only[:, :, 2] = red_channel   # Kırmızı
        # Yeşil kanal 0 kalır (siyah)
        cv2.imwrite(f"{self.results_dir}/processed_images/blue_red_only.jpg", blue_red_only)
        print(f"   ✅ Sadece mavi+kırmızı kanalları: blue_red_only.jpg")
        
        # 3. Sadece yeşil ve kırmızı kanalları
        green_red_only = np.zeros_like(image_bgr)
        green_red_only[:, :, 1] = green_channel # Yeşil
        green_red_only[:, :, 2] = red_channel   # Kırmızı
        # Mavi kanal 0 kalır (siyah)
        cv2.imwrite(f"{self.results_dir}/processed_images/green_red_only.jpg", green_red_only)
        print(f"   ✅ Sadece yeşil+kırmızı kanalları: green_red_only.jpg")
        
        # 4. Sadece mavi kanal (diğerleri 0)
        blue_only = np.zeros_like(image_bgr)
        blue_only[:, :, 0] = blue_channel  # Sadece mavi
        cv2.imwrite(f"{self.results_dir}/processed_images/blue_only.jpg", blue_only)
        print(f"   ✅ Sadece mavi kanal: blue_only.jpg")
        
        # 5. Sadece yeşil kanal (diğerleri 0)
        green_only = np.zeros_like(image_bgr)
        green_only[:, :, 1] = green_channel  # Sadece yeşil
        cv2.imwrite(f"{self.results_dir}/processed_images/green_only.jpg", green_only)
        print(f"   ✅ Sadece yeşil kanal: green_only.jpg")
        
        # 6. Sadece kırmızı kanal (diğerleri 0)
        red_only = np.zeros_like(image_bgr)
        red_only[:, :, 2] = red_channel  # Sadece kırmızı
        cv2.imwrite(f"{self.results_dir}/processed_images/red_only.jpg", red_only)
        print(f"   ✅ Sadece kırmızı kanal: red_only.jpg")
        
        # 7. Kanal sırasını değiştirme (BGR -> RGB)
        bgr_to_rgb = np.zeros_like(image_bgr)
        bgr_to_rgb[:, :, 0] = red_channel    # BGR'de 0. kanal = Kırmızı
        bgr_to_rgb[:, :, 1] = green_channel  # BGR'de 1. kanal = Yeşil
        bgr_to_rgb[:, :, 2] = blue_channel   # BGR'de 2. kanal = Mavi
        cv2.imwrite(f"{self.results_dir}/processed_images/bgr_to_rgb.jpg", bgr_to_rgb)
        print(f"   ✅ BGR -> RGB dönüşümü: bgr_to_rgb.jpg")
        
        # 8. Kanal yoğunluklarını artırma
        enhanced_blue = np.zeros_like(image_bgr)
        enhanced_blue[:, :, 0] = np.clip(blue_channel * 1.5, 0, 255).astype(np.uint8)  # Mavi yoğunluğunu artır
        enhanced_blue[:, :, 1] = green_channel
        enhanced_blue[:, :, 2] = red_channel
        cv2.imwrite(f"{self.results_dir}/processed_images/enhanced_blue.jpg", enhanced_blue)
        print(f"   ✅ Mavi kanal yoğunluğu artırıldı: enhanced_blue.jpg")
        
        return image_modified
    
    def run_demo(self):
        """
        Tüm demo'ları çalıştırır
        """
        print("🚀 OpenCV Temel Fonksiyonları Demo Başlıyor...")
        print("="*60)
        
        try:
            # 1. Görüntü okuma demo'su
            images = self.demo_imread()
            if images is None:
                print("❌ Demo başlatılamadı!")
                return
            
            image_bgr, image_gray, image_original = images
            
            # 2. Görüntü gösterme demo'su
            self.demo_imshow(image_bgr, image_gray)
            
            # 3. Boyutlandırma demo'su
            self.demo_resize(image_bgr)
            
            # 4. Piksel işlemleri demo'su
            self.demo_pixel_operations(image_bgr)
            
            print("\n" + "="*60)
            print("🎉 Demo tamamlandı!")
            print("📁 Sonuçlar 'results' klasöründe bulunabilir")
            print("📚 README.md dosyasını inceleyerek öğrenilenleri gözden geçirin")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    # Demo'yu başlat
    demo = OpenCVBasicsDemo()
    demo.run_demo() 