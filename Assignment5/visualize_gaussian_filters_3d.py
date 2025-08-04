"""
3D Gaussian Filter Visualization
Assignment 5: Custom Gaussian Blur - 3D Visualization

Bu script farklı kernel boyutları ve sigma değerleri için Gaussian filtrelerini
3 boyutlu olarak görselleştirir.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy import ndimage
import os

class GaussianFilter3DVisualizer:
    """
    Gaussian filtrelerini 3D olarak görselleştiren sınıf
    """
    
    def __init__(self):
        """Sınıfı başlatır ve gerekli klasörleri oluşturur"""
        self.create_directories()
        self.results_dir = "results/3d_visualization"
        
    def create_directories(self):
        """Gerekli klasörleri oluşturur"""
        directories = [
            "results",
            "results/3d_visualization",
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_gaussian_kernel_2d(self, kernel_size, sigma):
        """
        2D Gaussian kernel oluşturur
        """
        # Kernel merkezi
        center = kernel_size // 2
        
        # X ve Y koordinatları
        x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
        
        # Gaussian formülü
        kernel = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        
        # Normalize et
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def create_gaussian_kernel_1d(self, kernel_size, sigma):
        """
        1D Gaussian kernel oluşturur (separable convolution için)
        """
        center = kernel_size // 2
        x = np.arange(kernel_size)
        kernel = np.exp(-((x - center)**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def calculate_sigma_from_kernel_size(self, kernel_size):
        """
        Kernel boyutuna göre sigma değerini hesaplar (OpenCV formülü)
        """
        return 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    def visualize_single_gaussian_3d(self, kernel_size, sigma, title, filename):
        """
        Tek bir Gaussian filtresini 3D olarak görselleştirir
        """
        # Kernel oluştur
        kernel = self.create_gaussian_kernel_2d(kernel_size, sigma)
        
        # 3D plot oluştur
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # X ve Y koordinatları
        x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
        
        # 3D surface plot
        surf = ax.plot_surface(x, y, kernel, cmap='viridis', 
                              linewidth=0, antialiased=True)
        
        # Görselleştirme ayarları
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Weight')
        ax.set_title(f'{title}\nKernel Size: {kernel_size}x{kernel_size}, Sigma: {sigma:.2f}')
        
        # Colorbar ekle
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Kaydet
        plt.savefig(f"{self.results_dir}/{filename}.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ {title} -> {filename}.jpg")
        
        return kernel
    
    def visualize_multiple_gaussians_comparison(self):
        """
        Farklı parametrelerle Gaussian filtrelerini karşılaştırmalı olarak gösterir
        """
        print("\n" + "="*60)
        print("3D GAUSSIAN FILTER VISUALIZATION")
        print("="*60)
        
        # Farklı parametreler
        configurations = [
            (3, 0.5, "Small Kernel, Low Sigma"),
            (5, 1.0, "Medium Kernel, Medium Sigma"),
            (7, 1.5, "Large Kernel, High Sigma"),
            (9, 2.0, "Very Large Kernel, Very High Sigma"),
            (11, 2.5, "Extra Large Kernel, Extra High Sigma")
        ]
        
        kernels = []
        titles = []
        
        for kernel_size, sigma, title in configurations:
            kernel = self.visualize_single_gaussian_3d(
                kernel_size, sigma, title, f"gaussian_{kernel_size}x{kernel_size}_sigma_{sigma}"
            )
            kernels.append(kernel)
            titles.append(f"{kernel_size}x{kernel_size}, σ={sigma}")
        
        # Karşılaştırmalı görselleştirme
        self.create_comparison_plot(kernels, titles)
        
        return kernels
    
    def create_comparison_plot(self, kernels, titles):
        """
        Farklı Gaussian filtrelerini yan yana karşılaştırır
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (kernel, title) in enumerate(zip(kernels, titles)):
            if i < len(axes):
                ax = axes[i]
                
                # 2D heatmap
                im = ax.imshow(kernel, cmap='viridis', interpolation='nearest')
                ax.set_title(f'Gaussian Kernel\n{title}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                
                # Colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Son subplot'u gizle (5 kernel için 6 subplot var)
        if len(kernels) < len(axes):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/gaussian_kernels_comparison_2d.jpg", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Gaussian Kernels Comparison (2D) -> gaussian_kernels_comparison_2d.jpg")
    
    def visualize_separable_gaussian_3d(self):
        """
        Separable Gaussian filtresini 3D olarak gösterir
        """
        print(f"\n🔧 Separable Gaussian Filter Visualization:")
        
        kernel_size = 7
        sigma = 1.5
        
        # 1D kernel'lar
        kernel_1d_x = self.create_gaussian_kernel_1d(kernel_size, sigma)
        kernel_1d_y = kernel_1d_x  # Aynı kernel
        
        # 2D kernel (1D kernel'ların çarpımı)
        kernel_2d = np.outer(kernel_1d_y, kernel_1d_x)
        
        # 3D görselleştirme
        fig = plt.figure(figsize=(15, 5))
        
        # 1D X kernel
        ax1 = fig.add_subplot(131, projection='3d')
        x = np.arange(kernel_size)
        y = np.zeros_like(x)
        ax1.plot(x, y, kernel_1d_x, 'b-', linewidth=3, label='X Kernel')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Weight')
        ax1.set_title('1D Gaussian Kernel (X)')
        
        # 1D Y kernel
        ax2 = fig.add_subplot(132, projection='3d')
        x = np.zeros_like(kernel_1d_y)
        y = np.arange(kernel_size)
        ax2.plot(x, y, kernel_1d_y, 'r-', linewidth=3, label='Y Kernel')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Weight')
        ax2.set_title('1D Gaussian Kernel (Y)')
        
        # 2D kernel (separable)
        ax3 = fig.add_subplot(133, projection='3d')
        x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
        surf = ax3.plot_surface(x, y, kernel_2d, cmap='viridis', 
                               linewidth=0, antialiased=True)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Weight')
        ax3.set_title('2D Separable Gaussian Kernel')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/separable_gaussian_3d.jpg", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Separable Gaussian 3D -> separable_gaussian_3d.jpg")
    
    def visualize_sigma_effect_3d(self):
        """
        Farklı sigma değerlerinin etkisini 3D olarak gösterir
        """
        print(f"\n🔧 Sigma Effect Visualization:")
        
        kernel_size = 9
        sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        for i, sigma in enumerate(sigma_values):
            ax = axes[i]
            
            # Kernel oluştur
            kernel = self.create_gaussian_kernel_2d(kernel_size, sigma)
            x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
            
            # 3D surface plot
            surf = ax.plot_surface(x, y, kernel, cmap='viridis', 
                                  linewidth=0, antialiased=True)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Weight')
            ax.set_title(f'Gaussian Kernel\nσ = {sigma}')
        
        # Son subplot'u gizle
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/sigma_effect_3d.jpg", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Sigma Effect 3D -> sigma_effect_3d.jpg")
    
    def visualize_kernel_size_effect_3d(self):
        """
        Farklı kernel boyutlarının etkisini 3D olarak gösterir
        """
        print(f"\n🔧 Kernel Size Effect Visualization:")
        
        kernel_sizes = [3, 5, 7, 9, 11]
        sigma = 1.0
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        for i, kernel_size in enumerate(kernel_sizes):
            ax = axes[i]
            
            # Kernel oluştur
            kernel = self.create_gaussian_kernel_2d(kernel_size, sigma)
            x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
            
            # 3D surface plot
            surf = ax.plot_surface(x, y, kernel, cmap='viridis', 
                                  linewidth=0, antialiased=True)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Weight')
            ax.set_title(f'Gaussian Kernel\n{kernel_size}x{kernel_size}, σ = {sigma}')
        
        # Son subplot'u gizle
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/kernel_size_effect_3d.jpg", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Kernel Size Effect 3D -> kernel_size_effect_3d.jpg")
    
    def create_animated_gaussian_3d(self):
        """
        Gaussian filtresinin oluşumunu animasyonlu olarak gösterir
        """
        print(f"\n🔧 Animated Gaussian Formation:")
        
        kernel_size = 7
        sigma = 1.5
        
        # Farklı aşamalar
        stages = [
            (0.5, "Early Stage"),
            (1.0, "Middle Stage"),
            (1.5, "Final Stage")
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
        
        for i, (stage_sigma, title) in enumerate(stages):
            ax = axes[i]
            
            # Kernel oluştur
            kernel = self.create_gaussian_kernel_2d(kernel_size, stage_sigma)
            x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
            
            # 3D surface plot
            surf = ax.plot_surface(x, y, kernel, cmap='viridis', 
                                  linewidth=0, antialiased=True)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Weight')
            ax.set_title(f'{title}\nσ = {stage_sigma}')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/animated_gaussian_formation.jpg", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Animated Gaussian Formation -> animated_gaussian_formation.jpg")
    
    def run_visualization(self):
        """
        Tüm 3D görselleştirmeleri çalıştırır
        """
        print("🚀 3D Gaussian Filter Visualization Başlıyor...")
        print("="*60)
        
        try:
            # 1. Tek Gaussian filtreleri
            kernels = self.visualize_multiple_gaussians_comparison()
            
            # 2. Separable Gaussian
            self.visualize_separable_gaussian_3d()
            
            # 3. Sigma etkisi
            self.visualize_sigma_effect_3d()
            
            # 4. Kernel boyutu etkisi
            self.visualize_kernel_size_effect_3d()
            
            # 5. Animasyonlu oluşum
            self.create_animated_gaussian_3d()
            
            print("\n" + "="*60)
            print("🎉 3D Gaussian Filter Visualization tamamlandı!")
            print("📁 Sonuçlar 'results/3d_visualization' klasöründe bulunabilir")
            print("📚 README.md dosyasını inceleyerek öğrenilenleri gözden geçirin")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    # 3D görselleştirmeyi başlat
    visualizer = GaussianFilter3DVisualizer()
    visualizer.run_visualization() 