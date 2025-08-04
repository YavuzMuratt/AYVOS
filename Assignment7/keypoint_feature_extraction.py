"""
Keypoint ve Özellik Çıkarımı
Assignment 7: SIFT, ORB, SURF, FAST gibi algoritmalar ile anlamlı noktaların tespiti

Bu dosya keypoint detection, feature extraction ve matching algoritmalarını öğretmek için hazırlanmıştır.
SIFT, ORB, SURF, FAST gibi popüler algoritmaları kapsamlı şekilde ele alır.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class KeypointFeatureExtraction:
    """
    Keypoint detection ve feature extraction algoritmalarını gösteren sınıf
    """
    
    def __init__(self):
        """Sınıfı başlatır ve gerekli klasörleri oluşturur"""
        self.create_directories()
        self.image_path1 = "images/image1.jpg"
        self.image_path2 = "images/image2.jpg"
        self.results_dir = "results"
        
    def create_directories(self):
        """Gerekli klasörleri oluşturur"""
        directories = [
            "images",
            "results",
            "results/keypoints", 
            "results/matching",
            "results/applications",
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_sample_images(self):
        """
        Demo için örnek görüntüler oluşturur
        """
        # İlk görüntü - karmaşık desenler
        image1 = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Farklı şekiller ve desenler
        cv2.rectangle(image1, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.circle(image1, (300, 100), 50, (0, 255, 0), -1)
        cv2.ellipse(image1, (200, 300), (60, 30), 45, 0, 360, (0, 0, 255), -1)
        
        # İnce çizgiler ve köşeler
        for i in range(0, 400, 30):
            cv2.line(image1, (i, 0), (i+20, 50), (255, 255, 255), 2)
        
        # Gürültü ekle
        noise = np.random.normal(0, 15, image1.shape).astype(np.uint8)
        image1 = cv2.add(image1, noise)
        
        # Metin ekle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image1, "Keypoint Detection", (20, 380), font, 0.6, (255, 255, 255), 2)
        
        # İkinci görüntü - ilkinin döndürülmüş ve ölçeklenmiş hali
        image2 = cv2.resize(image1, (300, 300))
        rows, cols = image2.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
        image2 = cv2.warpAffine(image2, M, (cols, rows))
        
        # Gürültü ekle
        noise2 = np.random.normal(0, 20, image2.shape).astype(np.uint8)
        image2 = cv2.add(image2, noise2)
        
        # Görüntüleri kaydet
        cv2.imwrite(self.image_path1, image1)
        cv2.imwrite(self.image_path2, image2)
        print(f"✅ Örnek görüntüler oluşturuldu: {self.image_path1}, {self.image_path2}")
    
    def demo_keypoint_detection(self):
        """
        Farklı keypoint detection algoritmalarını gösterir
        """
        print("\n" + "="*60)
        print("1. KEYPOINT DETECTION ALGORİTMALARI")
        print("="*60)
        
        # Görüntü yoksa oluştur
        if not os.path.exists(self.image_path1):
            self.create_sample_images()
        
        # Görüntüyü oku
        image = cv2.imread(self.image_path1)
        if image is None:
            print("❌ Görüntü okunamadı!")
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print(f"\n📖 Orijinal görüntü boyutu: {gray.shape}")
        
        # 1. Harris Corner Detection
        print(f"\n🔧 1. Harris Corner Detection:")
        harris = cv2.cornerHarris(gray, 2, 3, 0.04)
        harris = cv2.dilate(harris, None)
        
        # Keypoint'leri görselleştir
        harris_img = image.copy()
        harris_img[harris > 0.01 * harris.max()] = [0, 0, 255]
        
        cv2.imwrite(f"{self.results_dir}/keypoints/harris_corners.jpg", harris_img)
        print(f"   ✅ Harris Corners -> harris_corners.jpg")
        
        # 2. SIFT Keypoint Detection
        print(f"\n🔧 2. SIFT Keypoint Detection:")
        sift = cv2.SIFT_create()
        sift_keypoints, sift_descriptors = sift.detectAndCompute(gray, None)
        
        # Keypoint'leri görselleştir
        sift_img = cv2.drawKeypoints(image, sift_keypoints, None, 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(f"{self.results_dir}/keypoints/sift_keypoints.jpg", sift_img)
        print(f"   ✅ SIFT Keypoints ({len(sift_keypoints)}) -> sift_keypoints.jpg")
        
        # 3. ORB Keypoint Detection
        print(f"\n🔧 3. ORB Keypoint Detection:")
        orb = cv2.ORB_create(nfeatures=500)
        orb_keypoints, orb_descriptors = orb.detectAndCompute(gray, None)
        
        # Keypoint'leri görselleştir
        orb_img = cv2.drawKeypoints(image, orb_keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(f"{self.results_dir}/keypoints/orb_keypoints.jpg", orb_img)
        print(f"   ✅ ORB Keypoints ({len(orb_keypoints)}) -> orb_keypoints.jpg")
        
        # 4. FAST Keypoint Detection
        print(f"\n🔧 4. FAST Keypoint Detection:")
        fast = cv2.FastFeatureDetector_create(threshold=25)
        fast_keypoints = fast.detect(gray, None)
        
        # Keypoint'leri görselleştir
        fast_img = cv2.drawKeypoints(image, fast_keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(f"{self.results_dir}/keypoints/fast_keypoints.jpg", fast_img)
        print(f"   ✅ FAST Keypoints ({len(fast_keypoints)}) -> fast_keypoints.jpg")
        
        # 5. SURF Keypoint Detection (OpenCV 4.5.4+ gerekli)
        print(f"\n🔧 5. SURF Keypoint Detection:")
        try:
            surf = cv2.xfeatures2d.SURF_create(400)
            surf_keypoints, surf_descriptors = surf.detectAndCompute(gray, None)
            
            # Keypoint'leri görselleştir
            surf_img = cv2.drawKeypoints(image, surf_keypoints, None, 
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(f"{self.results_dir}/keypoints/surf_keypoints.jpg", surf_img)
            print(f"   ✅ SURF Keypoints ({len(surf_keypoints)}) -> surf_keypoints.jpg")
        except:
            print(f"   ⚠️ SURF mevcut değil (OpenCV contrib gerekli)")
        
        # Karşılaştırma görüntüsü
        comparison = np.hstack([image, harris_img, sift_img, orb_img, fast_img])
        cv2.imwrite(f"{self.results_dir}/keypoints/keypoint_comparison.jpg", comparison)
        print(f"   ✅ Karşılaştırma -> keypoint_comparison.jpg")
        
        return gray, sift_keypoints, sift_descriptors, orb_keypoints, orb_descriptors
    
    def demo_feature_matching(self, gray1, sift_kp1, sift_desc1, orb_kp1, orb_desc1):
        """
        Feature matching algoritmalarını gösterir
        """
        print("\n" + "="*60)
        print("2. FEATURE MATCHING ALGORİTMALARI")
        print("="*60)
        
        # İkinci görüntüyü oku
        image2 = cv2.imread(self.image_path2)
        if image2 is None:
            print("❌ İkinci görüntü okunamadı!")
            return None
        
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        print(f"\n📖 İkinci görüntü boyutu: {gray2.shape}")
        
        # 1. SIFT Feature Matching
        print(f"\n🔧 1. SIFT Feature Matching:")
        sift = cv2.SIFT_create()
        sift_kp2, sift_desc2 = sift.detectAndCompute(gray2, None)
        
        # Brute Force Matcher
        bf = cv2.BFMatcher()
        sift_matches = bf.knnMatch(sift_desc1, sift_desc2, k=2)
        
        # Lowe's ratio test
        good_sift = []
        for match_pair in sift_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_sift.append(m)
        
        # Eşleştirmeleri görselleştir
        image1 = cv2.imread(self.image_path1)
        sift_matching_img = cv2.drawMatches(image1, sift_kp1, image2, sift_kp2, 
                                           good_sift, None, 
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"{self.results_dir}/matching/sift_matching.jpg", sift_matching_img)
        print(f"   ✅ SIFT Matching ({len(good_sift)} matches) -> sift_matching.jpg")
        
        # 2. ORB Feature Matching
        print(f"\n🔧 2. ORB Feature Matching:")
        orb = cv2.ORB_create(nfeatures=500)
        orb_kp2, orb_desc2 = orb.detectAndCompute(gray2, None)
        
        # Hamming distance için BF Matcher
        bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        orb_matches = bf_orb.match(orb_desc1, orb_desc2)
        
        # En iyi eşleştirmeleri seç
        orb_matches = sorted(orb_matches, key=lambda x: x.distance)
        good_orb = orb_matches[:int(len(orb_matches) * 0.75)]
        
        # Eşleştirmeleri görselleştir
        orb_matching_img = cv2.drawMatches(image1, orb_kp1, image2, orb_kp2, 
                                          good_orb, None, 
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"{self.results_dir}/matching/orb_matching.jpg", orb_matching_img)
        print(f"   ✅ ORB Matching ({len(good_orb)} matches) -> orb_matching.jpg")
        
        # 3. FLANN Matching (SIFT için)
        print(f"\n🔧 3. FLANN Matching:")
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        flann_matches = flann.knnMatch(sift_desc1, sift_desc2, k=2)
        
        # Lowe's ratio test
        good_flann = []
        for match_pair in flann_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_flann.append(m)
        
        # Eşleştirmeleri görselleştir
        flann_matching_img = cv2.drawMatches(image1, sift_kp1, image2, sift_kp2, 
                                            good_flann, None, 
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"{self.results_dir}/matching/flann_matching.jpg", flann_matching_img)
        print(f"   ✅ FLANN Matching ({len(good_flann)} matches) -> flann_matching.jpg")
        
        return gray2, sift_kp2, orb_kp2, good_sift, good_orb, good_flann
    
    def demo_homography_and_applications(self, gray1, gray2, sift_kp1, sift_kp2, 
                                       orb_kp1, orb_kp2, good_sift, good_orb):
        """
        Homography hesaplama ve gerçek dünya uygulamalarını gösterir
        """
        print("\n" + "="*60)
        print("3. HOMOGRAPHY VE UYGULAMALAR")
        print("="*60)
        
        # 1. Homography Estimation (SIFT ile)
        print(f"\n🔧 1. Homography Estimation:")
        if len(good_sift) >= 4:
            # Keypoint'lerin koordinatlarını al
            src_pts = np.float32([sift_kp1[m.queryIdx].pt for m in good_sift]).reshape(-1, 1, 2)
            dst_pts = np.float32([sift_kp2[m.trainIdx].pt for m in good_sift]).reshape(-1, 1, 2)
            
            # Homography hesapla
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                # Perspektif dönüşüm uygula
                h, w = gray1.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, H)
                
                # Dönüştürülmüş görüntüyü çiz
                image1 = cv2.imread(self.image_path1)
                image2 = cv2.imread(self.image_path2)
                transformed_img = cv2.polylines(image2.copy(), [np.int32(dst)], True, (0, 255, 0), 3)
                
                cv2.imwrite(f"{self.results_dir}/applications/homography_result.jpg", transformed_img)
                print(f"   ✅ Homography Result -> homography_result.jpg")
        
        # 2. Image Stitching (Basit panorama)
        #print(f"\n🔧 2. Image Stitching:")
        #if len(good_sift) >= 10:
            # Warp perspektif
            #h1, w1 = gray1.shape
            #h2, w2 = gray2.shape
            
            # İkinci görüntüyü birinci görüntünün koordinat sistemine dönüştür
            #warped_img = cv2.warpPerspective(image2, H, (w1 + w2, h1))
            
            # Görüntüleri birleştir
            #warped_img[0:h1, 0:w1] = image1
            
            #cv2.imwrite(f"{self.results_dir}/applications/image_stitching.jpg", warped_img)
            #print(f"   ✅ Image Stitching -> image_stitching.jpg")
        
        # 3. Object Detection (Template matching ile)
        print(f"\n🔧 3. Object Detection:")
        # İlk görüntüden bir bölgeyi template olarak al
        template = gray1[100:200, 100:200]
        
        # Template matching
        result = cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Eşleşen bölgeyi işaretle
        h, w = template.shape
        image2_with_detection = cv2.imread(self.image_path2)
        cv2.rectangle(image2_with_detection, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
        
        cv2.imwrite(f"{self.results_dir}/applications/object_detection.jpg", image2_with_detection)
        print(f"   ✅ Object Detection -> object_detection.jpg")
        
        # 4. Feature Density Analysis
        print(f"\n🔧 4. Feature Density Analysis:")
        # Keypoint yoğunluğunu analiz et
        sift_density = len(sift_kp1) / (gray1.shape[0] * gray1.shape[1])
        orb_density = len(orb_kp1) / (gray1.shape[0] * gray1.shape[1])
        
        # Yoğunluk haritası oluştur
        density_map = np.zeros_like(gray1)
        for kp in sift_kp1:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(density_map, (x, y), 5, 255, -1)
        
        # Gaussian blur ile yumuşat
        density_map = cv2.GaussianBlur(density_map, (15, 15), 0)
        
        # Heatmap olarak kaydet
        plt.figure(figsize=(8, 6))
        plt.imshow(density_map, cmap='hot')
        plt.colorbar()
        plt.title('Feature Density Map')
        plt.savefig(f"{self.results_dir}/applications/feature_density_map.jpg")
        plt.close()
        
        print(f"   ✅ Feature Density Map -> feature_density_map.jpg")
        print(f"   📊 SIFT Density: {sift_density:.6f}, ORB Density: {orb_density:.6f}")
    
    def demo_performance_comparison(self, gray1, gray2):
        """
        Farklı algoritmaların performansını karşılaştırır
        """
        print("\n" + "="*60)
        print("4. PERFORMANS KARŞILAŞTIRMASI")
        print("="*60)
        
        algorithms = {
            'SIFT': cv2.SIFT_create(),
            'ORB': cv2.ORB_create(nfeatures=500),
            'FAST': cv2.FastFeatureDetector_create(threshold=25)
        }
        
        results = {}
        
        for name, detector in algorithms.items():
            print(f"\n🔧 {name} Performance:")
            
            # Keypoint detection süresi
            start_time = cv2.getTickCount()
            if name == 'FAST':
                keypoints = detector.detect(gray1, None)
                descriptors = None
            else:
                keypoints, descriptors = detector.detectAndCompute(gray1, None)
            end_time = cv2.getTickCount()
            
            detection_time = (end_time - start_time) / cv2.getTickFrequency()
            
            # Matching süresi (descriptor varsa)
            matching_time = 0
            match_count = 0
            if descriptors is not None:
                # İkinci görüntüden de özellikler çıkar
                kp2, desc2 = detector.detectAndCompute(gray2, None)
                
                if desc2 is not None and len(desc2) > 0:
                    # Matching
                    if name == 'ORB':
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    else:
                        matcher = cv2.BFMatcher()
                    
                    start_time = cv2.getTickCount()
                    matches = matcher.match(descriptors, desc2)
                    end_time = cv2.getTickCount()
                    
                    matching_time = (end_time - start_time) / cv2.getTickFrequency()
                    match_count = len(matches)
            
            results[name] = {
                'keypoints': len(keypoints),
                'detection_time': detection_time,
                'matching_time': matching_time,
                'matches': match_count
            }
            
            print(f"   📊 Keypoints: {len(keypoints)}")
            print(f"   ⏱️ Detection Time: {detection_time:.4f}s")
            print(f"   ⏱️ Matching Time: {matching_time:.4f}s")
            print(f"   🔗 Matches: {match_count}")
        
        # Sonuçları görselleştir
        self.plot_performance_results(results)
        
        return results
    
    def plot_performance_results(self, results):
        """
        Performans sonuçlarını görselleştirir
        """
        names = list(results.keys())
        keypoint_counts = [results[name]['keypoints'] for name in names]
        detection_times = [results[name]['detection_time'] for name in names]
        matching_times = [results[name]['matching_time'] for name in names]
        match_counts = [results[name]['matches'] for name in names]
        
        # Çoklu grafik oluştur
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Keypoint sayısı
        ax1.bar(names, keypoint_counts, color=['red', 'blue', 'green'])
        ax1.set_title('Keypoint Count')
        ax1.set_ylabel('Number of Keypoints')
        
        # Detection süresi
        ax2.bar(names, detection_times, color=['red', 'blue', 'green'])
        ax2.set_title('Detection Time')
        ax2.set_ylabel('Time (seconds)')
        
        # Matching süresi
        ax3.bar(names, matching_times, color=['red', 'blue', 'green'])
        ax3.set_title('Matching Time')
        ax3.set_ylabel('Time (seconds)')
        
        # Match sayısı
        ax4.bar(names, match_counts, color=['red', 'blue', 'green'])
        ax4.set_title('Match Count')
        ax4.set_ylabel('Number of Matches')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/applications/performance_comparison.jpg")
        plt.close()
        
        print(f"   ✅ Performance Comparison -> performance_comparison.jpg")
    
    def run_demo(self):
        """
        Tüm demo'ları çalıştırır
        """
        print("🚀 Keypoint ve Özellik Çıkarımı Demo Başlıyor...")
        print("="*60)
        
        try:
            # 1. Keypoint detection
            result = self.demo_keypoint_detection()
            if result is None:
                print("❌ Demo başlatılamadı!")
                return
            
            gray1, sift_kp1, sift_desc1, orb_kp1, orb_desc1 = result
            
            # 2. Feature matching
            result2 = self.demo_feature_matching(gray1, sift_kp1, sift_desc1, orb_kp1, orb_desc1)
            if result2 is None:
                print("❌ Feature matching başlatılamadı!")
                return
            
            gray2, sift_kp2, orb_kp2, good_sift, good_orb, good_flann = result2
            
            # 3. Homography ve uygulamalar
            self.demo_homography_and_applications(gray1, gray2, sift_kp1, sift_kp2, 
                                                orb_kp1, orb_kp2, good_sift, good_orb)
            
            # 4. Performans karşılaştırması
            self.demo_performance_comparison(gray1, gray2)
            
            print("\n" + "="*60)
            print("🎉 Keypoint ve Özellik Çıkarımı Demo tamamlandı!")
            print("📁 Sonuçlar 'results' klasöründe bulunabilir")
            print("📚 README.md dosyasını inceleyerek öğrenilenleri gözden geçirin")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    # Demo'yu başlat
    demo = KeypointFeatureExtraction()
    demo.run_demo() 