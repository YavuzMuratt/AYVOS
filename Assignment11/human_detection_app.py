"""
YOLOv8 ile İnsan Tespiti ve Sayımı
Assignment 11: YOLOv8 Human Detection and Counting

Bu uygulama YOLOv8 modeli kullanarak insan tespiti ve sayımı yapar.
Şık bir arayüz ile video ve görüntü dosyalarından tespit gerçekleştirir.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import time
from datetime import datetime
import threading
import json

class HumanDetectionApp:
    """
    YOLOv8 ile insan tespiti ve sayımı yapan ana uygulama sınıfı
    """
    
    def __init__(self):
        """Uygulamayı başlatır ve arayüzü oluşturur"""
        self.root = tk.Tk()
        self.root.title("YOLOv8 İnsan Tespiti ve Sayımı")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # YOLOv8 modeli
        self.model = None
        self.is_detecting = False
        self.current_video = None
        self.video_capture = None
        self.detection_results = []
        
        # İnsan tracking için değişkenler
        self.tracked_humans = {}  # {track_id: {'bbox': [x1,y1,x2,y2], 'frames_missing': 0}}
        self.next_track_id = 0
        self.total_unique_humans = 0
        self.max_frames_missing = 900  # 30 saniye (30 FPS varsayımı) boyunca görünmeyen track'i sil
        
        # Klasörleri oluştur
        self.create_directories()
        
        # Arayüzü oluştur
        self.create_gui()
        
        # Modeli yükle
        self.load_model()
    
    def create_directories(self):
        """Gerekli klasörleri oluşturur"""
        directories = [
            "model",  # Model klasörü
            "videos",
            "images", 
            "results",
            "results/counts",
            "results/analysis"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_gui(self):
        """Kullanıcı arayüzünü oluşturur"""
        # Ana frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Başlık
        title_label = tk.Label(main_frame, text="YOLOv8 İnsan Tespiti ve Sayımı", 
                              font=("Arial", 20, "bold"), fg='white', bg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        # Kontrol paneli
        control_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Model durumu
        self.model_status = tk.StringVar(value="Model Yükleniyor...")
        status_label = tk.Label(control_frame, textvariable=self.model_status, 
                               font=("Arial", 12), fg='white', bg='#34495e')
        status_label.pack(pady=10)
        
        # Butonlar
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(pady=10)
        
        # Video seçme butonu
        self.video_btn = tk.Button(button_frame, text="Video Seç", 
                                  command=self.select_video, 
                                  font=("Arial", 12), bg='#3498db', fg='white',
                                  relief=tk.RAISED, bd=2, padx=20, pady=5)
        self.video_btn.pack(side=tk.LEFT, padx=5)
        
        # Görüntü seçme butonu
        self.image_btn = tk.Button(button_frame, text="Görüntü Seç", 
                                  command=self.select_image, 
                                  font=("Arial", 12), bg='#e74c3c', fg='white',
                                  relief=tk.RAISED, bd=2, padx=20, pady=5)
        self.image_btn.pack(side=tk.LEFT, padx=5)
        
        # Kamera butonu
        self.camera_btn = tk.Button(button_frame, text="Kamera Başlat", 
                                   command=self.start_camera, 
                                   font=("Arial", 12), bg='#27ae60', fg='white',
                                   relief=tk.RAISED, bd=2, padx=20, pady=5)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Durdur butonu
        self.stop_btn = tk.Button(button_frame, text="Durdur", 
                                 command=self.stop_detection, 
                                 font=("Arial", 12), bg='#f39c12', fg='white',
                                 relief=tk.RAISED, bd=2, padx=20, pady=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Ayarlar frame
        settings_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        settings_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Güven eşiği
        tk.Label(settings_frame, text="Güven Eşiği:", font=("Arial", 12), 
                fg='white', bg='#34495e').pack(side=tk.LEFT, padx=10, pady=10)
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(settings_frame, from_=0.01, to=0.99, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL,
                                   resolution=0.01, length=300,
                                   bg='#34495e', fg='white', highlightbackground='#34495e')
        confidence_scale.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Güven eşiği değerini göster
        self.confidence_label = tk.Label(settings_frame, text="0.50", font=("Arial", 10), 
                                        fg='#e74c3c', bg='#34495e')
        self.confidence_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Güven eşiği değiştiğinde label'ı güncelle
        def update_confidence_label(*args):
            self.confidence_label.config(text=f"{self.confidence_var.get():.2f}")
        
        self.confidence_var.trace('w', update_confidence_label)
        
        # Tracking ayarları
        tracking_frame = tk.Frame(settings_frame, bg='#34495e')
        tracking_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        tk.Label(tracking_frame, text="Track Silme Süresi (sn):", font=("Arial", 10), 
                fg='white', bg='#34495e').pack(side=tk.LEFT, padx=5)
        
        self.tracking_time_var = tk.IntVar(value=30)
        tracking_time_scale = tk.Scale(tracking_frame, from_=5, to=60, 
                                      variable=self.tracking_time_var, orient=tk.HORIZONTAL,
                                      resolution=5, length=150,
                                      bg='#34495e', fg='white', highlightbackground='#34495e')
        tracking_time_scale.pack(side=tk.LEFT, padx=5)
        
        # Tracking süresini güncelle
        def update_tracking_time(*args):
            seconds = self.tracking_time_var.get()
            fps = 30  # Varsayılan FPS
            self.max_frames_missing = seconds * fps
        
        self.tracking_time_var.trace('w', update_tracking_time)
        
        # Sonuçlar frame
        results_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video/görüntü alanı
        self.video_label = tk.Label(results_frame, text="Video/Görüntü burada görünecek", 
                                   font=("Arial", 14), fg='white', bg='#34495e')
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # İstatistikler
        stats_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        # İnsan sayısı
        self.human_count_var = tk.StringVar(value="Tespit Edilen İnsan: 0")
        count_label = tk.Label(stats_frame, textvariable=self.human_count_var, 
                              font=("Arial", 14, "bold"), fg='#e74c3c', bg='#34495e')
        count_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # FPS
        self.fps_var = tk.StringVar(value="FPS: 0")
        fps_label = tk.Label(stats_frame, textvariable=self.fps_var, 
                            font=("Arial", 14), fg='#27ae60', bg='#34495e')
        fps_label.pack(side=tk.RIGHT, padx=20, pady=10)
    
    def load_model(self):
        """YOLOv8 modelini yükler"""
        try:
            self.model_status.set("Model Yükleniyor...")
            self.root.update()
            
            # Kendi eğitilmiş modelinizi yükle
            model_path = "model/best.pt"
            
            if not os.path.exists(model_path):
                # Eğer kendi modeliniz yoksa, varsayılan modeli kullan
                self.model = YOLO('yolov8n.pt')
                self.model_status.set("Varsayılan Model Yüklendi ✓")
            else:
                # Kendi eğitilmiş modelinizi yükle
                self.model = YOLO(model_path)
                self.model_status.set("Hazır")
            
        except Exception as e:
            self.model_status.set("Model Yüklenemedi!")
            messagebox.showerror("Hata", f"Model yüklenirken hata oluştu: {str(e)}")
    
    def select_video(self):
        """Video dosyası seçer"""
        if self.model is None:
            messagebox.showerror("Hata", "Önce modeli yükleyin!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_video = file_path
            self.start_video_detection()
    
    def select_image(self):
        """Görüntü dosyası seçer"""
        if self.model is None:
            messagebox.showerror("Hata", "Önce modeli yükleyin!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Görüntü Dosyası Seç",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def start_camera(self):
        """Kamera ile gerçek zamanlı tespit başlatır"""
        if self.model is None:
            messagebox.showerror("Hata", "Önce modeli yükleyin!")
            return
        
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            messagebox.showerror("Hata", "Kamera açılamadı!")
            return
        
        self.start_detection()
    
    def start_video_detection(self):
        """Video dosyasından tespit başlatır"""
        if not self.current_video:
            return
        
        self.video_capture = cv2.VideoCapture(self.current_video)
        if not self.video_capture.isOpened():
            messagebox.showerror("Hata", "Video dosyası açılamadı!")
            return
        
        self.start_detection()
    
    def start_detection(self):
        """Tespit işlemini başlatır"""
        self.is_detecting = True
        self.detection_results = []
        
        # Tracking'i sıfırla
        self.tracked_humans = {}
        self.next_track_id = 0
        self.total_unique_humans = 0
        
        # Buton durumlarını güncelle
        self.video_btn.config(state=tk.DISABLED)
        self.image_btn.config(state=tk.DISABLED)
        self.camera_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Tespit thread'ini başlat
        detection_thread = threading.Thread(target=self.detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
    
    def detection_loop(self):
        """Ana tespit döngüsü"""
        frame_count = 0
        start_time = time.time()
        
        while self.is_detecting and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            # Tespit yap
            results = self.model(frame, conf=self.confidence_var.get(), classes=[0])  # class 0 = person
            
            # Sonuçları işle
            processed_frame = self.process_detection_results(frame, results, frame_count)
            
            # FPS hesapla
            frame_count += 1
            if frame_count % 30 == 0:  # Her 30 frame'de bir FPS güncelle
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                self.fps_var.set(f"FPS: {fps:.1f}")
            
            # Görüntüyü arayüzde göster
            self.update_display(processed_frame)
            
            # Kısa bekleme
            cv2.waitKey(1)
        
        self.video_capture.release()
    
    def process_detection_results(self, frame, results, frame_count=None):
        """Tespit sonuçlarını işler ve görselleştirir"""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Koordinatları al
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Sadece insan tespitlerini işle (class 0)
                    if box.cls[0].cpu().numpy() == 0:
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence
                        })
        
        # Tracking güncelle ve insan sayısını al
        if frame_count is not None:  # Video işleme
            current_human_count = self.update_tracking(detections)
        else:  # Tek görüntü işleme
            current_human_count = len(detections)
        
        # Bounding box'ları çiz ve tracking ID'leri göster
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Bounding box çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Güven skorunu yaz
            label = f"Person: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # İnsan sayısını güncelle
        if frame_count is not None:
            self.human_count_var.set(f"Anlık İnsan: {current_human_count} | Toplam Benzersiz: {self.total_unique_humans}")
        else:
            self.human_count_var.set(f"Tespit Edilen İnsan: {current_human_count}")
        
        # Sonuçları kaydet (frame_count varsa)
        if frame_count is not None:
            self.detection_results.append({
                'frame': frame_count,
                'human_count': current_human_count,
                'total_unique_humans': self.total_unique_humans,
                'timestamp': datetime.now().isoformat()
            })
        
        return frame
    
    def calculate_iou(self, bbox1, bbox2):
        """İki bounding box arasındaki IoU (Intersection over Union) hesaplar"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection koordinatları
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Intersection alanı
        if x2_i > x1_i and y2_i > y1_i:
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
        else:
            intersection = 0
        
        # Union alanı
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update_tracking(self, detections):
        """İnsan tracking'i günceller ve yeni insanları sayar"""
        current_humans = []
        
        # Mevcut detection'ları işle
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # En iyi eşleşmeyi bul
            best_match_id = None
            best_iou = 0.3  # Minimum IoU eşiği
            
            for track_id, track_info in self.tracked_humans.items():
                if track_info['frames_missing'] < self.max_frames_missing:
                    iou = self.calculate_iou(bbox, track_info['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = track_id
            
            if best_match_id is not None:
                # Mevcut track'i güncelle
                self.tracked_humans[best_match_id]['bbox'] = bbox
                self.tracked_humans[best_match_id]['frames_missing'] = 0
                current_humans.append(best_match_id)
            else:
                # Yeni track oluştur
                self.tracked_humans[self.next_track_id] = {
                    'bbox': bbox,
                    'frames_missing': 0
                }
                current_humans.append(self.next_track_id)
                self.next_track_id += 1
                self.total_unique_humans += 1
        
        # Eksik track'leri güncelle
        for track_id in list(self.tracked_humans.keys()):
            if track_id not in current_humans:
                self.tracked_humans[track_id]['frames_missing'] += 1
                if self.tracked_humans[track_id]['frames_missing'] >= self.max_frames_missing:
                    del self.tracked_humans[track_id]
        
        return len([t for t in self.tracked_humans.values() if t['frames_missing'] == 0])
    
    def process_image(self, image_path):
        """Görüntü dosyasını işler"""
        try:
            # Görüntüyü oku
            image = cv2.imread(image_path)
            if image is None:
                messagebox.showerror("Hata", "Görüntü dosyası okunamadı!")
                return
            
            # Tespit yap
            results = self.model(image, conf=self.confidence_var.get(), classes=[0])
            
            # Sonuçları işle
            processed_image = self.process_detection_results(image, results, 0)  # Tek görüntü için frame_count=0
            
            # Görüntüyü arayüzde göster
            self.update_display(processed_image)
            
            # Sonucu kaydet
            output_path = f"results/detections/detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(output_path, processed_image)
            
            messagebox.showinfo("Başarılı", f"Tespit tamamlandı!\nSonuç kaydedildi: {output_path}")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü işlenirken hata oluştu: {str(e)}")
    
    def update_display(self, frame):
        """Arayüzde görüntüyü günceller"""
        try:
            # Frame'i yeniden boyutlandır
            height, width = frame.shape[:2]
            max_width = 800
            max_height = 600
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # BGR'den RGB'ye dönüştür
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # PIL Image'e dönüştür
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Arayüzde göster
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Referansı koru
            
        except Exception as e:
            print(f"Görüntü güncellenirken hata: {str(e)}")
    
    def stop_detection(self):
        """Tespit işlemini durdurur"""
        self.is_detecting = False
        
        if self.video_capture:
            self.video_capture.release()
        
        # Buton durumlarını güncelle
        self.video_btn.config(state=tk.NORMAL)
        self.image_btn.config(state=tk.NORMAL)
        self.camera_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Sonuçları kaydet
        self.save_results()
        
        # Arayüzü temizle
        self.video_label.configure(text="Video/Görüntü burada görünecek")
        self.human_count_var.set("Tespit Edilen İnsan: 0")
        self.fps_var.set("FPS: 0")
    
    def save_results(self):
        """Tespit sonuçlarını kaydeder"""
        if not self.detection_results:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # İnsan sayımı sonuçlarını kaydet
        count_file = f"results/counts/human_count_{timestamp}.txt"
        with open(count_file, 'w', encoding='utf-8') as f:
            f.write("YOLOv8 İnsan Tespiti Sonuçları\n")
            f.write("=" * 40 + "\n")
            f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Toplam Frame: {len(self.detection_results)}\n")
            
            total_humans = sum(result['human_count'] for result in self.detection_results)
            avg_humans = total_humans / len(self.detection_results) if self.detection_results else 0
            
            f.write(f"Toplam Tespit: {total_humans}\n")
            f.write(f"Ortalama İnsan Sayısı: {avg_humans:.2f}\n")
            f.write(f"Toplam Benzersiz İnsan: {self.total_unique_humans}\n")
            f.write("\nFrame Detayları:\n")
            
            for result in self.detection_results:
                unique_count = result.get('total_unique_humans', 0)
                f.write(f"Frame {result['frame']}: {result['human_count']} insan (Benzersiz: {unique_count})\n")
        
        # JSON formatında da kaydet
        json_file = f"results/analysis/detection_analysis_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.detection_results, f, indent=2, ensure_ascii=False)
        
        print(f"Sonuçlar kaydedildi:\n{count_file}\n{json_file}")
        
        # Grafikleri oluştur
        self.create_visualization_plots()
    
    def create_visualization_plots(self):
        """Tespit sonuçlarını görselleştiren grafikleri oluşturur"""
        if not self.detection_results:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Veri hazırlama
        frames = [result['frame'] for result in self.detection_results]
        human_counts = [result['human_count'] for result in self.detection_results]
        unique_counts = [result.get('total_unique_humans', 0) for result in self.detection_results]
        
        # 1. Zaman Serisi Grafiği - İnsan Sayısı
        plt.figure(figsize=(15, 10))
        
        # Ana grafik
        plt.subplot(2, 2, 1)
        plt.plot(frames, human_counts, 'b-', linewidth=2, label='Anlık İnsan Sayısı')
        plt.plot(frames, unique_counts, 'r-', linewidth=2, label='Toplam Benzersiz İnsan')
        plt.xlabel('Frame')
        plt.ylabel('İnsan Sayısı')
        plt.title('İnsan Sayısı Zaman Serisi')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Histogram - İnsan Sayısı Dağılımı
        plt.subplot(2, 2, 2)
        plt.hist(human_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('İnsan Sayısı')
        plt.ylabel('Frame Sayısı')
        plt.title('İnsan Sayısı Dağılımı')
        plt.grid(True, alpha=0.3)
        
        # 3. Box Plot - İstatistiksel Özet
        plt.subplot(2, 2, 3)
        plt.boxplot([human_counts, unique_counts], labels=['Anlık', 'Benzersiz'])
        plt.ylabel('İnsan Sayısı')
        plt.title('İstatistiksel Özet')
        plt.grid(True, alpha=0.3)
        
        # 4. Kümülatif Grafik
        plt.subplot(2, 2, 4)
        cumulative_unique = []
        current_max = 0
        for count in unique_counts:
            if count > current_max:
                current_max = count
            cumulative_unique.append(current_max)
        
        plt.plot(frames, cumulative_unique, 'g-', linewidth=2, label='Kümülatif Benzersiz')
        plt.xlabel('Frame')
        plt.ylabel('Toplam Benzersiz İnsan')
        plt.title('Kümülatif Benzersiz İnsan Sayısı')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/analysis/visualization_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Detaylı Analiz Grafiği
        self.create_detailed_analysis_plot(timestamp)
        
        print(f"Grafikler oluşturuldu: results/analysis/visualization_{timestamp}.png")
    
    def create_detailed_analysis_plot(self, timestamp):
        """Detaylı analiz grafiği oluşturur"""
        if not self.detection_results:
            return
        
        frames = [result['frame'] for result in self.detection_results]
        human_counts = [result['human_count'] for result in self.detection_results]
        
        # Yeni bir figure oluştur
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Yoğunluk Analizi
        ax1.plot(frames, human_counts, 'b-', alpha=0.7)
        ax1.fill_between(frames, human_counts, alpha=0.3, color='blue')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('İnsan Sayısı')
        ax1.set_title('İnsan Yoğunluğu Analizi')
        ax1.grid(True, alpha=0.3)
        
        # 2. Hareket Analizi (Değişim Hızı)
        if len(human_counts) > 1:
            changes = [abs(human_counts[i] - human_counts[i-1]) for i in range(1, len(human_counts))]
            ax2.plot(frames[1:], changes, 'r-', linewidth=1)
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Değişim Miktarı')
            ax2.set_title('İnsan Sayısı Değişim Hızı')
            ax2.grid(True, alpha=0.3)
        
        # 3. İstatistiksel Özet
        stats_data = [
            max(human_counts),
            min(human_counts),
            sum(human_counts) / len(human_counts),
            len([x for x in human_counts if x > 0])
        ]
        stats_labels = ['Maksimum', 'Minimum', 'Ortalama', 'İnsanlı Frame']
        
        bars = ax3.bar(stats_labels, stats_data, color=['red', 'blue', 'green', 'orange'])
        ax3.set_ylabel('İnsan Sayısı')
        ax3.set_title('İstatistiksel Özet')
        
        # Değerleri çubukların üzerine yaz
        for bar, value in zip(bars, stats_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 4. Zaman Dilimi Analizi
        if len(frames) > 10:
            # Frame'leri 10 gruba böl
            chunk_size = len(frames) // 10
            chunk_averages = []
            chunk_labels = []
            
            for i in range(10):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < 9 else len(human_counts)
                avg = sum(human_counts[start_idx:end_idx]) / (end_idx - start_idx)
                chunk_averages.append(avg)
                chunk_labels.append(f'Bölüm {i+1}')
            
            ax4.bar(chunk_labels, chunk_averages, color='purple', alpha=0.7)
            ax4.set_xlabel('Zaman Dilimi')
            ax4.set_ylabel('Ortalama İnsan Sayısı')
            ax4.set_title('Zaman Dilimi Analizi')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'results/analysis/detailed_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detaylı analiz grafiği oluşturuldu: results/analysis/detailed_analysis_{timestamp}.png")
    
    def run(self):
        """Uygulamayı çalıştırır"""
        self.root.mainloop()

def main():
    """Ana fonksiyon"""
    app = HumanDetectionApp()
    app.run()

if __name__ == "__main__":
    main() 