import cv2
import numpy as np
from ultralytics import YOLO

# 1. Inisialisasi Model YOLOv8-Pose (Versi Nano agar ringan di laptop)
print("Memuat model YOLOv8-Pose... Mohon tunggu.")
model = YOLO('yolov8n-pose.pt')

# 2. Konfigurasi Variabel Global
baseline_dist = None  # Menyimpan jarak ideal setelah kalibrasi
TOLERANSI = 0.85      # Batas toleransi (85% dari posisi tegak). Di bawah ini dianggap membungkuk.
status_postur = "BELUM DIKALIBRASI"
warna_status = (255, 191, 0) # Warna Biru Langit (Default)

# 3. Mengaktifkan Kamera/Webcam
cap = cv2.VideoCapture(0)

# Mengatur resolusi kamera (Opsional, sesuaikan dengan webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n=== ERGOSIT-AI BERHASIL DIJALANKAN ===")
print("PANDUAN EXPO:")
print("1. Duduklah dengan posisi TEGAK di depan kamera.")
print("2. Tekan tombol 'C' pada keyboard untuk KALIBRASI.")
print("3. Coba membungkuk untuk menguji sistem.")
print("4. Tekan tombol 'Q' untuk keluar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal mengakses webcam.")
        break

    # Flip kamera secara horizontal agar seperti cermin (lebih natural saat expo)
    frame = cv2.flip(frame, 1)
    tinggi_frame, lebar_frame, _ = frame.shape

    # 4. Jalankan Deteksi Pose YOLO
    results = model(frame, verbose=False, stream=True)
    
    current_dist = None
    
    for r in results:
        # Mengambil koordinat keypoints tubuh
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            keypoints = r.keypoints.xy[0].cpu().numpy()
            
            # Memastikan titik penting terdeteksi (0: Hidung, 5: Bahu Kiri, 6: Bahu Kanan)
            if len(keypoints) > 6:
                hidung = keypoints[0]
                bahu_kiri = keypoints[5]
                bahu_kanan = keypoints[6]
                
                # Memeriksa apakah titik hidung dan kedua bahu valid (tidak bernilai 0)
                if hidung[1] > 0 and bahu_kiri[1] > 0 and bahu_kanan[1] > 0:
                    # Hitung titik tengah bahu (Y-axis)
                    y_tengah_bahu = (bahu_kiri[1] + bahu_kanan[1]) / 2
                    y_hidung = hidung[1]
                    
                    # Jarak vertikal antara hidung dan garis bahu
                    current_dist = y_tengah_bahu - y_hidung
                    
                    # Gambarkan visualisasi titik di layar
                    cv2.circle(frame, (int(hidung[0]), int(hidung[1])), 6, (0, 255, 255), -1)
                    cv2.circle(frame, (int(bahu_kiri[0]), int(bahu_kiri[1])), 6, (255, 0, 0), -1)
                    cv2.circle(frame, (int(bahu_kanan[0]), int(bahu_kanan[1])), 6, (255, 0, 0), -1)
        
        # Gambar garis tulang (skeleton) bawaan YOLO untuk efek visual Expo
        frame = r.plot(boxes=False) 

    # 5. Logika Analisis Postur
    if current_dist is not None:
        if baseline_dist is None:
            status_postur = "TEKAN 'C' UNTUK KALIBRASI"
            warna_status = (0, 165, 255) # Oranye
        else:
            # Jika jarak saat ini lebih kecil dari batas toleransi kalibrasi
            if current_dist < (baseline_dist * TOLERANSI):
                status_postur = "PERINGATAN: ANDA MEMBUNGKUK!"
                warna_status = (0, 0, 255) # Merah
            else:
                status_postur = "POSTUR BAGUS (TEGAK)"
                warna_status = (0, 255, 0) # Hijau
    else:
        status_postur = "PENGGUNA TIDAK TERDETEKSI"
        warna_status = (128, 128, 128) # Abu-abu

    # 6. Desain Interface HUD (Heads-Up Display) untuk Expo
    # Membuat panel background transparan di bagian atas
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (lebar_frame, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Menampilkan Status Utama
    cv2.putText(frame, f"STATUS: {status_postur}", (30, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, warna_status, 2)
    
    # Menampilkan Info Metrik Jarak di Pojok Kanan Atas
    txt_jarak = f"Jarak Saat Ini: {int(current_dist) if current_dist else 0}px"
    txt_target = f"Target Kalibrasi: {int(baseline_dist) if baseline_dist else 0}px"
    cv2.putText(frame, txt_jarak, (lebar_frame - 300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, txt_target, (lebar_frame - 300, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Menampilkan Instruksi di Bagian Bawah Layar
    cv2.putText(frame, "Kontrol: [C] Kalibrasi Posisi Tegak  |  [Q] Keluar Aplikasi", 
                (30, tinggi_frame - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 7. Menampilkan Output Frame
    cv2.imshow("ErgoSit-AI - Monitor Postur Kerja", frame)

    # 8. Manajemen Keyboard Event
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') or key == ord('C'):
        if current_dist is not None:
            baseline_dist = current_dist
            print(f"Kalibrasi Berhasil! Posisi ideal dikunci pada jarak: {int(baseline_dist)}px")
        else:
            print("Gagal kalibrasi. Pastikan wajah dan bahu Anda terlihat jelas di kamera.")
            
    elif key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Aplikasi ditutup dengan aman.")