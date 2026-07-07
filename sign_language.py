import cv2
from ultralytics import YOLO

# 1. Load model hasil training bahasa isyarat kamu
# Jika belum ditraining, kamu bisa pakai model default 'yolov8n.pt' dulu untuk testing jalurnya
model = YOLO(r'D:\python\Multimedia_S6\best.pt') 

# webcam
cap = cv2.VideoCapture(0)

print("=== SIGNTALK-AI READY FOR MINI EXPO ===")
print("Arahkan tangan Anda membentuk bahasa isyarat ke kamera...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame agar seperti cermin
    frame = cv2.flip(frame, 1)

    # 3. Jalankan prediksi YOLO
    # Set threshold ke 0.5 (50%) agar deteksi lebih akurat dan tidak berkedip
    results = model(frame, conf=0.5, stream=True)

    for r in results:
        # Gunakan fungsi bawaan YOLO untuk menggambar bounding box + label huruf
        frame = r.plot() 
        
        # Logika tambahan untuk Expo (Opsional):
        # Kamu bisa mengambil teks huruf yang terdeteksi untuk digabungkan menjadi sebuah kata.

    # 4. Tampilkan di layar
    cv2.imshow("SignTalk-AI: Sign Language Translator", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()