import cv2
import numpy as np
from time import sleep

# Tentukan ukuran minimum persegi panjang untuk objek kendaraan
width_min = 80
height_min = 80 

# Batasan kesalahan antar piksel
offset = 10

# Posisi garis hitungan
pos_line = 550 

# Delay antara setiap frame
delay = 600

# Inisialisasi deteksi objek
detec = []

# Inisialisasi jumlah kendaraan
car_count = 0

# Fungsi untuk menghitung pusat persegi panjang
def hitung_pusat(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Inisialisasi video capture dari file video
cap = cv2.VideoCapture('video/video.mp4')

# Inisialisasi algoritma pengurangan latar belakang
subtraction = cv2.createBackgroundSubtractorMOG2()

# Font untuk teks jumlah kendaraan
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
font_thickness = 2

while True:
    # Baca frame dari video
    ret, frame1 = cap.read()
    
    # Tunda untuk memberikan efek visual
    time = float(1 / delay)
    sleep(time) 
    
    # Ubah frame ke citra keabuan
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Haluskan citra
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    
    # Terapkan algoritma pengurangan latar belakang
    img_sub = subtraction.apply(blur)
    
    # Dilasi citra
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    
    # Temukan kontur pada citra hasil dilasi
    contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Gambar garis hitungan pada frame
    cv2.line(frame1, (25, pos_line), (1200, pos_line), (255, 127, 0), 3) 
    
    # Inisialisasi objek yang bergerak
    car_count = 0
    
    # Loop melalui setiap kontur
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Validasi kontur berdasarkan ukuran minimum
        if w >= width_min and h >= height_min:
            # Gambar persegi panjang pada frame
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)   
            
            # Hitung pusat persegi panjang dan tambahkan ke dalam list detek
            center = hitung_pusat(x, y, w, h)
            detec.append(center)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)
            
            # Hitung jumlah objek bergerak
            car_count += 1
    
    # Tampilkan jumlah kendaraan di atas frame
    cv2.putText(frame1, f'objek bergerak yang terdeteksi: {car_count}', (10, 50), font, font_scale, font_color, font_thickness)
    
    # Tampilkan frame dengan objek kendaraan yang terdeteksi
    cv2.imshow("Deteksi Objek Yang Bergerak", frame1)
    
    # Periksa input keyboard untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Tutup jendela dan lepaskan video capture
cv2.destroyAllWindows()
cap.release()