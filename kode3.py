import cv2
import numpy as np 

# Siapkan kamera

# jika di komputer linux
# kamera = cv2.VideoCapture('/home/user/Experiment/pakcardus/faiq/video/video.mp4')
# kamera = cv2.VideoCapture('/home/user/Experiment/pakcardus/faiq/video/IMG_3209.MOV')

# jika di komputer windows 
# kamera = cv2.VideoCapture('C:\laragon\www\comvis\video\HD\1x\1xhd30fps.MOV')
kamera = cv2.VideoCapture('video/HD/2x/2xhd30fps.MOV')

# Tentukan ukuran minimum persegi panjang
ukuran_minimal_lebar = 80
ukuran_minimal_tinggi = 80

# Tentukan posisi garis hitungan
posisi_garis_hitungan = 650

# Inisialisasi algoritma pengurangan latar belakang
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Fungsi untuk menangani pusat persegi panjang
def hitung_pusat(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1

    return cx, cy 

# List untuk menyimpan pusat-pusat persegi panjang yang terdeteksi
detect = []

# Batasan kesalahan antar piksel
offset = 6

# Penghitung jumlah kendaraan yang terdeteksi
counter = 0

while True:
    ret, frame1 = kamera.read()  # Ambil frame dari kamera
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Ubah frame ke citra keabuan
    blur = cv2.GaussianBlur(grey, (5, 5), 5)  # Haluskan citra

    # Terapkan setiap frame dengan algoritma pengurangan latar belakang
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))  # Dilasi citra
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # Temukan kontur pada citra hasil dilasi
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Gambar garis hitungan pada frame
    cv2.line(frame1, (25, posisi_garis_hitungan), (1900, posisi_garis_hitungan), (255, 127, 0), 3)

    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validasi_counter = (w >= ukuran_minimal_lebar) and (h >= ukuran_minimal_tinggi)
    
        # Tambahkan pengecekan properti objek atau warna untuk memastikan bahwa yang terdeteksi adalah mobil
        # Misalnya, Anda bisa menambahkan pengecekan berdasarkan rasio aspek
        # aspect_ratio = float(w)/h
        # validasi_mobil = (aspect_ratio >= 1.0) and (aspect_ratio <= 4.0)  # Contoh pengecekan rasio aspek
    
        if not validasi_counter :
            continue
        
        # Gambar persegi panjang dan tulis jumlah kendaraan pada frame
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Kendaraan " + str(counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 240, 0), 2)

        # Hitung pusat persegi panjang dan tambahkan ke dalam list detect
        center = hitung_pusat(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 5, (0, 0, 255), -1)

        for (x, y) in detect:
            if y < (posisi_garis_hitungan + offset) and y > (posisi_garis_hitungan - offset):
                counter += 1
                detect.remove((x, y))
                print("Jumlah Kendaraan: " + str(counter))

    # Tampilkan jumlah kendaraan pada frame
    cv2.putText(frame1, "Jumlah Kendaraan : " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    

    
    # imshownya disini
    cv2.imshow('Video Asli', frame1)
    # cv2.imshow('detecter',dilatada)




    # Tunggu tombol 'Enter' ditekan untuk keluar dari loop
    if cv2.waitKey(1) == 13:
        break

# Tutup semua jendela dan lepaskan kamera
cv2.destroyAllWindows()
kamera.release()
