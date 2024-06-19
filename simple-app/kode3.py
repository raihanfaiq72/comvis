import cv2
import numpy as np 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from time import sleep

# Siapkan kamera
# kamera = cv2.VideoCapture('video/video.mp4')
kamera = cv2.VideoCapture('../video/1xhd30fps.MOV')
# kamera = cv2.VideoCapture('video/4k/1x/1x4k30fps.MOV')


# Variabel untuk menyimpan label sebenarnya dan label prediksi
true_labels = []
predicted_labels = []

# Tentukan ukuran minimum persegi panjang
ukuran_minimal_lebar = 80
ukuran_minimal_tinggi = 80

# Inisialisasi delay untuk setiap frame
delay = 600

# Inisialisasi jumlah kendaraan yang terdeteksi
counter = 0

# Font untuk menampilkan teks pada frame
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
font_thickness = 2

# Tentukan posisi garis hitungan
posisi_garis_hitungan = 650

# Inisialisasi algoritma pengurangan latar belakang
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Fungsi untuk menghitung pusat persegi panjang
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

while True:
    ret, frame1 = kamera.read()  # Ambil frame dari kamera
    
    # Pengecekan apakah frame berhasil dibaca atau tidak
    if not ret:
        print("Gagal membaca frame")
        break

    # Ubah frame ke citra keabuan
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  
    # Haluskan citra
    blur = cv2.GaussianBlur(grey, (5, 5), 5)  
    
    # Tunda untuk efek visual
    time = float(1/delay)
    sleep(time)
    
    # Terapkan algoritma pengurangan latar belakang pada setiap frame
    img_sub = algo.apply(blur)
    # Dilasi citra
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # Temukan kontur pada citra hasil dilasi
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Gambar garis hitungan pada frame
    cv2.line(frame1, (25, posisi_garis_hitungan), (1900, posisi_garis_hitungan), (255, 127, 0), 3)

    # Inisialisasi jumlah objek kendaraan yang terdeteksi
    car_count = 0
    
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
    
        # Validasi ukuran objek kendaraan
        validasi_counter = (w >= ukuran_minimal_lebar) and (h >= ukuran_minimal_tinggi)
    
        if not validasi_counter :
            continue
        
        # Gambar persegi panjang dan tulis jumlah kendaraan pada frame
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Kendaraan " + str(counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 240, 0), 2)

        # Hitung pusat persegi panjang dan tambahkan ke dalam list deteksi
        center = hitung_pusat(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 5, (0, 0, 255), -1)
        car_count += 1

        for (x, y) in detect:
            if y < (posisi_garis_hitungan + offset) and y > (posisi_garis_hitungan - offset):
                counter += 1
                detect.remove((x, y))
                print("Jumlah Kendaraan: " + str(counter))

        # Tentukan label berdasarkan validasi_counter
        actual_label = "kendaraan" if validasi_counter else "bukan kendaraan"
        predicted_label = "kendaraan" if validasi_counter else "bukan kendaraan"

        # Tambahkan label aktual ke dalam list true_labels
        true_labels.append(actual_label)

        # Tambahkan label prediksi ke dalam list predicted_labels
        predicted_labels.append(predicted_label)
        
    # Tampilkan jumlah kendaraan pada frame
    cv2.putText(frame1, "Jumlah Kendaraan : " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, 'objek bergerak yang terdeteksi: {}'.format(car_count), (10, 50), font, font_scale, font_color, font_thickness)
    
    # Tampilkan frame
    cv2.imshow('Video Asli', frame1)
    # cv2.imshow('detecter',dilatada)

    # Evaluasi performa model setelah beberapa iterasi
    if cv2.waitKey(1) == 13:
        # Hitung matriks kebingungan (confusion matrix)
        cm = confusion_matrix(true_labels, predicted_labels)

        # Hitung metrik evaluasi
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        # Cetak hasil evaluasi
        print("Confusion Matrix:")
        print(cm)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # Keluar dari loop
        break

# Tutup semua jendela dan lepaskan kamera
cv2.destroyAllWindows()
kamera.release()
