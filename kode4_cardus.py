import cv2
import numpy as np
import matplotlib.pyplot as plt

class VideoProcessor:
    def __init__(self, video_file):
        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)

        if not self.cap.isOpened():
            print("Gagal membuka video")
            exit()

    def process_frame(self, frame):
        grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.GaussianBlur(grey_frame, (5, 5), 5)
        return blur_frame

    def start_processing(self):
        plt.figure()
        frame_number = 1
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Selesai membaca video")
                break

            if frame_number == 1:
                processed_frame1 = self.process_frame(frame)
            else:
                processed_frame2 = self.process_frame(frame)
                diff_frame = np.square(np.subtract(processed_frame1.astype(np.int32), processed_frame2.astype(np.int32)))
                _, thresholded_diff = cv2.threshold(diff_frame.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

                kernel = np.ones((3,3), np.uint8)
                thresholded_diff = cv2.erode(thresholded_diff, kernel, iterations=1) 
                kernel = np.ones((5,5), np.uint8)
                filled_diff = cv2.dilate(thresholded_diff, kernel, iterations=4)

                contours, _ = cv2.findContours(filled_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    validasi_counter = (w >= ukuran_minimal_lebar) and (h >= ukuran_minimal_tinggi)

                    if not validasi_counter:
                        continue

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame_number += 1

            cv2.imshow('Deteksi Mobil', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    def release(self):
        self.cap.release()

# Parameter konfigurasi
video_file = 'video/HD/2x/2xhd30fps.MOV'
ukuran_minimal_lebar = 80
ukuran_minimal_tinggi = 80

# Memulai proses
processor = VideoProcessor(video_file)
processor.start_processing()
processor.release()

cv2.destroyAllWindows()
