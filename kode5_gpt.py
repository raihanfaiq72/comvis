import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. Eksplorasi Data
# Lakukan analisis eksploratori terhadap dataset
# Contoh: Baca dataset, visualisasikan contoh citra, dan hitung distribusi kelas

# Tentukan path ke direktori dataset
dataset_dir = "video/HD/2x/2xhd30fps.MOV"

# Inisialisasi list untuk menyimpan nama kelas dan jumlah citra per kelas
class_names = []
num_images_per_class = []

# Iterasi melalui setiap subdirektori dalam direktori dataset
for class_dir in os.listdir(dataset_dir):
    if os.path.isdir(os.path.join(dataset_dir, class_dir)):
        class_names.append(class_dir)
        num_images = len(os.listdir(os.path.join(dataset_dir, class_dir)))
        num_images_per_class.append(num_images)

# Visualisasikan distribusi kelas
plt.figure(figsize=(10, 6))
plt.bar(class_names, num_images_per_class, color='skyblue')
plt.xlabel('Kelas')
plt.ylabel('Jumlah Citra')
plt.title('Distribusi Kelas dalam Dataset')
plt.xticks(rotation=45)
plt.show()

# Tampilkan beberapa contoh citra untuk setiap kelas
num_samples_per_class = 3
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(class_names):
    class_dir = os.path.join(dataset_dir, class_name)
    image_files = os.listdir(class_dir)[:num_samples_per_class]
    for j, image_file in enumerate(image_files):
        image_path = os.path.join(class_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(len(class_names), num_samples_per_class, i * num_samples_per_class + j + 1)
        plt.imshow(image)
        plt.title(class_name)
        plt.axis('off')
plt.tight_layout()
plt.show()


# 2. Pra-pemrosesan
# Resizing, normalisasi, dan augmentasi data
# Contoh: Resize citra, normalisasi piksel, dan lakukan augmentasi

# 3. Pengembangan Model
# Pilih arsitektur model CNN dan latih model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 4. Evaluasi Performa
# Evaluasi model menggunakan metrik-metrik yang diminta
# Contoh: Hitung akurasi, presisi, recall, dan F1-score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# 5. Uji Coba
# Terapkan model pada citra-citra baru dan evaluasi performanya

new_images = load_new_images()
new_images_preprocessed = preprocess(new_images)
new_predictions = model.predict(new_images_preprocessed)

# Jawab Pertanyaan
# Jawablah pertanyaan-pertanyaan yang diberikan berdasarkan temuan dari langkah-langkah sebelumnya

# Simpan laporan tugas
# Simpan hasil eksperimen, analisis, dan jawaban pertanyaan dalam laporan tugas
