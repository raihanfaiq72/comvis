import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Definisikan model CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.fc1 = nn.Linear(64*28*28, 512)
        self.fc2 = nn.Linear(512, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64*28*28)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Fungsi untuk membaca video
def read_video(path):
    frames = []
    video_capture = cv2.VideoCapture(path)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        # Resize frame ke ukuran yang diperlukan oleh model
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    video_capture.release()
    return frames

# Dataset untuk frame video
class VideoDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return frame

# Path video
video_path = 'video/HD/1x/1xhd60fps.MOV'

# Membaca video
frames = read_video(video_path)

# Inisialisasi model CNN
model = CNN()

# Mengubah model ke mode evaluasi
model.eval()

# Memproses setiap frame menggunakan model CNN
predictions = []
with torch.no_grad():
    for frame in frames:
        frame = torch.unsqueeze(frame, 0)  # Tambahkan dimensi batch
        prediction = model(frame)
        predictions.append(prediction.item())

# Contoh evaluasi kinerja model
# (Anda mungkin perlu mengubah ini sesuai dengan kebutuhan Anda)
true_labels = np.ones(len(predictions))  # Misalnya, kita anggap semua frame adalah objek yang sama
predicted_labels = np.array(predictions) > 0.5
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Menampilkan hasil evaluasi
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
