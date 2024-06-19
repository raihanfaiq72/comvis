import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

driveawal ='/home/user/Experiment/seminar-data-mining/diabetes_modified.csv' #kalau pakai local

df = pd.read_csv(driveawal+'diabetes_modified.csv')
df

df.info()

# tampilkan jumlah null
null_counts = df.isnull().sum()
print("\nJumlah nilai null di setiap kolom:")
print(null_counts)

# Memeriksa duplikat
duplikat = df.duplicated()
# Menampilkan baris yang duplikat
baris_duplikat = df[df.duplicated()]
print("\n Jumlah baris duplikat:\n", len(baris_duplikat))

X = df.drop(columns=["Outcome"])  # Mengambil semua kolom kecuali kolom "label"
y = df["Outcome"]  # Mengambil kolom "label

X.head() #tampilkan contoh data

def plots(feature):
    fig = plt.figure(constrained_layout = True, figsize=(10,3))
    gs = gridspec.GridSpec(nrows=1, ncols=4, figure=fig)

    ax1 = fig.add_subplot(gs[0,:3])
    sns.histplot(df.loc[df["Outcome"]==0,feature],
                 kde = False, color = "#004a4d",
                  bins=40,
                 label="Not Diabetes", ax=ax1);
    sns.histplot(df.loc[df["Outcome"]==1,feature],
                 kde = False, color = "#7d0101",
                 bins=40,
                 label="Diabetes", ax=ax1);
    ax2 = fig.add_subplot(gs[0,3])
    sns.boxplot(X[feature], orient="v", color = "#989100",
                width = 0.2, ax=ax2);

    ax1.legend(loc="upper right");


plots("Glucose")


# Menghitung matriks korelasi Pearson
correlation_matrix = df.corr(method='pearson')
plt.figure(figsize=(12,10))
# Membuat heatmap dari matriks korelasi
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriks Korelasi Pearson')
plt.show()


# Menentukan threshold
threshold = 0.3

# Mencari pasangan atribut yang berpengaruh berdasarkan threshold
influential_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) >= threshold:
            influential_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

# Menampilkan hasil
print("Atribut yang saling berpengaruh berdasarkan threshold:")
for pair in influential_pairs:
    print(f"Atribut {pair[0]} dan {pair[1]} memiliki korelasi {pair[2]:.2f}")

print('Pada tahapan ini anda juga bisa membuang atribut yang tidak relevan')


# Menghitung jumlah 0 dan 1
unique, counts = np.unique(y, return_counts=True)
count_dict = dict(zip(unique, counts))
keys_list = list(count_dict.keys())

# Plot jumlah 0 dan 1 ke dalam diagram batang
plt.bar(count_dict.keys(), count_dict.values(), color=['blue', 'green'])
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Count of Label')
plt.xticks(keys_list, keys_list)
plt.show()


df.describe() #jika data categorical "string" bisa gunakan bar plot


# Membuat boxplot untuk setiap kolom
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.values)
plt.title('Boxplot Semua Kolom')
plt.show()