import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

# Membaca data Iris dari CSV (contoh dataset)
datas3 = pd.read_csv("iris.csv")
x = datas3.iloc[:, :4]
y = datas3.iloc[:, 4]

# Membagi data menjadi data latihan dan data uji
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.5)

print(x_train)
print()
print(y_train)
print()

# Membuat model Gaussian Naive Bayes
classifier = ComplementNB()
classifier.fit(x_train, y_train)

# Melakukan prediksi terhadap data uji
y_pred = classifier.predict(x_test)

# Menghitung dan mencetak akurasi
accuracy = accuracy_score(y_pred, y_test)
print('Akurasi awal:', accuracy * 100)

# Mengurangkan jumlah data latihan (misalnya, hanya menggunakan 10% data latihan)
x_train_small, _, y_train_small, _ = train_test_split(x_train, y_train, random_state=0, test_size=0.9)
classifier_small = ComplementNB()
classifier_small.fit(x_train_small, y_train_small)

# Melakukan prediksi terhadap data uji
y_pred_small = classifier_small.predict(x_test)

# Menghitung dan mencetak akurasi setelah mengurangkan data latihan
accuracy_small = accuracy_score(y_pred_small, y_test)
print('Akurasi setelah mengurangkan data latihan:', accuracy_small * 100)
