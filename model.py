import pandas as pd
import random

from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main(random_state = random.randint(1, 1000)):

    # membuka file data iris
    datas3 = pd.read_csv("iris.csv");

    x = datas3.iloc[:,:4]
    y = datas3.iloc[:,4]

    # tambahkan nama kolom di datas3 dan edit ke csv
    datas3.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    datas3.to_csv('iris2.csv', index_label='no')

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=random_state,test_size=0.4)

    # Membuat model dengan algoritma Complement Naive Bayes
    classifier = ComplementNB()
    classifier.fit(x_train, y_train) # melatih model 

    y_pred = classifier.predict(x_test) # melakukan prediksi terhadap data uji

    # mengubah y_pred menjadi dataframe dan menambahkan kolom class serta correct_class
    y_pred_final = pd.DataFrame(y_pred)
    y_pred_final.columns = ['class']
    y_pred_final['correct_class'] = pd.DataFrame(y_test).iloc[:, 0].reset_index(drop=True)

    # print akurasi model
    print('\n--- Program Menghitung Performa Klasifikasi Data Iris ---')
    print(f'\nAkurasi prediksi model = {accuracy_score(y_pred,y_test) * 100}%') # menghitung akurasi

    # membuat dataframe hasil final prediksi
    x_test_final = pd.DataFrame(x_test)
    index_awal = x_test_final.index
    penggabungan = pd.concat([x_test_final.reset_index(drop=True), y_pred_final], axis=1)
    penggabungan.index += 1

    penggabungan['index_awal'] = index_awal # menambahkan kolom index awal

    # pembagian data prediksi benar dan salah
    prediksi_benar = prediksi_salah = pd.DataFrame(columns=['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class', 'correct_class', 'index_awal'])
    counter_benar = counter_salah = 0

    # looping untuk membagi ke dalam dataframe prediksi benar dan salah
    for index, row in penggabungan.iterrows():

        if row['class'] == row['correct_class']:
            prediksi_benar = pd.concat([prediksi_benar, row.to_frame().T], ignore_index=True)
            counter_benar += 1

        else:
            prediksi_salah = pd.concat([prediksi_salah, row.to_frame().T], ignore_index=True)
            counter_salah += 1

    # memulai index dari satu
    prediksi_benar.index += 1
    prediksi_salah.index += 1

    # export ke excel
    with pd.ExcelWriter("final.xlsx") as writer :
        penggabungan.to_excel(writer, sheet_name="Hasil")
        prediksi_benar.to_excel(writer, sheet_name="Prediksi Benar")
        prediksi_salah.to_excel(writer, sheet_name="Prediksi Salah")
        datas3.to_excel(writer, sheet_name="Data Asli", index_label='index')

    # printing jumlah prediksi benar dan salah
    print(f'Jumlah hasil prediksi benar = {counter_benar}')
    print(f'Jumlah hasil prediksi salah = {counter_salah}')

    print('\n--- Data lengkap hasil klasifikasi bisa dilihat di file final.xlsx ---\n')


# pemanggilan function main
main()