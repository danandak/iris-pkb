# %%
import pandas as pd
import sys

from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix


# %%
# membuka file data latihan
datas3 = pd.read_csv("iris.csv");
x = datas3.iloc[:,:4]
y = datas3.iloc[:,4]

# tambahkan nama kolom di datas3 dan edit ke csv
datas3.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# datas3.to_csv('iris2.csv', index=True)
# datas3.index = datas3.index + 1
datas3.to_csv('iris2.csv', index_label='no')
# sys.exit()

# printing data untuk x dan y
print(x)
print()
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.4)

# printing jumlah x_train dan y_train
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print()


# %%
# Membuat model dengan algoritma Complement Naive Bayes
classifier = ComplementNB()
classifier.fit(x_train, y_train) # melatih model dengan 60 data latih

y_pred = classifier.predict(x_test) # melakukan prediksi terhadap data uji
# print(y_pred)


test_y_pred = pd.DataFrame(y_pred)
test_y_pred.columns = ['class']
print(test_y_pred)

print()

test_x_test = pd.DataFrame(x_test)
# buat varibel test di print tanpa index
test_x_test = test_x_test.reset_index(drop=True)
print(test_x_test)
# sys.exit()

# concat x_test dan y_pred
excel_data = pd.concat([test_x_test, test_y_pred], axis=1)
print(excel_data)

# excel_data = {
#     'test' : x_test, 'class': y_pred
# }

# df = pd.DataFrame(excel_data)

excel_data.to_excel("result.xlsx", index=False)

print('accuracy is',accuracy_score(y_pred,y_test) * 100) # menghitung 

correct_predictions = (y_pred == y_test).sum()
incorrect_predictions = len(y_test) - correct_predictions

print('Prediksi yang benar:', correct_predictions)
print('Prediksi yang salah:', incorrect_predictions)

# detailkan yang benar dan yang salah dari data uji
print()
print('Detail prediksi yang benar:')
true_pred = x_test[y_pred == y_test]
print(x_test[y_pred == y_test])
print()
print('Detail prediksi yang salah:')
print(x_test[y_pred != y_test])
print()