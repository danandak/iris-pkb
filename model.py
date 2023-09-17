# %%
import pandas as pd

from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


# %%
# membuka file data latihan
datas3 = pd.read_csv("iris.csv");
x = datas3.iloc[:,:4]
y = datas3.iloc[:,4]

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
test = pd.DataFrame(y_pred)
print(test)

print()

print(pd.DataFrame(x_test))

excel_data = pd.concat([pd.DataFrame(x_test), test], axis=1)
# print(excel_data)

# excel_data = {
#     'test' : x_test, 'class': y_pred
# }

# df = pd.DataFrame(excel_data)

# excel_data.to_excel("result.xlsx", index=False)

print('accuracy is',accuracy_score(y_pred,y_test) * 100) # menghitung akurasi