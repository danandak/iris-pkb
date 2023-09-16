# %%
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


# %%
# membuka file data latihan
datas = pd.read_excel("datas.xlsx", sheet_name="All Labelling")
x_train = datas['tweet']
y_train = datas['sentimen']

datas2 = pd.read_excel("datas.xlsx", sheet_name="20 Random")
x_test = datas2['tweet']
y_test = datas2['sentimen']


# %%
# Pelatihan model
pipe = Pipeline(steps=[('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB())])

tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}

classifier = GridSearchCV(pipe, tuned_parameters, cv=10)

classifier.fit(x_train, y_train) # melatih model dengan 60 data latih
print(classification_report(y_test, classifier.predict(x_test), digits=4)) # menampilkan detail performa model

# %%
# membuka data mentah (144 data)
datas_test = pd.read_excel("datas.xlsx", sheet_name='Test After 20')
data2 = datas_test['tweet']

# mengetes model menggunakan data uji
prediksi = classifier.predict(data2)
counter = 1; pos = 0; neg = 0; net = 0
for x in prediksi:
    # print(f'sentimen data tweet ke-{counter}\n{x}'); counter+=1
    if x == 'Positif' :
        pos+=1
    elif x == 'Negatif' :
        neg+=1
    else:
        net+=1

print("\nJumlah masing-masing sentimen setelah diklasifikasikan : ")
print(f"Sentimen Positif : {pos}\nSentimen Netral : {net}\nSentimen Negatif : {neg}\n")

# ekspor hasil analisis ke excel
datas_test['sentimen'] = prediksi
datas_test.to_excel("hasil.xlsx", index=False)