import matplotlib.pyplot as plt
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
path = "crawling.xlsx"

# membaca file hasil crawling
df = pd.read_excel(path, sheet_name='before')
df2 = pd.read_excel(path, sheet_name='after')
datas = df['tweet'].tolist()
datas.extend(df2['tweet'].tolist())

def preprocess(text):
    # mengecilkan huruf
    text = (text).lower()  #     
    text = re.sub(r"\d+", "", text) #menghapus angka

    # menghilangkan simbol dan tanda baca
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()) 

    # menghilangkan stopwords
    text = stopword.remove(text)

    return text


data = [preprocess(t) for t in datas]

# menghilangkan data duplikat
data = [*set(data)]

excel_data = {
    'tweet' : data, 'sentimen':''
}

df = pd.DataFrame(excel_data)

df.to_excel('cleaned.xlsx', index=False)