import twint
import pandas as pd

# configuration 1 (data rentang waktu pertama)
c = twint.Config()
c.Search = 'penundaan pemilu'
c.Lang = 'in'
c.Since = '2022-01-01'
c.Until = '2022-02-23'
c.Limit = 300
c.Store_csv = True
c.Output = "before.csv"

twint.run.Search(c)

df = pd.read_csv("before.csv")
tweets_before = df['tweet'].tolist()
df1 = pd.DataFrame(tweets_before, columns=['tweet'])

# configuration 2 (data rentang waktu kedua)
c = twint.Config()
c.Search = 'penundaan pemilu'
c.Lang = 'in'
c.Since = '2022-02-24'
c.Until = '2022-05-27'
c.Limit = 300
c.Store_csv = True
c.Output = "after.csv"

twint.run.Search(c)

df = pd.read_csv("after.csv")
tweets_after = df['tweet'].tolist()
df2 = pd.DataFrame(tweets_after, columns=['tweet'])

# write to excel
with pd.ExcelWriter("crawling.xlsx") as writer :
    df1.to_excel(writer, sheet_name="before")
    df2.to_excel(writer, sheet_name="after")