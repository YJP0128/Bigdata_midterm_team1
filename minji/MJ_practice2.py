import pandas as pd
import numpy as np
df=pd.read_csv("국민건강보험공단_건강검진정보_2023.csv", encoding= 'cp949')

df = df.iloc[:, 0:7]
df.columns= ['기준년도','가입자일련번호','성별코드','연령대코드','시도코드','신장','체중',]
print(df.head)
print(df.describe)

df= df.drop(columns=['가입자일련번호'])


print(df.isna().sum())
print(df.isna().mean()*100)

num_cols=['기준년도',]
cat_cols=[]

for col in num_cols:
std=df[col].std()
mean=df[col].mean()
data=data[(data[col] > mean-3*std) & (data[col]< mean+3*std)]

from sklearn.preprocessing import MinMaxScaler
