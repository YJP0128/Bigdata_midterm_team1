import pandas as pd
import numpy as np
columns = ['사용일자', '노선명', '역명', '승차총승객수', '하차총승객수', '등록일자']
data = pd.read_csv("CARD_SUBWAY_MONTH_202503.csv",encoding='utf-8',skiprows=1,names=columns, usecols=range(6),thousands=",")
columns = ['사용일자', '노선명', '역명', '승차총승객수', '하차총승객수', '등록일자']



print(data.head())
print(data.describe())
print(data.info())


print(data.isna().sum())
print(data.isna().mean()*100)

data= data.drop(columns=['등록일자'])

numerical_cols=['승차총승객수','하차총승객수'] #나누기 
categorical_cols=['역명','사용일자','노선명']

for col in numerical_cols: 
    mean=data[col].mean()
    std=data[col].std()
    data=data[(data[col] > mean-3*std) & (data[col]< mean+3*std)]

from sklearn.preprocessing import MinMaxScaler #연속형 데이터 정규화
scalar=MinMaxScaler()
data[numerical_cols]= scalar.fit_transform(data[numerical_cols])

from sklearn.preprocessing import LabelEncoder #범주형데이터 정규화
le = LabelEncoder()

for col in categorical_cols: #합치기
    data[col]= le.fit_transform(data[col])

data.to_csv('지하철 정규화.csv', index=False)
