import pandas as pd
file_path = "C:\\Users\\kdecs\\Desktop\\VSCode\\dekmidterms\\국민건강보험공단_건강검진정보_2023.csv"
df = pd.read_csv(file_path, encoding='cp949')

# 데이터프레임 분석 및 확인
df.info()
print(df.describe())
print(df.head())

#데이터 중복치 제거
initial_df = df.shape[0]
df = df.drop_duplicates()
print(df.shape, initial_df-df.shape[0])

# 데이터 결측치 확인
for col in df:
    print(f"{col}: {df[col].isnull().sum()} 개")

# 데이터 결측치 처리
df2 = pd.DataFrame()

for col in df:
    emp_row = df[col].isna().sum()
    total_row = df.shape[0]
    if ((emp_row / total_row) * 100) <= 10:
        df2[col] = df[col]

print(df2.head())
'''
print(df.head())
print(df.isna().sum())

print(df.isnull().sum())
print(df.shape[0])
print(df['치아우식증유무'])
'''