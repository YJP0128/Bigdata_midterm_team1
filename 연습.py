#Step 
# 1. 데이터 읽기
#Step 2. 데이터 타입별 분리 (object vs int/float)
#Step 3. 결측치 확인 및 처리
#Step 4. 이상치 확인 및 처리
#Step 5. 범주형 인코딩
#Step 6. 수치형 스케일링 (필요할 경우)
#Step 7. 새로운 컬럼 병합/생성 (필요할 경우)
import pandas as pd
df= pd.read_csv('StudentsPerformance.csv.xls')

df.drop(['lunch'], axis=1, inplace=True)
df.head()
from sklearn.preprocessing import LabelEncoder
df_processed = df.copy()

categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'test preparation course'] #범주화 컬럼을 한번에 인코딩
le = LabelEncoder()
for col in categorical_cols:
    df_processed[col] = le.fit_transform(df_processed[col])

for score_col in ['math score', 'reading score', 'writing score']: #연속형 칼럼들 이상치 처리 
    Q1 = df_processed[score_col].quantile(0.25)
    Q3 = df_processed[score_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_processed[score_col] = df_processed[score_col].clip(lower_bound, upper_bound)

df_processed.head()
print(df.isnull().sum()/len(df)*100) #결측치 확인 
df.drop_duplicates(inplace=True)
