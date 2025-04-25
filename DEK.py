import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 파일 읽기
file_path = "C:\\Users\\kdecs\\Desktop\\VSCode\\dekmidterms\\국민건강보험공단_건강검진정보_2023.csv"
df = pd.read_csv(file_path, encoding='cp949')

# 데이터프레임 분석 및 확인
df.info()
print(df.describe())
print(df.head())

#데이터 중복치 제거
def remove_dup(df):
    df_original = df.copy()
    df = df.drop_duplicates()
    return df_original, df

# 데이터 결측치 개수 확인
def count_nan(df):
    count_nan_dict = {}
    for col in df:
        count_nan_dict[col] = df[col].isnull().sum()
    return count_nan_dict

# 결측치가 너무 많은 행 제거
def remove_nan_col(df):
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()

    for col in df:
        emp_row = df[col].isna().sum()
        total_row = df.shape[0]
        if ((emp_row / total_row) * 100) <= 10:
            df2[col] = df[col]
        else:
            df3[col] = df[col]
    return df2, df3


# 행 분류 - 연속형, 범주형
def num_col_classifier(df):
    num_col = pd.DataFrame() # 연속형 행 분류

    for col in df:
        no_unique = df[col].nunique()
        total_no = len(df[col])
        ratio = no_unique / total_no
        if df[col].dtype == int:
            if ratio >= 0.05:
                num_col[col] = df[col]
        elif df[col].dtype == float:
            if ratio >= 0.05:
                num_col[col] = df[col]
    return num_col

def cat_col_classifier(df):
    cat_col = pd.DataFrame() # 범주형 행 분류

    for col in df:
        no_unique = df[col].nunique()
        total_no = len(df[col])
        ratio = no_unique / total_no
        if df[col].dtype == int:
            if ratio <= 0.05:
                cat_col[col] = df[col] # type: ignore
        elif df[col].dtype == float:
            if ratio <= 0.05:
                cat_col[col] = df[col]
        else:
            cat_col[col] = df[col]
    return cat_col

num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('label_encoder', LabelEncoder())
])

col_transfomer = ColumnTransformer(transformers = [
    ('num_pipeline', num_pipeline, num_col_classifier(df)),
    ('cat_pipeline', cat_pipeline, cat_col_classifier(df))
])

steps = [
    ('remove_dup', FunctionTransformer(remove_dup, validate=True)),
    ('count_nan', FunctionTransformer(count_nan, validate=True)),
    ('remove_nan_col', FunctionTransformer(remove_nan_col, validate=True)),
    ('col_classifier', FunctionTransformer(num_col_classifier, validate=True)),
    ('col_classifier', FunctionTransformer(cat_col_classifier, validate=True)),
    ('column_transform', col_transfomer)
]

pipe_final = Pipeline(steps)
preprocessed_df = pipe_final.fit_transform(df)

output_file = 'preprocessed_data.csv'
preprocessed_df.to_csv(output_file, index=False)

print("Preprocessed DataFrame:")
print(preprocessed_df)
print(f"\nSaved to {output_file}")

