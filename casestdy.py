from enum import unique
from sklearn.linear_model import LinearRegression
import py
from matplotlib import pyplot as pic
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def branch_name_process(df, column):
    unique_branch = pd.unique(df[column])

#Quy trình xây dựng mô hình hồi quy tuyến tính
#b1: Chọn feature đặc trưng nào để đưa mô hình dự đoán
df = pd.read_csv("data\Case_study_CarPrice_Assignment.csv")
corr = df.corr(method='kendall')
corr.to_csv("corr.csv")

#b2: Lọc nhiễu

#b3: normalizer data hay không

#b4: Chọn mô hình

#b5: 