from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
# Đọc dữ liệu
df = pd.read_csv("Case_study_CarPrice_Assignment.csv")
print(df.head())
print(df.info())
print(df.shape)
# Xử lí dữ liệu
#preprocessing
# tên xe(str), số cửa, chiều dài, mã lực, nguyên lieu(str), độ an toàn
def branch_name_process(df, column):
    unique_branch = list(pd.unique(df[column]))
    for idx, branch_name in enumerate(unique_branch):
        # get index
        index = df.index[df[column] == branch_name].tolist()
        df.loc[index,column] = int(idx)
    return df
# Quy trinh xay dung mo hinh  hoi quy tuyen tinh
# Vẽ heatmap
# sns.heatmap(df.corr(), vmin=0, vmax=1, annot=True, cmap=plt.cm.Reds)
# plt.show()
# b1: chon feature dac trung nao de dua mo hinh du doan
df['BranchName'] = df.apply(lambda x:str(x['CarName']).split(" ")[0],axis=1).reset_index(drop=True)

# su dung cong cu cua pandas(requirments: du  lieu cot nay phai co dinh, ko thay doi)
df['BranchName'] = df['BranchName'].astype('category').cat.codes
# tmp = df.corr()

# print(tmp.head(1))

# b2: loc nhieu(cuc ky quan trong)
x = df[['carwidth','curbweight','enginesize','citympg','highwaympg','BranchName']]
y = df["price"]
X_train, X_test,y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=42)
## Huấn luyện mô hình
#Create a regressor object
LR= LinearRegression()

#Fit training set to the regressor
LR.fit(X_train,y_train)

print("Mô hình hồi quy tuyến tính đã được huấn luyện, có các tham số:")
print("Intercept =", LR.intercept_) # Độ chặn 
print("Coefficients:", LR.coef_) # Độ dốc
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
## Sử dụng mô hình
# Đưa ra dự đoán với biến hồi quy
y_prediction = LR.predict(X_test)
## Đánh giá mô hình
# Calculate R2-score
score=r2_score(y_test,y_prediction)
print('R2-score is ',score) # Độ ảnh hưởng
print('Mean_sqrd_error is==',mean_squared_error(y_test,y_prediction)) # Trung bình phương sai số
print('Root_mean_squared error of is==',np.sqrt(mean_squared_error(y_test,y_prediction))) # Căn bậc hai của trung bình phương sai số

