import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('ApartmentTrading.csv', encoding = 'UTF-8')

# Preprocessing: %dien_tich, %huong_ban_cong, phong_ngu, noi_that, %huong_nha, &lat, &long, %du_an, %id_duong, %id_phuong, %gia_m2
df['ten_quan'] = df['ten_quan'].replace('Quận Bắc Từ Liêm', 'Bắc Từ Liêm') # đồng nhất tên quận
df = df[df['ten_quan'] == 'Bắc Từ Liêm'] #filter df theo quận
df = df[df['dien_tich'].notnull()].reset_index() # 98.33% notnull
# chọn features
df = df[['dien_tich', 'huong_ban_cong', 'phong_ngu', 'noi_that', 'huong_nha', 'lat', 'long', 'du_an', 'id_duong', 'id_phuong', 'gia_m2']]

# sns.boxplot(x=df['dien_tich'])
# plt.show()
# sns.boxplot(x=df['phong_ngu'])
# plt.show()
# sns.boxplot(x=df['gia_m2'])
# plt.show()

# phong_ngu = []
# for i in df['phong_ngu']:
#     if pd.isna(i):
#         phong_ngu.append(df['dien_tich'][df['phong_ngu'][i].index()]//40)
#     else:
#         phong_ngu.append(i)
# df['phong_ngu_new'] = phong_ngu

# print(df['phong_ngu_new'])

df['phong_ngu'] = df['phong_ngu'].fillna(3)

# loại bỏ ngoại lai, dùng cột dien_tich làm tham chiếu
Q1 = df['dien_tich'].quantile(0.25)
Q3 = df['dien_tich'].quantile(0.75)
IQR = Q3 - Q1
df1 = df[~((df['dien_tich'] < (Q1 - 1.5*IQR) ) | (df['dien_tich'] > (Q3 + 1.5*IQR)))]

# loại bỏ ngoại lai, dùng cột phong_ngu làm tham chiếu
Q1_ = df1['phong_ngu'].quantile(0.25)
Q3_ = df1['phong_ngu'].quantile(0.75)
IQR = Q3_ - Q1_
df2 = df1[~((df1['phong_ngu'] < (Q1_ - 1.5*IQR) ) | (df1['phong_ngu'] > (Q3_ + 1.5*IQR)))]

# loại bỏ ngoại lai, dùng cột gia_m2 làm tham chiếu
Q1_1 = df2['gia_m2'].quantile(0.25)
Q3_1 = df2['gia_m2'].quantile(0.75)
IQR = Q3_1 - Q1_1
df3 = df2[~((df2['gia_m2'] < (Q1_1- 1.5*IQR) ) | (df2['gia_m2'] > (Q3_1 + 1.5*IQR)))]

# fill dữ liệu khuyết thiếu
df3['huong_nha'] = df3['huong_nha'].str.replace('-', ' ').fillna('KXĐ').astype('category').cat.codes
df3['huong_ban_cong'] = df3['huong_ban_cong'].str.replace('-', ' ').fillna('KXĐ').astype('category').cat.codes
df3['du_an'] = df3['du_an'].fillna('unknown').astype('category').cat.codes
df3['id_duong'] = df3['id_duong'].fillna('unknown').astype('category').cat.codes
df3['id_phuong'] = df3['id_phuong'].fillna('unknown').astype('category').cat.codes
df3['gia_m2'] = df3['gia_m2'].fillna(df3['gia_m2'].median())
df3['noi_that'] = df3['noi_that'].fillna('Nội thất cơ bản') 

df3.dropna(inplace=True)
df3.info()

# Xây dựng mô hình
X, y = df3.drop(columns=['noi_that','gia_m2']), df3['gia_m2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Đánh giá mô hình
score = r2_score(y_test,y_pred)
print('R-squared: ',score)
print('MAPE:', mean_absolute_percentage_error(y_test,y_pred))

parameters={"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
           "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }
tuning_model=GridSearchCV(regressor,param_grid=parameters,scoring='neg_mean_squared_error',cv=3,verbose=3)
tuning_model.fit(X_train, y_train)
y_pred= tuning_model.best_estimator_.predict(X_test)
print("MAPE: ",mean_absolute_percentage_error(y_test, y_pred))

print("R2 score:", r2_score(y_test, y_pred))