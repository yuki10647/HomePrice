# %%
from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
train =  pd.read_csv('train_pricing.csv')
test = pd.read_csv('test_pricing.csv')

# %%
all_data = pd.concat([train.drop(columns='SalePrice'),test])

# %%
num2str_list = ['MSSubClass', 'YrSold', 'MoSold']
for column in num2str_list:
    all_data[column] = all_data[column].astype(str)

for column in all_data.columns:
    if all_data[column].dtype == '0':
        all_data[column] = all_data[column].fillna('None')
    else:
        all_data[column] = all_data[column].fillna(0)

# %%
def add_new_columns(df):
    # 建物内の総面積 = 1階の面積 + 2階の面積 + 地下の面積
    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]

    # 一部屋あたりの平均面積 = 建物の総面積 / 部屋数
    df['AreaPerRoom'] = df['TotalSF']/df['TotRmsAbvGrd']

    # 築年数 + 最新リフォーム年 : この値が大きいほど値段が高くなりそう
    df['YearBuiltPlusRemod']=df['YearBuilt']+df['YearRemodAdd']

    # お風呂の総面積
    # Full bath : 浴槽、シャワー、洗面台、便器全てが備わったバスルーム
    # Half bath : 洗面台、便器が備わった部屋)(シャワールームがある場合もある)
    # シャワーがない場合を想定してHalf Bathには0.5の係数をつける
    df['TotalBathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

    # 合計の屋根付きの玄関の総面積 
    # Porch : 屋根付きの玄関 日本風にいうと縁側
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

    # プールの有無
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    # 2階の有無
    df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

    # ガレージの有無
    df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

    # 地下室の有無
    df['HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    # 暖炉の有無
    df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# カラムを追加
add_new_columns(all_data)

# %%
all_data = pd.get_dummies(all_data)
all_data

# %%
train = pd.merge(all_data.iloc[train.index[0]:train.index[-1]],train['SalePrice'],left_index=True,right_index=True)
test = all_data.iloc[train.index[-1]:]

# %%
train = train[(train['LotArea']<20000) & (train['SalePrice']<400000)& (train['YearBuilt']>1920)] # 外れ値除去
train['SalePriceLog'] = np.log(train['SalePrice'])                                               # 対数変換
# 外れ値除去、対数変換後のヒストグラム、歪度、尖度
sns.distplot(train['SalePriceLog'])
print(f"歪度: {round(train['SalePriceLog'].skew(),4)}" )
print(f"尖度: {round(train['SalePriceLog'].kurt(),4)}" )

# %%
# 学習データ、説明変数
train_X = train.drop(columns = ['SalePrice','SalePriceLog'])
# 学習データ、目的変数
train_y = train['SalePriceLog']

# 予測データ、目的変数
test = test

# %%
from sklearn.linear_model import ElasticNet as en
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
 #from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# pipeline つないでない StandardScalerなし

def get_best_estimator(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring="neg_mean_squared_error", cv=5)
    grid_model.fit(train_X, train_y)
    rmse = np.sqrt(-1*grid_model.best_score_)
    print('{0}, param:{1}, rmse:{2}'.format(model.__class__.__name__, grid_model.best_params_, np.round(rmse, 4)))
    return grid_model.best_estimator_

# 試行回数足りてないよ！というメッセージだがあまり差がみられないので今回は無視してよい？

# %%
rf_regressor_params = {'n_estimators':[1000, 2000, 3000],
                       'max_depth':[5, 6, 7, 8]}

rf_regressor_reg = RandomForestRegressor(random_state=42)

rf_regressor_be = get_best_estimator(rf_regressor_reg, rf_regressor_params)

# %%
preds = np.expm1(rf_regressor_be.predict(test))

# %%
sns.distplot(preds)