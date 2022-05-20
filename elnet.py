# %%
# 必要モジュールのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# %%
# データの読み込み
train = pd.read_csv("C:/Users/fa-fa/Downloads/train (1).csv")
test = pd.read_csv("C:/Users/fa-fa/Downloads/test (1).csv")

# 学習データの説明変数と、予測用データを結合
all_data = pd.concat([train.drop(columns='SalePrice'),test])

#数字の大小関係が予測に影響のないデータは、数字データを文字列に変換する。
num2str_list = ['MSSubClass','YrSold','MoSold']      #MSSubClass:販売されている住居のタイプ、YrSold:売れた年、MoSOld:売れた月
for column in num2str_list:
    all_data[column] = all_data[column].astype(str)

#欠損値を処理する
for column in all_data.columns:
    if all_data[column].dtype=='O':
        all_data[column] = all_data[column].fillna('None')
    else:
        all_data[column] = all_data[column].fillna(0)

# %%
# 特徴量エンジニアリングによりカラムを追加する関数
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
    # BsmtFullBath : 地下の風呂状況
    # シャワーがない場合を想定してHalf Bathには0.5の係数をつける
    df['TotalBathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

    # 合計の屋根付きの玄関の総面積
    # Porch : 屋根付きの玄関 日本風にいうと縁側
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

    # プールの有無(lambda関数を使う): 列の値が0以上だったら、1にする。それ以外は0
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
# pd.get_dummiesを使うとカテゴリ変数化(文字列を数値に変換）
all_data = pd.get_dummies(all_data)
all_data.head()

# %%
# 学習データと予測データに分割して元のデータフレームに戻す。
train = pd.merge(all_data.iloc[train.index[0]:train.index[-1]],train['SalePrice'],left_index=True,right_index=True)
test = all_data.iloc[train.index[-1]:]

# %% 
# 目的変数
sns.distplot(train['SalePrice'])

# %%
# 正規分布に近づける。方法は外れ値を除くか、対数変換をするか。
train = train[(train['LotArea']<20000) & (train['SalePrice']<400000)& (train['YearBuilt']>1920)] # 外れ値除去
train['SalePriceLog'] = np.log(train['SalePrice'])  
sns.distplot(np.log(train['SalePriceLog']))

# %%
# 学習データ、説明変数
train_X = train.drop(columns = ['SalePrice','SalePriceLog'])
# 学習データ、目的変数
train_y = train['SalePriceLog']
# 予測データ、目的変数
test = test

# %%
def lasso_tuning(train_x,train_y):
    # alphaパラメータのリスト
    param_list = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0] 

    for cnt,alpha in enumerate(param_list):
        # パラメータを設定したラッソ回帰モデル
        lasso = Lasso(alpha=alpha) 
        # パイプライン生成
        pipeline = make_pipeline(StandardScaler(), lasso)

        # 学習データ内でホールドアウト検証のために分割 テストデータの割合は0.3 seed値を0に固定
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

        # 学習
        pipeline.fit(X_train,y_train)

        # RMSE(平均誤差)を計算
        train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train)))
        test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))
        # ベストパラメータを更新
        if cnt == 0:
            best_score = test_rmse
            best_param = alpha
        elif best_score > test_rmse:
            best_score = test_rmse
            best_param = alpha

    # ベストパラメータのalphaと、そのときのMSEを出力
    print('alpha : ' + str(best_param))
    print('test score is : ' +str(round(best_score,4)))

    # ベストパラメータを返却
    return best_param

# best_alphaにベストパラメータのalphaが渡される。
best_alpha = lasso_tuning(train_X,train_y)

# %%
# ラッソ回帰モデルにベストパラメータを設定
lasso = Lasso(alpha = best_alpha)
# パイプラインの作成
pipeline = make_pipeline(StandardScaler(), lasso)
# 学習
pipeline.fit(train_X,train_y)

# %%
pred = pipeline.predict(test)

# %%
# 予測結果のプロット
sns.distplot(pred)
# 歪度と尖度
print(f"歪度: {round(pd.Series(pred).skew(),4)}" )
print(f"尖度: {round(pd.Series(pred).kurt(),4)}" )

# %%
# 指数変換
pred_exp = np.exp(pred)
# 指数変換した予測結果をプロット
sns.distplot(pred_exp)
# 歪度と尖度
print(f"歪度: {round(pd.Series(pred_exp).skew(),4)}" )
print(f"尖度: {round(pd.Series(pred_exp).kurt(),4)}" )

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Elasticnetの使用
def get_best_estimator(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring="neg_mean_squared_error", cv=5)
    grid_model.fit(train_X, train_y)
    rmse = np.sqrt(-1*grid_model.best_score_)
    print('{0}, param:{1}, rmse:{2}'.format(model.__class__.__name__, grid_model.best_params_,\
                                            np.round(rmse, 4)))
    return grid_model.best_estimator_

params = {'alpha': [0.01, 0.1, 1, 10 ,100],
          'l1_ratio':[0.3, 0.4, 0.5, 0.6, 0.7]}

model = ElasticNet(max_iter = 10000)

elastic_be = get_best_estimator(model,params)
