# %%
# 必要モジュールのインポート
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# データの読み込み
train =  pd.read_csv('train_pricing.csv') 
train

# %%
test = pd.read_csv('test_pricing.csv')
test

# %% 
train.info()

# %%

# 学習データの説明変数と、予測用データを結合
all_df = pd.concat([train.drop(columns='SalePrice'),test])

#数字の大小関係が予測に影響のないデータは、数字データを文字列に変換する。
num2str_list = ['MSSubClass','YrSold','MoSold']      #MSSubClass:販売されている住居のタイプ、YrSold:売れた年、MoSOld:売れた月
for column in num2str_list:
    all_df[column] = all_df[column].astype(str)

#欠損値を処理する
for column in all_df.columns:
    if all_df[column].dtype=='O':
        all_df[column] = all_df[column].fillna('None')
    else:
        all_df[column] = all_df[column].fillna(0)

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
add_new_columns(all_df)


# %%
# pd.get_dummiesを使うとカテゴリ変数化(文字列を数値に変換）
all_df = pd.get_dummies(all_df)
print(all_df.head())
