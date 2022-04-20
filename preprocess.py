# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:15:13 2022

@author: tayun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn  

#目的変数はSaleprice(住宅価格)

#学習用データの読み込み
train = pd.read_csv('data/train.csv')

#データの基本内容
train.head()
train.shape
train.columns
train.dtypes

""" 欠損値のあるパラメータ全てを表示
#それぞれのカラムの欠損値の数をリストに格納
train_null_list = list(train.isnull().sum())
for i in range(len(train.columns)):
    #欠損値のあるカラムを表示
    if train_null_list[i] > 0 :
        print(train.columns[i])
"""

"""欠損値が特徴量であるかの検証（グラフ化）

#➀欠損値の多いパラメータ（PoolQC,Alley,Fence)の欠損値をNoDataに変換
#PoolQC:プールの有無、Alley:路地の有無、Fence:フェンスの有無
train_cp = train
train_cp['PoolQC'] = train_cp["PoolQC"].fillna('NoData')
train_cp['Alley'] = train_cp["Alley"].fillna('NoData')
train_cp['Fence'] = train_cp["Fence"].fillna('NoData')

#それぞれのパラメータの住宅価格の平均値算出（例：PoolQCがExである住宅の価格の平均値）
df1 = train_cp.groupby('PoolQC').mean()
df2 = train_cp.groupby('Alley').mean()
df3 = train_cp.groupby('Fence').mean()

#グラフの描画領域の設定
fig = plt.figure(figsize=(16,6))
ax = fig.add_subplot(1,2,1)

#パラメータ別で先ほど算出した平均値をグラフ化（alphaはグラフの透過度を指定）
df1.plot.bar(y=['SalePrice'], alpha=0.6, figsize=(12,3))
df2.plot.bar(y=['SalePrice'], alpha=0.6, figsize=(12,3))
df3.plot.bar(y=['SalePrice'], alpha=0.6, figsize=(12,3))
#レイアウトを自動調整
plt.tight_layout()
plt.show()

#これらのグラフからNODataも特徴のあるデータ（住宅価格に影響を及ぼす）であることがわかった。
"""

#本格的に前処理を行う
#参考：https://qiita.com/muscle_nishimi/items/901ed94f3cdf1c8d893a
# 予測用データセットの読み込み
test = pd.read_csv('data/test.csv',index_col=0)
# 学習データの説明変数と、予測用データを結合
all_df = pd.concat([train.drop(columns='SalePrice'),test])

#➀数字の大小関係が予測に影響のないデータを文字列に変換
#MSSubClass:販売されている住居のタイプ、YrSold:売れた年、MoSOld:売れた月
num2str_list = ['MSSubClass','YrSold','MoSold']
for column in num2str_list:
    all_df[column] = all_df[column].astype(str)

#➁文字列(オブジェクト)の欠損をNone、それ以外の欠損を0で埋める。
# 変数の型ごとに欠損値の扱いが異なるため、変数ごとに処理
for column in all_df.columns:
    # dtypeがobjectの場合、文字列の変数
    if all_df[column].dtype=='O':
        all_df[column] = all_df[column].fillna('None')
    # dtypeがint , floatの場合、数字の変数
    else:
        all_df[column] = all_df[column].fillna(0)


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

# pd.get_dummiesを使うとカテゴリ変数化(文字列を数値に変換）
all_df = pd.get_dummies(all_df)
print(all_df.head())












        
    









