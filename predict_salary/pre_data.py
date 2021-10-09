import pandas as pd
import numpy as np

df = pd.read_csv("salarys.csv", encoding='utf8')

del df['专业']
del df['省份']
del df['城市']
df_xl = pd.get_dummies(df['学历编码'])
df_zy = pd.get_dummies(df['专业编码'])
df_wd = pd.get_dummies(df['纬度'])
df_jd = pd.get_dummies(df['经度'])
df_sf = pd.get_dummies(df['省份编码'])
df = pd.concat([df, df_xl, df_zy, df_wd, df_jd, df_sf], axis=1)

del df['学历编码']
del df['专业编码']
del df['纬度']
del df['经度']
del df['省份编码']
print(df.head())


def z_score(series):
    _mean = series.sum() / series.count()
    print(_mean)
    std = (((series - _mean) ** 2).sum() / (series.count() - 1)) ** 0.5
    print(std)
    new_series = (series - _mean) / std
    return new_series


df['综合能力'] = z_score(df['综合能力'])
print(df['综合能力'].mean())
print(df['综合能力'].std())
df.to_csv('salary_handled.csv')


np.random.seed(1314)
index = np.random.permutation(np.arange(size))
train_X = df.iloc(index[:200000])