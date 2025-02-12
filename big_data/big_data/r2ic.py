# 因子重要性，月度数据
import pandas as pd
import numpy as np
import os
import time
import gc
from sklearn.metrics import r2_score
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置显示中文字体
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

from ysquant.api.stock.daily_sharry_reader import DailySharryReader
from model_backtest import  backtest_main
from daily2monthly import transform_to_monthly

if __name__ == "__main__":
    dsr = DailySharryReader()
    
    # 读数据
    factor_value_dir = '/nas197/user_home/guozhaopeng/aa_data/'
    index_ = dsr.date_idx[(dsr.date_idx >= 20150101) & (dsr.date_idx <= 20241231)]
    columns_ = dsr.symbol_idx
    data_arr = np.zeros((len(os.listdir(factor_value_dir)), len(index_), len(columns_)))
    fa_list = []
    factors = os.listdir(factor_value_dir)
    for i, file in tqdm(enumerate(factors), total=len(factors)):
        df = pd.read_pickle(os.path.join(factor_value_dir, file))
        fa_list.append(file.split('.')[0])
        df = df.reindex(index=index_, columns=columns_)
        df = df.replace(0.0, np.nan)
        
        # 如果要转换成月频数据，my_cs_rank()要注释掉
        data_arr[i] = np.array(df)
        # data_arr[i] = my_cs_rank(np.array(df))    # (86, 2000+, 5000+)
        
    n_fa = len(fa_list)

    adj_close = dsr.get_field_data('adj_factor') * dsr.get_field_data('close')
    stock_ret = adj_close.shift(-10) / adj_close - 1
    label = stock_ret.reindex(index=index_, columns=columns_)  # t+10 / t - 1

    daily_data = pd.DataFrame(data_arr.reshape(n_fa, -1).T, columns=fa_list)
    daily_data['stock_code'] = columns_ * len(index_)
    daily_data['date'] = index_.repeat(len(columns_))
    daily_data['target'] = label.values.reshape(1, -1).T
    daily_data = daily_data[['stock_code', 'date', 'target'] + fa_list]
    daily_data = daily_data.dropna(subset='target')
    daily_data['date'] = pd.to_datetime(daily_data['date'].astype(str), format='%Y%m%d')
    daily_data = daily_data.sort_values(by=['stock_code', 'date'])
    daily_data = daily_data.reset_index(drop=True)

    monthly_data = transform_to_monthly(daily_data)
    
    # 回测所有结果
    folder = '/nas197/user_home/guozhaopeng/aa_results/'
    files = os.listdir(folder)
    files = [i for i in files if i.endswith('pkl')]
    res = pd.DataFrame(index=[i[:-4] for i in files], columns=['R2', 'IC'])
    # 遍历每一个因子
    for filename in tqdm(files, desc='遍历所有结果'):
        print('backtest file: ', filename)
        results = pd.read_pickle(os.path.join(folder, filename))
        
        # 合并实际收益率
        if 'monthly' in filename:
            results = results.merge(right=monthly_data[['stock_code', 'date', 'target']], on=['stock_code', 'date'], how='left')
        else:
            results = results.merge(right=daily_data[['stock_code', 'date', 'target']], on=['stock_code', 'date'], how='left')
        
        # 计算去除该因子后的r2
        r2 = r2_score(y_true=results['target'], y_pred=results['predicted_return'])
        ic = results[['predicted_return', 'target']].corr().loc['predicted_return', 'target']
        
        # 保存结果
        res.loc[filename[:-4], 'R2'] = r2
        res.loc[filename[:-4], 'IC'] = ic
        
        res.to_csv('/nas197/user_home/guozhaopeng/aa_results/results/All_results_R2IC.csv', index=True)