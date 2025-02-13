import pandas as pd
import numpy as np
from tqdm import tqdm
from ysquant.api.stock.daily_sharry_reader import DailySharryReader

# 基本面因子
fundamental_factors = ['DebtToAsset', 'CurrentRatio', 'QuickRatio', 'TotalAssetsTRate',\
                    'InventoryTRate', 'CashRateOfSales', 'EPS', 'ETOP', 'ROE', 'ROA',\
                    'TotalAssetGrowRate', 'NetProfitGrowRate', 'OperRevGrowRate',\
                    'NetCashFlowGrowRate', 'PE', 'PB', 'PS', 'PCF', 'LogCAP']

# 另类因子
other_factors = ['forecast_dps_fy1_time_linear_weight', 'forecast_dps_fy2_time_linear_weight',\
                'forecast_dps_fy3_time_linear_weight', 'forecast_eps_fy3_author_time_weight',\
                'forecast_op_fy2_author_time_weight', 'forecast_op_fy3_author_time_weight',\
                'forecast_pe_fy1_time_linear_weight', 'forecast_pe_fy2_time_linear_weight',\
                'forecast_pe_fy3_time_linear_weight', 'forecast_rd_fy1_time_linear_weight',\
                'forecast_rd_fy2_time_linear_weight', 'forecast_rd_fy3_time_linear_weight',\
                'forecast_roe_fy2_time_linear_weight', 'forecast_roe_fy3_time_linear_weight',\
                'est_pb_upperiod', 'inv_est_pb_corr_vol', 'inv_est_peg_corr_vol', 'analyst_connected_firm_reverse',\
                'funds_network_factor', 'HNpct_top12', 'HNpct_top8', 'HNpct_diff2', 'S_FELLOW_DATE_discntratio',\
                'S_FELLOW_DATE_ratio', 'repurchase_AMT', 'repurchase_QTY', 'repurchase_TOTAL_SHARE_RATIO',\
                'repurchase_S_DQ_HIGH', 'repurchase_S_DQ_LOW', 'MjrHolderTrade_buy_HOLDER_QUANTITY_NEW_RATIO',\
                'MjrHolderTrade_sell_HOLDER_QUANTITY_NEW_RATIO', 'InsiderTrade_MeanRatio', 'InsiderTrade_MaxBuyRatio']

# barra风险因子
barra_factors = ['beta', 'momentum', 'size', 'earnings_yield', 'residual_volatility', 'growth',\
                'book_to_price', 'leverage', 'liquidity', 'non_linear_size']

# 量价因子
PriceVolume_factors = ['EMA10', 'EMA20', 'EMA60', 'EMA120', 'RSI', 'KDJ_K', 'KDJ_D', 'KDJ_J',\
                    'Std20', 'Std60', 'Std120', 'DownVolatility', 'UpVolatility', 'Turnover10',\
                    'Turnover20', 'Turnover120', 'RelTurnover10', 'RelTurnover20', 'RelTurnover5',\
                    'VolEMA10', 'VolEMA20', 'VolStd20', 'VolChg20', 'VolChg5']

# 定义因子转换逻辑
def transform_to_monthly(daily_data: pd.DataFrame):
    # 获取年月
    daily_data['year_month'] = daily_data['date'].dt.to_period('M')
    
    # 按股票代码和年月分组
    grouped = daily_data.groupby(['stock_code', 'year_month'])

    # 初始化一个空的DataFrame，用于存储月频数据
    monthly_data = []

    # 遍历每个分组
    for name, group in tqdm(grouped, desc='日频数据转换为月频数据'):
        stock_code, year_month = name
        monthly_record = {'stock_code': stock_code, 'date': year_month}

        # 对每个因子进行处理
        # 基本面因子：取月末值
        for factor in fundamental_factors:
            monthly_record[factor] = group[factor].iloc[-1]
        
        # 另类因子：取月末值
        for factor in other_factors:
            monthly_record[factor] = group[factor].iloc[-1]
        
        # barra因子：取月均值
        for factor in barra_factors:
            monthly_record[factor] = group[factor].mean()
        
        # 量价因子：取月均值
        for factor in PriceVolume_factors:
            monthly_record[factor] = group[factor].mean()

        # 将当前记录添加到月频数据中
        monthly_data.append(monthly_record)

    # 将月频数据转换为DataFrame
    monthly_df = pd.DataFrame(monthly_data)
    
    # 横截面rank
    for factor in fundamental_factors+other_factors+barra_factors+PriceVolume_factors:
        monthly_df[factor] = monthly_df.groupby(by=['date'])[factor].rank(method='min', na_option='keep', pct=True)
    
    # 合并未来收益率target
    dsr = DailySharryReader()
    close = dsr.get_field_data('adj_factor') * dsr.get_field_data('close')
    close.index = close.index.astype('str')
    close.index = pd.to_datetime(close.index, format='%Y%m%d')

    monthly_close = close.resample('M').last()

    # 未来一个月收益率
    future_return = monthly_close.pct_change().shift(-1)
    future_return = future_return.stack().reset_index()
    future_return.columns = ['date', 'stock_code', 'target']
    future_return['date'] = future_return['date'].dt.to_period('M')
    
    monthly_df = pd.merge(left=monthly_df, right=future_return, on=['stock_code', 'date'], how='inner')
    
    return monthly_df