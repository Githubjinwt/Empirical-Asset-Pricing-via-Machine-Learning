import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
import os
import copy
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from utils import *

from joblib import Parallel, delayed
import logging
import random
import calendar

from trading_days import TradingDay

import warnings

warnings.filterwarnings('ignore')


class FactorCalculator:
    def __init__(self, target_factor_dir):
        self.target_dir = target_factor_dir
        self.td = TradingDay()  # np array
        self.date_idx = np.load("date_idx.npy")
        self.symbol_idx = np.load("symbol_idx.npy")

    def pivot_report_table(self, table, report_period=None, report_period_col='REPORT_PERIOD',
                           statement_type='408001000', index_col="opdate", field="NET_PROFIT_EXCL_MIN_INT_INC".lower(),
                           columns_col="S_INFO_WINDCODE", ann_date_col='ANN_DT', aggfunc=None):
        """
        报告期披露的数据处理，避免未来信息
        :param table: 原始数据长表
        :param report_period: 报告期，'1231','0331','0630','0930'
        :param report_period_col: 报告期对应列名
        :param statement_type:
        :param index_col: 转宽表的index对应列名，一般为数据更新获取时间
        :param field: 需要转化的数据列名
        :param columns_col: 转宽表的columns对应列名，一般为股票代码列
        :param ann_date_col: 公告日列名
        :param aggfunc:
        :return: 日频数据宽表
        """
        if 'STATEMENT_TYPE' in table.columns:
            table_tmp = table[table['STATEMENT_TYPE'] == statement_type]
        else:
            table_tmp = table.copy()

        table_tmp = table_tmp[table_tmp[ann_date_col] > 20080101]

        if report_period == '1231':
            table_tmp = table_tmp[table_tmp[report_period_col].str.contains('1231')]
            # table_tmp['last_opdate'] = table_tmp[ann_date_col].apply(lambda x:next_n_trading_days(x,21*16,td))
            table_tmp['last_opdate'] = table_tmp[ann_date_col].apply(lambda x: self.td.trading_days[
                self.td.get_loc(self.td.last_trading_day(self.td.next_trading_day(x))) + 21 * 7])

        else:
            table_tmp.loc[table_tmp[report_period_col].str[-4:] == '0331', 'last_opdate'] = table_tmp.loc[
                table_tmp[report_period_col].str[-4:] == '0331', ann_date_col].apply(lambda x: self.td.trading_days[
                self.td.get_loc(self.td.last_trading_day(self.td.next_trading_day(x))) + 21 * 5])
            table_tmp.loc[table_tmp[report_period_col].str[-4:] == '0630', 'last_opdate'] = table_tmp.loc[
                table_tmp[report_period_col].str[-4:] == '0630', ann_date_col].apply(lambda x: self.td.trading_days[
                self.td.get_loc(self.td.last_trading_day(self.td.next_trading_day(x))) + 21 * 4])
            table_tmp.loc[table_tmp[report_period_col].str[-4:] == '0930', 'last_opdate'] = table_tmp.loc[
                table_tmp[report_period_col].str[-4:] == '0930', ann_date_col].apply(lambda x: self.td.trading_days[
                self.td.get_loc(self.td.last_trading_day(self.td.next_trading_day(x))) + 21 * 7])
            table_tmp.loc[table_tmp[report_period_col].str[-4:] == '1231', 'last_opdate'] = table_tmp.loc[
                table_tmp[report_period_col].str[-4:] == '1231', ann_date_col].apply(lambda x: self.td.trading_days[
                self.td.get_loc(self.td.last_trading_day(self.td.next_trading_day(x))) + 21 * 4])
            if not table_tmp.loc[
                ~table_tmp[report_period_col].str[-4:].isin(['0331', '0630', '0930', '1231']), 'last_opdate'].empty:
                table_tmp.loc[
                    ~table_tmp[report_period_col].str[-4:].isin(['0331', '0630', '0930', '1231']), 'last_opdate'] = \
                    table_tmp.loc[
                        ~table_tmp[report_period_col].str[-4:].isin(
                            ['0331', '0630', '0930', '1231']), ann_date_col].apply(
                        lambda x: self.td.trading_days[
                            self.td.get_loc(self.td.last_trading_day(self.td.next_trading_day(x))) + 21 * 7])
        ##只保留opdate在last_opdate之前的数据
        table_tmp = table_tmp[table_tmp[index_col] < table_tmp['last_opdate']]
        ##把没有改数，只改了opdate的数据去重
        table_tmp = table_tmp.drop_duplicates([columns_col, report_period_col, field, ann_date_col], keep='first')
        update_dates = list(table_tmp[report_period_col].unique())
        update_dates.sort(reverse=True)
        result = None
        # for i in arr[1:]:
        for i, update_date in tqdm(enumerate(update_dates)):
            if report_period == '1231':
                if aggfunc is None:
                    arr = table_tmp[table_tmp[report_period_col] == update_date].pivot(index=index_col,
                                                                                       columns=columns_col,
                                                                                       values=field)
                else:
                    arr = table_tmp[table_tmp[report_period_col] == update_date].pivot_table(index=index_col,
                                                                                             columns=columns_col,
                                                                                             values=field,
                                                                                             aggfunc=aggfunc)
                arr = arr.reindex(self.td.trading_days).ffill(limit=21 * 16).reindex(self.date_idx).reindex(
                    self.symbol_idx, axis=1)
            else:
                if update_date[-4:] == '0331':
                    limit = 21 * 5  # (一)季度报告应当在每个会计年度第3个月结束后的1个月内
                elif update_date[-4:] == '0630':
                    limit = 21 * 4  # 中期报告应当在每个会计年度的上半年结束之日起2个月内
                elif update_date[-4:] == '0930':
                    limit = 21 * 7  # (三)季度报告应当在每个会计年度第9个月结束后的1个月内
                elif update_date[-4:] == '1231':
                    limit = 21 * 4  # 年度报告应当在每个会计年度结束之日起4个月内
                else:
                    limit = 21 * 7

                if aggfunc is None:
                    arr = table_tmp[table_tmp[report_period_col] == update_date].pivot(index=index_col,
                                                                                       columns=columns_col,
                                                                                       values=field)
                else:  # opdate不能晚于ann dt的一定时间，这个时间和ffill的limit保持一致
                    arr = table_tmp[table_tmp[report_period_col] == update_date].pivot_table(index=index_col,
                                                                                             columns=columns_col,
                                                                                             values=field,
                                                                                             aggfunc=aggfunc)

                arr = arr.reindex(self.td.trading_days).ffill(limit=limit).reindex(self.date_idx).reindex(
                    self.symbol_idx, axis=1)
            if i == 0:
                result = np.array(arr)
            else:
                mask = np.isnan(result)
                result[mask] = np.array(arr)[mask]
            # print(update_date)
            # print(result.shape)
            # total_data = data.copy() if i==0 else data.combine_first(total_data)
        if result is None:
            return pd.DataFrame()
        return pd.DataFrame(result, index=self.date_idx, columns=self.symbol_idx)

    # 基本面因子

    def calc_DebttoAsset(self):
        data_a = pd.read_pickle('asharebalancesheet.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_a['ann_dt'] = data_a['ann_dt'].astype(int)
        total_asset = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                              field='tot_assets', columns_col="symbol", aggfunc='last',
                                              ann_date_col='ann_dt')
        total_liability = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                                  field='tot_liab', columns_col="symbol", aggfunc='last',
                                                  ann_date_col='ann_dt')
        debt_to_asset = total_liability / total_asset
        debt_to_asset.to_pickle(os.path.join(self.target_dir, 'DebtToAsset.pkl'))

    def calc_CurrentRatio(self):
        data_a = pd.read_pickle('asharebalancesheet.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_a['ann_dt'] = data_a['ann_dt'].astype(int)
        current_asset = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                                field='tot_cur_assets', columns_col="symbol", aggfunc='last',
                                                ann_date_col='ann_dt')
        current_liability = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                                    field='tot_cur_liab', columns_col="symbol", aggfunc='last',
                                                    ann_date_col='ann_dt')
        current_ratio = current_asset / current_liability
        current_ratio.to_pickle(os.path.join(self.target_dir, 'CurrentRatio.pkl'))

    def calc_QuickRatio(self):
        data_a = pd.read_pickle('asharebalancesheet.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_a['ann_dt'] = data_a['ann_dt'].astype(int)
        current_asset = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                                field='tot_cur_assets', columns_col="symbol", aggfunc='last',
                                                ann_date_col='ann_dt')
        current_liability = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                                    field='tot_cur_liab', columns_col="symbol", aggfunc='last',
                                                    ann_date_col='ann_dt')
        inventories = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                              field='inventories', columns_col="symbol", aggfunc='last',
                                              ann_date_col='ann_dt')
        quick_ratio = (current_asset - inventories) / current_liability
        quick_ratio.to_pickle(os.path.join(self.target_dir, 'QuickRatio.pkl'))

    def calc_TotalAssetsTRate(self):
        data_a = pd.read_pickle('asharebalancesheet.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_a['ann_dt'] = data_a['ann_dt'].astype(int)
        data_c = pd.read_pickle('ashareincome.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_c['ann_dt'] = data_c['ann_dt'].astype(int)
        total_asset = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                              field='tot_assets', columns_col="symbol", aggfunc='last',
                                              ann_date_col='ann_dt')
        oper_rev = self.pivot_report_table(data_c, report_period_col='report_period', index_col="ann_dt",
                                           field='tot_oper_rev', columns_col="symbol", aggfunc='last',
                                           ann_date_col='ann_dt')
        total_asset_turnrate = oper_rev / total_asset
        total_asset_turnrate.to_pickle(os.path.join(self.target_dir, 'TotalAssetsTRate.pkl'))

    def calc_InventoryTRate(self):
        data_a = pd.read_pickle('asharebalancesheet.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_a['ann_dt'] = data_a['ann_dt'].astype(int)
        data_c = pd.read_pickle('ashareincome.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_c['ann_dt'] = data_c['ann_dt'].astype(int)
        inventories = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                              field='inventories', columns_col="symbol", aggfunc='last',
                                              ann_date_col='ann_dt')
        oper_cost = self.pivot_report_table(data_c, report_period_col='report_period', index_col="ann_dt",
                                            field='tot_oper_cost', columns_col="symbol", aggfunc='last',
                                            ann_date_col='ann_dt')
        inventory_turnrate = oper_cost / inventories
        inventory_turnrate.to_pickle(os.path.join(self.target_dir, 'InventoryTRate.pkl'))

    def calc_CashRateOfSales(self):
        data_d = pd.read_pickle('asharettmhis.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_d['ann_dt'] = data_d['ann_dt'].astype(int)
        cash_rate_of_sales = self.pivot_report_table(data_d, report_period_col='report_period', index_col="ann_dt",
                                                     field='s_fa_ocftoor_ttm', columns_col="symbol", aggfunc='last',
                                                     ann_date_col='ann_dt')
        cash_rate_of_sales.to_pickle(os.path.join(self.target_dir, 'CashRateOfSales.pkl'))

    def calc_EPS(self):
        data_c = pd.read_pickle('ashareincome.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_c['ann_dt'] = data_c['ann_dt'].astype(int)
        eps = self.pivot_report_table(data_c, report_period_col='report_period', index_col="ann_dt",
                                      field='s_fa_eps_basic',
                                      columns_col="symbol", aggfunc='last', ann_date_col='ann_dt')
        eps.to_pickle(os.path.join(self.target_dir, 'EPS.pkl'))

    def calc_ETOP(self):
        data_c = pd.read_pickle('ashareincome.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_c['ann_dt'] = data_c['ann_dt'].astype(int)
        total_mv = pd.read_pickle('market_value.pkl').reindex(index=self.date_idx, columns=self.symbol_idx)
        net_profit = self.pivot_report_table(data_c, report_period_col='report_period', index_col="ann_dt",
                                             field='net_profit_excl_min_int_inc', columns_col="symbol", aggfunc='last',
                                             ann_date_col='ann_dt')
        etop = net_profit / total_mv
        etop.to_pickle(os.path.join(self.target_dir, 'ETOP.pkl'))

    def calc_ROE(self):
        data_a = pd.read_pickle('asharebalancesheet.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_a['ann_dt'] = data_a['ann_dt'].astype(int)
        data_c = pd.read_pickle('ashareincome.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_c['ann_dt'] = data_c['ann_dt'].astype(int)
        net_profit = self.pivot_report_table(data_c, report_period_col='report_period', index_col="ann_dt",
                                             field='net_profit_excl_min_int_inc', columns_col="symbol", aggfunc='last',
                                             ann_date_col='ann_dt')
        shrhldr_eqy = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                              field='tot_shrhldr_eqy_incl_min_int', columns_col="symbol",
                                              aggfunc='last',
                                              ann_date_col='ann_dt')
        roe = net_profit / shrhldr_eqy
        roe.to_pickle(os.path.join(self.target_dir, 'ROE.pkl'))

    def calc_ROA(self):
        data_c = pd.read_pickle('ashareincome.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_c['ann_dt'] = data_c['ann_dt'].astype(int)
        net_profit = self.pivot_report_table(data_c, report_period_col='report_period', index_col="ann_dt",
                                             field='net_profit_excl_min_int_inc', columns_col="symbol", aggfunc='last',
                                             ann_date_col='ann_dt')
        data_a = pd.read_pickle('asharebalancesheet.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_a['ann_dt'] = data_a['ann_dt'].astype(int)
        total_asset = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                              field='tot_assets', columns_col="symbol", aggfunc='last',
                                              ann_date_col='ann_dt')
        roa = net_profit / total_asset
        roa.to_pickle(os.path.join(self.target_dir, 'ROA.pkl'))

    def calc_TotalAssetGrowRate(self):
        data_a = pd.read_pickle('asharebalancesheet.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_a['ann_dt'] = data_a['ann_dt'].astype(int)
        total_asset = self.pivot_report_table(data_a, report_period_col='report_period', index_col="ann_dt",
                                              field='tot_assets', columns_col="symbol", aggfunc='last',
                                              ann_date_col='ann_dt')
        tot_asset_growrate = total_asset.pct_change().replace(0.0, np.nan).ffill(limit=63)
        tot_asset_growrate.to_pickle(os.path.join(self.target_dir, 'TotalAssetGrowRate.pkl'))

    def calc_NetProfitGrowRate(self):
        data_c = pd.read_pickle('ashareincome.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_c['ann_dt'] = data_c['ann_dt'].astype(int)
        net_profit = self.pivot_report_table(data_c, report_period_col='report_period', index_col="ann_dt",
                                             field='net_profit_excl_min_int_inc', columns_col="symbol", aggfunc='last',
                                             ann_date_col='ann_dt')
        net_profit_growrate = net_profit.pct_change().replace(0.0, np.nan).ffill(limit=63)
        net_profit_growrate.to_pickle(os.path.join(self.target_dir, 'NetProfitGrowRate.pkl'))

    def calc_OperRevGrowRate(self):
        data_c = pd.read_pickle('ashareincome.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_c['ann_dt'] = data_c['ann_dt'].astype(int)
        oper_rev = self.pivot_report_table(data_c, report_period_col='report_period', index_col="ann_dt",
                                           field='tot_oper_rev', columns_col="symbol", aggfunc='last',
                                           ann_date_col='ann_dt')
        oper_rev_growrate = oper_rev.pct_change().replace(0.0, np.nan).ffill(limit=63)
        oper_rev_growrate.to_pickle(os.path.join(self.target_dir, 'OperRevGrowRate.pkl'))

    def calc_NetCashFlowGrowRate(self):
        data_b = pd.read_pickle('asharecashflow.pkl').dropna(subset=['symbol', 'ann_dt', 'report_period'])
        data_b['ann_dt'] = data_b['ann_dt'].astype(int)
        free_cash_flow = self.pivot_report_table(data_b, report_period_col='report_period', index_col="ann_dt",
                                                 field='free_cash_flow', columns_col="symbol", aggfunc='last',
                                                 ann_date_col='ann_dt')
        free_cash_flow_growrate = free_cash_flow.pct_change().replace(0.0, np.nan).ffill(limit=63)
        free_cash_flow_growrate.to_pickle(os.path.join(self.target_dir, 'NetCashFlowGrowRate.pkl'))

    def calc_PE(self):
        pe = pd.read_pickle('pe.pkl').reindex(index=self.date_idx, columns=self.symbol_idx)
        pe.to_pickle(os.path.join(self.target_dir, 'PE.pkl'))

    def calc_PB(self):
        pb = pd.read_pickle('pb.pkl').reindex(index=self.date_idx, columns=self.symbol_idx)
        pb.to_pickle(os.path.join(self.target_dir, 'PB.pkl'))

    def calc_PS(self):
        ps = pd.read_pickle('ps_ttm.pkl').reindex(index=self.date_idx, columns=self.symbol_idx)
        ps.to_pickle(os.path.join(self.target_dir, 'PS.pkl'))

    def calc_PCF(self):
        pcf = pd.read_pickle('pcf_ocf_ttm.pkl').reindex(index=self.date_idx, columns=self.symbol_idx)
        pcf.to_pickle(os.path.join(self.target_dir, 'PCF.pkl'))

    def calc_LogCAP(self):
        total_mv = pd.read_pickle('market_value.pkl').reindex(index=self.date_idx, columns=self.symbol_idx)
        log_mv = np.log(total_mv)
        log_mv.to_pickle(os.path.join(self.target_dir, 'LogCAP.pkl'))

    # 量价因子

    def calc_EMA10(self):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        ema10 = adj_close.ewm(span=10, adjust=False).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        ema10.to_pickle(os.path.join(self.target_dir, 'EMA10.pkl'))

    def calc_EMA20(self):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        ema20 = adj_close.ewm(span=20, adjust=False).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        ema20.to_pickle(os.path.join(self.target_dir, 'EMA20.pkl'))

    def calc_EMA60(self):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        ema60 = adj_close.ewm(span=60, adjust=False).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        ema60.to_pickle(os.path.join(self.target_dir, 'EMA60.pkl'))

    def calc_EMA120(self):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        ema120 = adj_close.ewm(span=120, adjust=False).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        ema120.to_pickle(os.path.join(self.target_dir, 'EMA120.pkl'))

    def calc_RSI(self):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        window = 20
        delta = adj_close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.reindex(index=self.date_idx, columns=self.symbol_idx)
        rsi.to_pickle(os.path.join(self.target_dir, 'RSI.pkl'))

    def kdj(self, window=9):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        adj_high = pd.read_pickle('high') * adj_factor
        adj_low = pd.read_pickle('low') * adj_factor
        low = adj_low.rolling(window=window).min()
        high = adj_high.rolling(window=window).max()
        rsv = (adj_close - low) / (high - low) * 100
        rsv = rsv.loc[self.td.trading_days[
                          self.td.get_loc(self.td.last_trading_day(self.td.next_trading_day(self.date_idx[0]))) - 3]:,
              :]
        k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        d = k.ewm(alpha=1 / 3, adjust=False).mean()
        j = 3 * d - 2 * k
        return k.reindex(index=self.date_idx, columns=self.symbol_idx), d.reindex(index=self.date_idx,
                                                                                  columns=self.symbol_idx), j.reindex(
            index=self.date_idx, columns=self.symbol_idx)

    def calc_KDL_K(self):
        k, _, _ = self.kdj()
        k.to_pickle(os.path.join(self.target_dir, 'KDJ_K.pkl'))

    def calc_KDJ_D(self):
        _, d, _ = self.kdj()
        d.to_pickle(os.path.join(self.target_dir, 'KDJ_D.pkl'))

    def calc_KDJ_J(self):
        _, _, j = self.kdj()
        j.to_pickle(os.path.join(self.target_dir, 'KDJ_J.pkl'))

    def calc_Std20(self):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        ret = adj_close.pct_change() * 100
        std20 = ret.rolling(window=20).std().reindex(index=self.date_idx, columns=self.symbol_idx)
        std20.to_pickle(os.path.join(self.target_dir, 'Std20.pkl'))

    def calc_Std60(self):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        ret = adj_close.pct_change() * 100
        std60 = ret.rolling(window=60).std().reindex(index=self.date_idx, columns=self.symbol_idx)
        std60.to_pickle(os.path.join(self.target_dir, 'Std60.pkl'))

    def calc_Std120(self):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        ret = adj_close.pct_change() * 100
        std120 = ret.rolling(window=120).std().reindex(index=self.date_idx, columns=self.symbol_idx)
        std120.to_pickle(os.path.join(self.target_dir, 'Std120.pkl'))

    def calc_Turnover10(self):
        turnover = pd.read_pickle('turnover_ratio.pkl')
        turnover10 = turnover.rolling(window=10).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        turnover10.to_pickle(os.path.join(self.target_dir, 'Turnover10.pkl'))

    def calc_Turnover20(self):
        turnover = pd.read_pickle('turnover_ratio.pkl')
        turnover20 = turnover.rolling(window=20).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        turnover20.to_pickle(os.path.join(self.target_dir, 'Turnover20.pkl'))

    def calc_Turnover120(self):
        turnover = pd.read_pickle('turnover_ratio.pkl')
        turnover120 = turnover.rolling(window=120).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        turnover120.to_pickle(os.path.join(self.target_dir, 'Turnover120.pkl'))

    def calc_RelTurnover10(self):
        turnover = pd.read_pickle('turnover_ratio.pkl')
        turnover10 = turnover.rolling(window=10).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        turnover120 = turnover.rolling(window=120).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        relturnover10 = turnover10 / turnover120
        relturnover10.to_pickle(os.path.join(self.target_dir, 'RelTurnover10.pkl'))

    def calc_RelTurnover20(self):
        turnover = pd.read_pickle('turnover_ratio.pkl')
        turnover20 = turnover.rolling(window=20).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        turnover120 = turnover.rolling(window=120).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        relturnover20 = turnover20 / turnover120
        relturnover20.to_pickle(os.path.join(self.target_dir, 'RelTurnover20.pkl'))

    def calc_RelTurnover5(self):
        turnover = pd.read_pickle('turnover_ratio.pkl')
        turnover5 = turnover.rolling(window=5).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        turnover120 = turnover.rolling(window=120).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        relturnover5 = turnover5 / turnover120
        relturnover5.to_pickle(os.path.join(self.target_dir, 'RelTurnover5.pkl'))

    def calc_VolEMA20(self):
        volume = pd.read_pickle('volume.pkl')
        vol20 = volume.ewm(span=20, adjust=False).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        vol20.to_pickle(os.path.join(self.target_dir, 'VolEMA20.pkl'))

    def calc_VolEMA10(self):
        volume = pd.read_pickle('volume.pkl')
        vol10 = volume.ewm(span=10, adjust=False).mean().reindex(index=self.date_idx, columns=self.symbol_idx)
        vol10.to_pickle(os.path.join(self.target_dir, 'VolEMA10.pkl'))

    def calc_VolStd20(self):
        volume = pd.read_pickle('volume.pkl')
        volstd20 = volume.rolling(window=20).std().reindex(index=self.date_idx, columns=self.symbol_idx)
        volstd20.to_pickle(os.path.join(self.target_dir, 'VolStd20.pkl'))

    def calc_VolChg20(self):
        volume = pd.read_pickle('volume.pkl')
        volchg20 = (volume / volume.shift(1).rolling(window=20).mean()).reindex(index=self.date_idx,
                                                                                columns=self.symbol_idx)
        volchg20.to_pickle(os.path.join(self.target_dir, 'VolChg20.pkl'))

    def calc_VolChg5(self):
        volume = pd.read_pickle('volume.pkl')
        volchg5 = (volume / volume.shift(1).rolling(window=5).mean()).reindex(index=self.date_idx,
                                                                              columns=self.symbol_idx)
        volchg5.to_pickle(os.path.join(self.target_dir, 'VolChg5.pkl'))

    @staticmethod
    def positive_volatility(x):
        positive_returns = x[x > 0]
        if len(positive_returns) >= 2:
            return np.std(positive_returns, ddof=1)
        else:
            return np.nan

    @staticmethod
    def negtive_volatility(x):
        negtive_returns = x[x < 0]
        if len(negtive_returns) >= 2:
            return np.std(negtive_returns, ddof=1)
        else:
            return np.nan

    def calc_UpVolatility(self):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        ret = adj_close.pct_change() * 100
        up_volatility = ret.loc[20141001:, :].rolling(window=60, min_periods=2).apply(self.positive_volatility).reindex(
            index=self.date_idx, columns=self.symbol_idx)
        up_volatility.to_pickle(os.path.join(self.target_dir, 'UpVolatility.pkl'))

    def calc_DownVolatility(self):
        close = pd.read_pickle('close.pkl')
        adj_factor = pd.read_pickle('adj_factor.pkl')
        adj_close = close * adj_factor
        ret = adj_close.pct_change() * 100
        down_volatility = ret.loc[20141001:, :].rolling(window=60, min_periods=2).apply(
            self.negtive_volatility).reindex(index=self.date_idx, columns=self.symbol_idx)
        down_volatility.to_pickle(os.path.join(self.target_dir, 'DownVolatility.pkl'))

    # 另类因子

    @staticmethod
    def author_weight(data_long, field, index_, columns_):
        tmp = data_long.copy(deep=True)
        tmp[field] = tmp[field] * tmp['weight']
        pivot_data = pd.pivot_table(tmp, index='date', columns='stock_code', values=field, aggfunc='sum')
        pivot_weight = pd.pivot_table(tmp, index='date', columns='stock_code', values='weight', aggfunc='sum')
        aa = pivot_data / pivot_weight
        aa = aa.reindex(index=index_, columns=columns_)
        return aa

    @staticmethod
    def time_weight(valid_data):
        # 计算时间加权平均
        time_weight = np.broadcast_to(np.arange(63, 0, -1)[:, np.newaxis, np.newaxis], valid_data.shape)
        time_weight = np.where(np.isnan(valid_data), np.nan, time_weight)
        time_weight = time_weight / np.nansum(time_weight, axis=0)
        return time_weight

    @staticmethod
    def get_valid_data(data_long, data_pivot, index_, columns_):
        pivot_enddate = pd.pivot_table(data_long, index='date', columns='stock_code', values='valid_enddate',
                                       aggfunc='min')
        pivot_enddate = pivot_enddate.reindex(index=index_, columns=columns_)

        for i in range(63):
            pivot_enddate_shift = pivot_enddate.shift(i)
            mask_shift = (pivot_enddate_shift >= pivot_enddate_shift.index[:, None])  # 提取有效数据（当前日期在年报发布之前）
            data_shift = data_pivot.shift(i)
            data_shift = data_shift[mask_shift]
            valid_data = np.array([data_shift]) if i == 0 else np.concatenate(
                (valid_data, data_shift.values[np.newaxis, :, :]), axis=0)
        return valid_data

    def get_gogoal_factor(self, fy, field, mode):
        """
        手动构建分析师预期数据
        :param fy: 预测跨度
        :param field: 字段名
        :param mode: 权重规则，0：等权；1：时间加权；3：分析师加权；4：时间+分析师加权
        :return:
        """
        # 读原始数据
        crystal_aut = pd.read_pickle('der_crystalball_author.pkl')
        newfortune_aut = pd.read_pickle('der_new_fortune_author.pkl')
        rpt_forecast = pd.read_pickle('rpt_forecast_stk.pkl')
        rpt_report_author = pd.read_pickle('rpt_report_author.pkl')
        actual_rpt_date = pd.read_pickle("定期报告披露日期.pkl")
        # 数据预处理
        rpt_forecast['create_date'] = pd.to_datetime(rpt_forecast['create_date'].astype(str))
        rpt_forecast['updatetime'] = pd.to_datetime(rpt_forecast['updatetime'].dt.strftime('%Y-%m-%d'))
        rpt_forecast['date'] = np.where(rpt_forecast['create_date'] >= '2017-04-12', rpt_forecast['updatetime'],
                                        rpt_forecast['create_date'])
        rpt_forecast = rpt_forecast[(rpt_forecast['is_valid'] == 1) &
                                    ((rpt_forecast['reliability'] == 0) | (rpt_forecast['reliability'] >= 5))]
        rpt_forecast['stock_code'] = rpt_forecast['stock_code'].astype(str).str.zfill(6)
        rpt_forecast = rpt_forecast.replace(0.0, np.nan)
        rpt_forecast = rpt_forecast.dropna(subset=['report_year', 'report_quarter'])
        rpt_forecast['report_year'] = rpt_forecast['report_year'].astype(int)
        rpt_forecast['report_quarter'] = rpt_forecast['report_quarter'].astype(int)
        rpt_forecast = rpt_forecast[rpt_forecast['report_quarter'] == 4]

        actual_rpt_date['S_STM_ACTUAL_ISSUINGDATE'] = pd.to_datetime(
            actual_rpt_date['S_STM_ACTUAL_ISSUINGDATE'].astype(str), format='%Y%m%d')
        actual_rpt_date = actual_rpt_date[actual_rpt_date['REPORT_PERIOD'].astype(str).str[-4:] == '1231']
        actual_rpt_date = actual_rpt_date[actual_rpt_date['S_STM_ACTUAL_ISSUINGDATE'] <= (
                (actual_rpt_date['REPORT_PERIOD'].str[:4].astype(int) + 1).astype(str) + '-04-30')]
        actual_rpt_date = actual_rpt_date.drop_duplicates(subset=['S_INFO_WINDCODE', 'REPORT_PERIOD'])
        actual_rpt_date = actual_rpt_date[['S_INFO_WINDCODE', 'REPORT_PERIOD', 'S_STM_ACTUAL_ISSUINGDATE']]
        rpt_forecast = rpt_forecast.sort_values(by=['date'])
        actual_rpt_date = actual_rpt_date.sort_values(by=['S_STM_ACTUAL_ISSUINGDATE'])
        rpt_forecast = pd.merge_asof(rpt_forecast,
                                     actual_rpt_date[['S_INFO_WINDCODE', 'S_STM_ACTUAL_ISSUINGDATE', 'REPORT_PERIOD']],
                                     left_by='stock_code', right_by='S_INFO_WINDCODE', left_on='date',
                                     right_on='S_STM_ACTUAL_ISSUINGDATE', direction='backward')
        rpt_forecast = rpt_forecast.dropna(subset='S_INFO_WINDCODE')
        rpt_forecast['REPORT_PERIOD'] = rpt_forecast['REPORT_PERIOD'].astype(str).str[:4].astype(int)
        actual_rpt_date['report_year'] = actual_rpt_date['REPORT_PERIOD'].astype(str).str[:4].astype(int)
        actual_rpt_date['valid_enddate'] = actual_rpt_date['S_STM_ACTUAL_ISSUINGDATE']
        rpt_forecast = pd.merge(rpt_forecast, actual_rpt_date[['S_INFO_WINDCODE', 'report_year', 'valid_enddate']],
                                on=['S_INFO_WINDCODE', 'report_year'], how='left')
        rpt_forecast.loc[rpt_forecast['valid_enddate'].isna(), 'valid_enddate'] = pd.to_datetime('2099-12-31')
        rpt_forecast['fore_type'] = rpt_forecast['report_year'] - rpt_forecast['REPORT_PERIOD']

        rpt_report_author = rpt_report_author[
            (rpt_report_author['is_main'] == 1) & (rpt_report_author['is_valid'] == 1)]
        rpt_forecast = pd.merge(rpt_forecast,
                                rpt_report_author[['report_id', 'author_id', 'organ_id', 'organ_name', 'author']],
                                on='report_id')
        newfortune_aut = newfortune_aut[newfortune_aut['is_valid'] == 1]
        crystal_aut = crystal_aut[crystal_aut['is_valid'] == 1]
        important_aut = pd.concat([newfortune_aut, crystal_aut])[
            ['report_year', 'author_id', 'author', 'prize_awarded']]
        important_aut.loc[important_aut['prize_awarded'] == '第一名', 'prize_awarded'] = 1
        important_aut.loc[important_aut['prize_awarded'] == '第二名', 'prize_awarded'] = 2
        important_aut.loc[important_aut['prize_awarded'] == '第三名', 'prize_awarded'] = 3
        important_aut.loc[important_aut['prize_awarded'] == '第四名', 'prize_awarded'] = 4
        important_aut.loc[important_aut['prize_awarded'] == '第五名', 'prize_awarded'] = 5
        important_aut.loc[~important_aut['prize_awarded'].isin(range(1, 6)), 'prize_awarded'] = 6
        important_aut = important_aut.sort_values(by=['prize_awarded'])
        important_aut = important_aut.drop_duplicates(subset=['report_year', 'author_id', 'author'],
                                                      keep='first')  # 保留同年同一分析师的最高奖项
        important_aut = important_aut.rename(columns={'report_year': 'award_year'})
        rpt_forecast['year_before_create'] = rpt_forecast['date'].dt.year.astype(int) - 1
        rpt_forecast = pd.merge(rpt_forecast, important_aut, left_on=['year_before_create', 'author_id', 'author'],
                                right_on=['award_year', 'author_id', 'author'], how='left')
        rpt_forecast = rpt_forecast[rpt_forecast['award_year'].notna()]
        rpt_forecast['date'] = rpt_forecast['date'].dt.strftime('%Y%m%d').astype(int)
        rpt_forecast['valid_enddate'] = rpt_forecast['valid_enddate'].dt.strftime('%Y%m%d').astype(int)

        index_1 = self.date_idx[
            (self.date_idx >= self.td.trading_days[
                self.td.get_loc(self.td.last_trading_day(self.td.next_trading_day(self.date_idx[0]))) - 63]) & (
                    self.date_idx <= self.date_idx[-1])]
        columns_ = self.symbol_idx
        sub_data = rpt_forecast[(rpt_forecast['date'] >= index_1[0]) & (rpt_forecast['date'] <= self.date_idx[-1])]
        tmp = sub_data[sub_data['fore_type'] == fy].dropna(subset=field)
        if mode == 0:  # 等权
            aa = pd.pivot_table(tmp, index='date', columns='stock_code', values=field, aggfunc='mean')
            aa = aa.reindex(index=index_1, columns=columns_)
            valid_data = self.get_valid_data(tmp, aa, index_1, columns_)
            factor = np.nanmean(valid_data, axis=0)
            factor = pd.DataFrame(factor, index=index_1, columns=columns_).reindex(index=self.date_idx)
        elif mode == 1:  # 时间加权
            aa = pd.pivot_table(tmp, index='date', columns='stock_code', values=field, aggfunc='mean')
            aa = aa.reindex(index=index_1, columns=columns_)
            valid_data = self.get_valid_data(tmp, aa, index_1, columns_)
            time_weight = self.time_weight(valid_data)
            factor = np.nansum(valid_data * time_weight, axis=0)
            factor = pd.DataFrame(factor, index=index_1, columns=columns_).reindex(index=self.date_idx)
        elif mode == 2:  # 分析师加权
            tmp2 = tmp[tmp['prize_awarded'].isin(range(1, 6))]
            tmp2['weight'] = 6 - tmp2['prize_awarded']
            aa = self.author_weight(tmp2, field, index_1, columns_)
            valid_data = self.get_valid_data(tmp2, aa, index_1, columns_)
            factor = np.nanmean(valid_data, axis=0)
            factor = pd.DataFrame(factor, index=index_1, columns=columns_).reindex(index=self.date_idx)
        elif mode == 3:  # 时间+分析师加权
            tmp2 = tmp[tmp['prize_awarded'].isin(range(1, 6))]
            tmp2['weight'] = 6 - tmp2['prize_awarded']
            aa = self.author_weight(tmp2, field, index_1, columns_)
            valid_data = self.get_valid_data(tmp2, aa, index_1, columns_)
            time_weight = self.time_weight(valid_data)
            factor = np.nansum(valid_data * time_weight, axis=0)
            factor = pd.DataFrame(factor, index=index_1, columns=columns_).reindex(index=self.date_idx)
        return factor

    def calc_forecast_dps_fy1_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=1, field='dps', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_dps_fy1_time_linear_weight.pkl'))

    def calc_forecast_dps_fy2_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=2, field='dps', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_dps_fy2_time_linear_weight.pkl'))

    def calc_forecast_dps_fy3_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=3, field='dps', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_dps_fy3_time_linear_weight.pkl'))

    def calc_forecast_eps_fy3_author_time_weight(self):
        data = self.get_gogoal_factor(fy=3, field='eps', mode=3)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_eps_fy3_author_time_weight.pkl'))

    def calc_forecast_op_fy2_author_time_weight(self):
        data = self.get_gogoal_factor(fy=2, field='op', mode=3)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_op_fy2_author_time_weight.pkl'))

    def calc_forecast_op_fy3_author_time_weight(self):
        data = self.get_gogoal_factor(fy=3, field='op', mode=3)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_op_fy3_author_time_weight.pkl'))

    def calc_forecast_pe_fy1_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=1, field='pe', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_pe_fy1_time_linear_weight.pkl'))

    def calc_forecast_pe_fy2_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=2, field='pe', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_pe_fy2_time_linear_weight.pkl'))

    def calc_forecast_pe_fy3_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=3, field='pe', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_pe_fy3_time_linear_weight.pkl'))

    def calc_forecast_rd_fy1_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=1, field='rd', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_rd_fy1_time_linear_weight.pkl'))

    def calc_forecast_rd_fy2_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=2, field='rd', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_rd_fy2_time_linear_weight.pkl'))

    def calc_forecast_rd_fy3_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=3, field='rd', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_rd_fy3_time_linear_weight.pkl'))

    def calc_forecast_roe_fy2_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=2, field='roe', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_roe_fy2_time_linear_weight.pkl'))

    def calc_forecast_roe_fy3_time_linear_weight(self):
        data = self.get_gogoal_factor(fy=3, field='roe', mode=1)
        data.to_pickle(os.path.join(self.target_dir, 'forecast_roe_fy3_time_linear_weight.pkl'))

    def neut_factor(self, factor, mv_df, industry_df):
        factor_value = factor.reindex_like(mv_df)
        adj_factor = mv_Neutral(factor_value, mv_df)
        adj_factor = pd.DataFrame(industryNeutral(adj_factor, industry_df),
                                  index=factor_value.index, columns=factor_value.columns)
        return adj_factor

    def calc_est_pb_upperiod(self):
        est_pb = pd.read_pickle('con_pb_1.pkl')
        est_pb.fillna(method="pad", inplace=True)
        adj_close = pd.read_pickle('close.pkl') * pd.read_pickle('adj_factor.pkl')
        adj_close.fillna(method="pad", inplace=True)
        pct_chg = adj_close.pct_change()
        mask = pct_chg > 0
        est_pb_up = est_pb[mask]
        volume = pd.read_pickle('volume.pkl')[mask]
        factor = (est_pb_up * volume).rolling(window=1010, min_periods=1).sum() / \
                 volume.rolling(window=10, min_periods=1).sum()
        mv_value = pd.read_pickle('market_value.pkl')
        df_isin_industry = pd.read_pickle('industry_df.pkl')
        factor = self.neut_factor(factor, mv_value, df_isin_industry)
        factor.to_pickle(os.path.join(self.target_dir, 'est_pb_upperiod.pkl'))

    def calc_inv_est_pb_corr_vol(self):
        est_pb = pd.read_pickle('con_pb_0.pkl')
        inv_est_pb = 1 / est_pb
        inv_est_pb.fillna(1)
        volume = pd.read_pickle('volume.pkl')
        corrcoef = inv_est_pb.rolling(10).corr(volume)
        factor = corrcoef.rolling(10).mean()
        mv_value = pd.read_pickle('market_value.pkl')
        df_isin_industry = pd.read_pickle('industry_df.pkl')
        factor = self.neut_factor(factor, mv_value, df_isin_industry)
        factor.to_pickle(os.path.join(self.target_dir, 'inv_est_pb_corr_vol.pkl'))

    def calc_inv_est_peg_corr_vol(self):
        est_pb = pd.read_pickle('con_peg_1.pkl')
        inv_est_pb = 1 / est_pb
        inv_est_pb.fillna(1)
        volume = pd.read_pickle('volume.pkl')
        corrcoef = inv_est_pb.rolling(10).corr(volume)
        factor = corrcoef.rolling(20).mean()
        mv_value = pd.read_pickle('market_value.pkl')
        df_isin_industry = pd.read_pickle('industry_df.pkl')
        factor = self.neut_factor(factor, mv_value, df_isin_industry)
        factor.to_pickle(os.path.join(self.target_dir, 'inv_est_peg_corr_vol.pkl'))

    def calc_analyst_connected_firm_reverse(self):
        crystal_aut = pd.read_pickle('der_crystalball_author.pkl')
        newfortune_aut = pd.read_pickle('der_new_fortune_author.pkl')
        rpt_forecast = pd.read_pickle('rpt_forecast_stk.pkl')
        rpt_report_author = pd.read_pickle('rpt_report_author.pkl')
        actual_rpt_date = pd.read_pickle("/nas197/user_home/xilin/底层数据/long/定期报告披露日期.pkl")
        rpt_forecast['create_date'] = pd.to_datetime(rpt_forecast['create_date'].astype(str))
        rpt_forecast['updatetime'] = pd.to_datetime(rpt_forecast['updatetime'].dt.strftime('%Y-%m-%d'))
        rpt_forecast['date'] = np.where(rpt_forecast['create_date'] >= '2017-04-12', rpt_forecast['updatetime'],
                                        rpt_forecast['create_date'])
        rpt_forecast = rpt_forecast[rpt_forecast['is_valid'] == 1]
        rpt_forecast = rpt_forecast[(rpt_forecast['reliability'] == 0) | (rpt_forecast['reliability'] >= 5)]
        rpt_forecast['stock_code'] = rpt_forecast['stock_code'].astype(str).str.zfill(6)
        rpt_forecast = rpt_forecast.replace(0.0, np.nan)
        rpt_forecast = rpt_forecast.dropna(subset=['report_year', 'report_quarter'])
        rpt_forecast['report_year'] = rpt_forecast['report_year'].astype(int)
        rpt_forecast['report_quarter'] = rpt_forecast['report_quarter'].astype(int)
        rpt_forecast = rpt_forecast[rpt_forecast['report_quarter'] == 4]
        rpt_report_author = rpt_report_author[
            (rpt_report_author['is_main'] == 1) & (rpt_report_author['is_valid'] == 1)]
        rpt_forecast = pd.merge(rpt_forecast, rpt_report_author[['report_id', 'author_id', 'author']], on='report_id')
        newfortune_aut = newfortune_aut[newfortune_aut['is_valid'] == 1]
        crystal_aut = crystal_aut[crystal_aut['is_valid'] == 1]
        important_aut = pd.concat([newfortune_aut, crystal_aut])[
            ['report_year', 'author_id', 'author', 'prize_awarded']]
        important_aut = important_aut.drop_duplicates(subset=['report_year', 'author_id', 'author'])
        important_aut = important_aut.rename(columns={'report_year': 'award_year'})
        rpt_forecast['year_before_create'] = rpt_forecast['date'].dt.year.astype(int) - 1
        rpt_forecast = pd.merge(rpt_forecast, important_aut, left_on=['year_before_create', 'author_id', 'author'],
                                right_on=['award_year', 'author_id', 'author'], how='left')
        rpt_forecast = rpt_forecast[rpt_forecast['award_year'].notna()]
        rpt_forecast = rpt_forecast[['stock_code', 'date', 'author_id', 'author']].drop_duplicates()
        rpt_forecast['date'] = pd.to_datetime(rpt_forecast['date']).dt.strftime('%Y%m%d').astype(int)
        rpt_forecast['is_coveraged'] = 1
        coverage_rolling = np.zeros(
            shape=(len(rpt_forecast[['author_id', 'author']].drop_duplicates()), len(self.date_idx), len(self.symbol_idx)))
        for i, (author_id, author) in tqdm(enumerate(rpt_forecast[['author_id', 'author']].drop_duplicates().values)):
            subdata = rpt_forecast[(rpt_forecast['author_id'] == author_id) & (rpt_forecast['author'] == author)]
            pivot_df = pd.pivot_table(subdata, index='date', columns='stock_code', values='is_coveraged',
                                      aggfunc='first')
            pivot_df = pivot_df.reindex(index=self.date_idx, columns=self.symbol_idx)
            pivot_df = (pivot_df.rolling(window=126, min_periods=1).sum() >= 1).astype(int)
            coverage_rolling[i] = np.array(pivot_df)
        coverage_rolling_T = np.transpose(coverage_rolling, (1, 0, 2))
        network = np.matmul(np.transpose(coverage_rolling_T, (0, 2, 1)), coverage_rolling_T)
        m, n = np.diag_indices(network.shape[1])
        network[:, m, n] = 0

        adj_price = pd.read_pickle('adj_factor.pkl') * pd.read_pickle('close.pkl')
        ret = adj_price.shift(1) / adj_price.shift(31) - 1  # t-1 / t-31 过去30日收益率，防止t日获取不到数据造成未来信息
        ret = ret.reindex(index=self.date_idx, columns=self.symbol_idx).fillna(0)
        network_batch = np.log(network + 1)
        ret_arr = np.array(ret)[:, :, np.newaxis]
        up = np.matmul(network_batch, ret_arr).squeeze()
        down = np.sum(network_batch, axis=2)
        factor1 = pd.DataFrame(up / down, index=self.date_idx, columns=self.symbol_idx).replace(0.0, np.nan)

        factor2 = factor1 - ret.reindex_like(factor1)
        factor2.to_pickle(os.path.join(self.target_dir, 'analyst_connected_firm_reverse.pkl'))

    def calc_funds_network_factor(self):
        data = pd.read_pickle('机构持股衍生数据.pkl')
        data = data.drop_duplicates(
            ['OBJECT_ID', 'S_INFO_WINDCODE', 'REPORT_PERIOD', 'S_HOLDER_QUANTITY', 'S_HOLDER_PCT', 'S_FLOAT_A_SHR',
             'ANN_DATE'], keep='first')
        data = data[data.S_HOLDER_HOLDERCATEGORY == '基金']
        close = pd.read_pickle('close.pkl')
        amt20 = pd.read_pickle('turnover.pkl').rolling(window=20).mean()
        adj_close = close * pd.read_pickle('adj_factor.pkl')
        pctchg20 = adj_close.pct_change().rolling(20).sum()

        pivot_Q = self.pivot_report_table(data, report_period_col='REPORT_PERIOD', index_col="opdate",
                                          field='S_HOLDER_QUANTITY', columns_col="S_INFO_WINDCODE", aggfunc='sum',
                                          ann_date_col='ANN_DATE')
        pivot_Q = pivot_Q.reindex(index=self.date_idx, columns=self.symbol_idx)
        sub_close = close.reindex(index=self.date_idx, columns=self.symbol_idx)
        sub_amt20 = amt20.reindex(index=self.date_idx, columns=self.symbol_idx)
        pivot_I = pivot_Q * sub_close / sub_amt20
        I_arr = np.array(pivot_I)
        network = np.minimum(I_arr[:, :, np.newaxis], I_arr[:, np.newaxis, :])

        pctchg20 = pctchg20.reindex(index=self.date_idx, columns=self.symbol_idx)
        chg = np.array(pctchg20)
        connected_stocks = np.any(~np.isnan(network), axis=2)
        med = np.nanmedian(np.where(connected_stocks, chg, np.nan), axis=1)
        alpha = chg - med[:, np.newaxis]
        alpha_df = pd.DataFrame(alpha, index=self.date_idx, columns=self.symbol_idx)
        exp = alpha[:, np.newaxis, :] * network
        exp_avg = np.nanmean(exp, axis=2)
        factor_noneut = pd.DataFrame(exp_avg, index=self.date_idx, columns=self.symbol_idx)
        # 横截面的alpha和行业中性化(市值替换为alpha)
        df_isin_industry = pd.read_pickle('industry_df.pkl')
        factor_neut = self.neut_factor(factor_noneut, alpha_df, df_isin_industry)
        factor_neut.to_pickle(os.path.join(self.target_dir, 'funds_network_factor.pkl'))

    @staticmethod
    def get_hnpct_nth_one_symbol(code, pctchg, pivot_data, n):
        nth_dates = pctchg[code].rolling(window=12, min_periods=1).apply(lambda x: x.sort_values(ascending=False).index[n] if len(x) >= n+1 else np.nan)
        aa = pivot_data.loc[nth_dates.dropna().astype(int).values, code]
        aa.index = nth_dates.dropna().index
        aa = aa.reindex(nth_dates.index)
        return aa

    def get_HNpct_factor(self):
        data = pd.read_pickle('股东户数.pkl')

        months_end = pd.date_range(start=pd.to_datetime(str(self.date_idx[0] - 20000), format='%Y%m%d'),
                                   end=pd.to_datetime(str(self.date_idx[-1]), format='%Y%m%d'), freq='M')  # 开始日期2年前至结束日期的每个月最后一天
        dates_df = pd.DataFrame({'S_HOLDER_ENDDATE': months_end})
        dates_df['S_HOLDER_ENDDATE'] = dates_df['S_HOLDER_ENDDATE'].dt.strftime('%Y%m%d').astype(int)
        dates_df['last_td_day'] = dates_df['S_HOLDER_ENDDATE'].apply(
            lambda x: self.td.last_trading_day(self.td.next_trading_day(x)))
        filtered_data = data[data['S_HOLDER_ENDDATE'].astype(int).isin(dates_df['S_HOLDER_ENDDATE'])]
        filtered_data['S_HOLDER_ENDDATE'] = filtered_data['S_HOLDER_ENDDATE'].astype(int)
        filtered_data['last_td_day'] = filtered_data['S_HOLDER_ENDDATE'].apply(
            lambda x: self.td.last_trading_day(self.td.next_trading_day(x)))
        HN_pivot = pd.pivot_table(filtered_data, index='last_td_day', columns='S_INFO_WINDCODE', values='S_HOLDER_NUM')
        HN_pivot = HN_pivot.reindex(index=dates_df['last_td_day'], columns=self.symbol_idx)
        HN_pivot = HN_pivot.ffill(limit=12)
        HNpct_pivot = HN_pivot.pct_change().replace(0, np.nan).ffill(limit=12)
        absHNpct_pivot = abs(HNpct_pivot)
        adj_close = pd.read_pickle('close.pkl') * pd.read_pickle('adj_factor.pkl')
        dates = adj_close[(adj_close.index >= self.date_idx[0] - 20000) & (adj_close.index <= self.date_idx[-1])].index
        adj_close = adj_close.reindex(index=dates_df['last_td_day'], columns=self.symbol_idx)
        pctchg = abs(adj_close.pct_change())
        opdate_pivot = pd.pivot_table(filtered_data, index='last_td_day', columns='S_INFO_WINDCODE', values='opdate')
        opdate_pivot = opdate_pivot.reindex(index=dates, columns=self.symbol_idx).ffill(limit=140)
        mask = opdate_pivot <= opdate_pivot.index[:, None]

        def get_hnpct_nth(pivot_data, n, startdate, enddate):
            # 从1到n股价振幅逐渐增大
            partial_func = partial(self.get_hnpct_nth_one_symbol, pctchg=pctchg, pivot_data=pivot_data,
                                   n=n)  # 并行计算每个symbol
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(partial_func, list(pctchg.columns))
            HNpct_nth = pd.concat(results, axis=1)
            HNpct_nth = HNpct_nth.reindex(index=dates).ffill(limit=140)
            HNpct_nth = HNpct_nth[mask].ffill(limit=140)
            HNpct_nth = HNpct_nth[(HNpct_nth.index >= startdate) & (HNpct_nth.index <= enddate)]
            return HNpct_nth

        factor_data = {}
        for i in range(1, 13):
            HNpct_nth = get_hnpct_nth(HNpct_pivot, i - 1, self.date_idx[0], self.date_idx[-1])
            factor_data[f'HNpct_{i}th'] = HNpct_nth
        return factor_data

    def calc_HNpct_top8(self):
        factor_data = self.get_HNpct_factor()
        for j in range(8):
            factor = factor_data[f'HNpct_{12 - j}th'] if j == 0 else factor + factor_data[f'HNpct_{12 - j}th']
        factor.to_pickle(os.path.join(self.target_dir, 'HNpct_top8.pkl'))

    def calc_HNpct_top12(self):
        factor_data = self.get_HNpct_factor()
        for j in range(12):
            factor = factor_data[f'HNpct_{12 - j}th'] if j == 0 else factor + factor_data[f'HNpct_{12 - j}th']
        factor.to_pickle(os.path.join(self.target_dir, 'HNpct_top12.pkl'))

    def calc_HNpct_diff2(self):
        factor_data = self.get_HNpct_factor()
        for j in range(2):
            factor = factor_data[f'HNpct_{12 - j}th'] if j == 0 else factor + factor_data[f'HNpct_{12 - j}th']
        for j in range(10):
            factor = factor - factor_data[f'HNpct_{j + 1}th']
        factor.to_pickle(os.path.join(self.target_dir, 'HNpct_diff2.pkl'))

    def get_barra_distance(self, start_dt, last_dt):
        def parallel_read(dt):
            return pd.read_feather(f"barra_distance/{dt}.ipc").\
                reindex(index=range(len(self.symbol_idx)), columns=self.symbol_idx).fillna(0).astype(np.float32).values
        dates = TradingDay().get_range(start_dt, last_dt)
        temp = np.array([Parallel(n_jobs=200, verbose=5)(delayed(parallel_read)(dt) for dt in tqdm(dates))])[0]
        return temp

    def event_overflow_3d(self, event, temp, dates):
        return pd.DataFrame(np.einsum('npq,np->nq', temp, event.reindex(dates).fillna(0).values), index=dates,
                            columns=self.symbol_idx)

    def calc_S_FELLOW_DATE_discntratio(self):
        data = pd.read_pickle('AShareSEO.pkl')
        data1 = data[data['S_FELLOW_DATE'].notna()]
        data1['S_FELLOW_DATE'] = data1['S_FELLOW_DATE'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['S_FELLOW_DATE'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])
        # discntratio
        factor = pd.pivot_table(data1, index='S_FELLOW_DATE', columns='S_INFO_WINDCODE', values='S_FELLOW_DISCNTRATIO', aggfunc='mean')
        factor = factor.reindex(index=self.date_idx, columns=self.symbol_idx)
        factor_barra = self.event_overflow_3d(factor.astype(float), temp, self.date_idx)
        factor_barra.to_pickle(os.path.join(self.target_dir, 'S_FELLOW_DATE_discntratio.pkl'))

    def calc_S_FELLOW_DATE_ratio(self):
        data = pd.read_pickle('AShareSEO.pkl')
        data1 = data[data['S_FELLOW_DATE'].notna()]
        data1['S_FELLOW_DATE'] = data1['S_FELLOW_DATE'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['S_FELLOW_DATE'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])
        # amount
        factor1 = pd.pivot_table(data1, index='S_FELLOW_DATE', columns='S_INFO_WINDCODE', values='S_FELLOW_AMOUNT', aggfunc='sum')
        factor1 = factor1.reindex(index=self.date_idx, columns=self.symbol_idx)

        # ratio
        free_shares = pd.read_pickle('free_shares_today.pkl').reindex(index=self.date_idx, columns=self.symbol_idx)
        factor2 = factor1 / free_shares
        factor2_barra = self.event_overflow_3d(factor2.astype(float), temp, self.symbol_idx)
        factor2_barra.to_pickle(os.path.join(self.target_dir, 'S_FELLOW_DATE_ratio.pkl'))

    def calc_repurchase_AMT(self):
        data = pd.read_pickle('AShareRepurchase.pkl')
        data1 = data[data['STARTDATE'].notna()]
        data1['STARTDATE'] = data1['STARTDATE'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['STARTDATE'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])

        factor = pd.pivot_table(data1, index='STARTDATE', columns='S_INFO_WINDCODE', values='AMT', aggfunc='mean')
        factor = factor.reindex(index=self.date_idx, columns=self.symbol_idx)
        factor_barra = self.event_overflow_3d(factor.astype(float), temp, self.date_idx)
        factor_barra.to_pickle(os.path.join(self.target_dir, 'repurchase_AMT.pkl'))

    def calc_repurchase_QTY(self):
        data = pd.read_pickle('AShareRepurchase.pkl')
        data1 = data[data['STARTDATE'].notna()]
        data1['STARTDATE'] = data1['STARTDATE'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['STARTDATE'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])

        factor = pd.pivot_table(data1, index='STARTDATE', columns='S_INFO_WINDCODE', values='QTY', aggfunc='mean')
        factor = factor.reindex(index=self.date_idx, columns=self.symbol_idx)
        factor_barra = self.event_overflow_3d(factor.astype(float), temp, self.date_idx)
        factor_barra.to_pickle(os.path.join(self.target_dir, 'repurchase_QTY.pkl'))

    def calc_repurchase_TOTAL_SHARE_RATIO(self):
        data = pd.read_pickle('AShareRepurchase.pkl')
        data1 = data[data['STARTDATE'].notna()]
        data1['STARTDATE'] = data1['STARTDATE'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['STARTDATE'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])

        factor = pd.pivot_table(data1, index='STARTDATE', columns='S_INFO_WINDCODE', values='QTY', aggfunc='mean')
        factor = factor.reindex(index=self.date_idx, columns=self.symbol_idx)
        free_shares = pd.read_pickle('free_shares_today.pkl').reindex(index=self.date_idx, columns=self.symbol_idx)
        factor2 = factor / free_shares
        factor2_barra = self.event_overflow_3d(factor2.astype(float), temp, self.date_idx)
        factor2_barra.to_pickle(os.path.join(self.target_dir, 'repurchase_TOTAL_SHARE_RATIO.pkl'))

    def calc_repurchase_S_DQ_HIGH(self):
        data = pd.read_pickle('AShareRepurchase.pkl')
        data1 = data[data['STARTDATE'].notna()]
        data1['STARTDATE'] = data1['STARTDATE'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['STARTDATE'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])

        factor = pd.pivot_table(data1, index='STARTDATE', columns='S_INFO_WINDCODE', values='S_DQ_HIGH', aggfunc='mean')
        factor = factor.reindex(index=self.date_idx, columns=self.symbol_idx)
        factor_barra = self.event_overflow_3d(factor.astype(float), temp, self.date_idx)
        factor_barra.to_pickle(os.path.join(self.target_dir, 'repurchase_S_DQ_HIGH.pkl'))

    def calc_repurchase_S_DQ_LOW(self):
        data = pd.read_pickle('AShareRepurchase.pkl')
        data1 = data[data['STARTDATE'].notna()]
        data1['STARTDATE'] = data1['STARTDATE'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['STARTDATE'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])

        factor = pd.pivot_table(data1, index='STARTDATE', columns='S_INFO_WINDCODE', values='S_DQ_LOW', aggfunc='mean')
        factor = factor.reindex(index=self.date_idx, columns=self.symbol_idx)
        factor_barra = self.event_overflow_3d(factor.astype(float), temp, self.date_idx)
        factor_barra.to_pickle(os.path.join(self.target_dir, 'repurchase_S_DQ_LOW.pkl'))

    def calc_MjrHolderTrade_buy_HOLDER_QUANTITY_NEW_RATIO(self):
        data = pd.read_pickle('AShareMjrHolderTrade.pkl')
        data1 = data[data['TRANSACT_STARTDATE'].notna()]
        data1['TRANSACT_STARTDATE'] = data1['TRANSACT_STARTDATE'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['TRANSACT_STARTDATE'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]
        data1 = data1[data1['TRANSACT_TYPE'] == 1]

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])

        factor = pd.pivot_table(data1, index='TRANSACT_STARTDATE', columns='S_INFO_WINDCODE', values='TRANSACT_QUANTITY_RATIO', aggfunc='mean')
        factor = factor.reindex(index=self.date_idx, columns=self.symbol_idx)
        factor_barra = self.event_overflow_3d(factor.astype(float), temp, self.date_idx)
        factor_barra.to_pickle(os.path.join(self.target_dir, 'MjrHolderTrade_buy_HOLDER_QUANTITY_NEW_RATIO.pkl'))

    def calc_MjrHolderTrade_sell_HOLDER_QUANTITY_NEW_RATIO(self):
        data = pd.read_pickle('AShareMjrHolderTrade.pkl')
        data1 = data[data['TRANSACT_STARTDATE'].notna()]
        data1['TRANSACT_STARTDATE'] = data1['TRANSACT_STARTDATE'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['TRANSACT_STARTDATE'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]
        data1 = data1[data1['TRANSACT_TYPE'] == 2]

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])

        factor = pd.pivot_table(data1, index='TRANSACT_STARTDATE', columns='S_INFO_WINDCODE', values='TRANSACT_QUANTITY_RATIO', aggfunc='mean')
        factor = factor.reindex(index=self.date_idx, columns=self.symbol_idx)
        factor_barra = self.event_overflow_3d(factor.astype(float), temp, self.date_idx)
        factor_barra.to_pickle(os.path.join(self.target_dir, 'MjrHolderTrade_sell_HOLDER_QUANTITY_NEW_RATIO.pkl'))

    def calc_InsiderTrade_MeanRatio(self):
        data = pd.read_pickle('AShareInsiderTrade.pkl')
        data1 = data[data['TRADE_DT'].notna()]
        data1['TRADE_DT'] = data1['TRADE_DT'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['TRADE_DT'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]
        data1 = data1[data1['CHANGE_VOLUME'] > 0]
        data1['POSITION_AFTER_TRADE'] = data1['POSITION_AFTER_TRADE'].fillna(0)
        data1['ratio'] = data1['CHANGE_VOLUME'] / (data1['POSITION_AFTER_TRADE'] - data1['CHANGE_VOLUME'])

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])

        factor = pd.pivot_table(data1, index='TRADE_DT', columns='S_INFO_WINDCODE', values='ratio', aggfunc='mean')
        factor = factor.reindex(index=self.date_idx, columns=self.symbol_idx)
        factor_barra = self.event_overflow_3d(factor.astype(float), temp, self.date_idx)
        factor_barra.to_pickle(os.path.join(self.target_dir, 'InsiderTrade_MeanRatio.pkl'))

    def calc_InsiderTrade_MaxRatio(self):
        data = pd.read_pickle('AShareInsiderTrade.pkl')
        data1 = data[data['TRADE_DT'].notna()]
        data1['TRADE_DT'] = data1['TRADE_DT'].astype(int)
        data1['ann_date'] = pd.to_datetime(data1['TRADE_DT'].astype(str), format='%Y%m%d')
        data1['opdate_date'] = pd.to_datetime(data1['opdate'].astype(str), format='%Y%m%d')
        data1['date_offset'] = (data1['opdate_date'] - data1['ann_date']).dt.days
        data1 = data1[data1['date_offset'] <= 365]
        data1 = data1[data1['CHANGE_VOLUME'] > 0]
        data1['POSITION_AFTER_TRADE'] = data1['POSITION_AFTER_TRADE'].fillna(0)
        data1['ratio'] = data1['CHANGE_VOLUME'] / (data1['POSITION_AFTER_TRADE'] - data1['CHANGE_VOLUME'])

        temp = self.get_barra_distance(self.date_idx[0], self.date_idx[-1])

        factor = pd.pivot_table(data1, index='TRADE_DT', columns='S_INFO_WINDCODE', values='ratio', aggfunc='max')
        factor = factor.reindex(index=self.date_idx, columns=self.symbol_idx)
        factor_barra = self.event_overflow_3d(factor.astype(float), temp, self.date_idx)
        factor_barra.to_pickle(os.path.join(self.target_dir, 'InsiderTrade_MaxRatio.pkl'))

    # barra因子

    def calc_size(self):
        barra_df = pd.read_pickle('barra_factor.pkl')
        data = pd.pivot(barra_df, index='date', columns='stock_code', values='size').\
            reindex(index=self.date_idx, columns=self.symbol_idx)
        data.to_pickle(os.path.join(self.target_dir, 'size.pkl'))

    def calc_beta(self):
        barra_df = pd.read_pickle('barra_factor.pkl')
        data = pd.pivot(barra_df, index='date', columns='stock_code', values='beta').\
            reindex(index=self.date_idx, columns=self.symbol_idx)
        data.to_pickle(os.path.join(self.target_dir, 'beta.pkl'))

    def calc_momentum(self):
        barra_df = pd.read_pickle('barra_factor.pkl')
        data = pd.pivot(barra_df, index='date', columns='stock_code', values='momentum').\
            reindex(index=self.date_idx, columns=self.symbol_idx)
        data.to_pickle(os.path.join(self.target_dir, 'momentum.pkl'))

    def calc_earnings_yield(self):
        barra_df = pd.read_pickle('barra_factor.pkl')
        data = pd.pivot(barra_df, index='date', columns='stock_code', values='earnings_yield').\
            reindex(index=self.date_idx, columns=self.symbol_idx)
        data.to_pickle(os.path.join(self.target_dir, 'earnings_yield.pkl'))

    def calc_residual_volatility(self):
        barra_df = pd.read_pickle('barra_factor.pkl')
        data = pd.pivot(barra_df, index='date', columns='stock_code', values='residual_volatility').\
            reindex(index=self.date_idx, columns=self.symbol_idx)
        data.to_pickle(os.path.join(self.target_dir, 'residual_volatility.pkl'))

    def calc_growth(self):
        barra_df = pd.read_pickle('barra_factor.pkl')
        data = pd.pivot(barra_df, index='date', columns='stock_code', values='growth').\
            reindex(index=self.date_idx, columns=self.symbol_idx)
        data.to_pickle(os.path.join(self.target_dir, 'growth.pkl'))

    def calc_book_to_price(self):
        barra_df = pd.read_pickle('barra_factor.pkl')
        data = pd.pivot(barra_df, index='date', columns='stock_code', values='book_to_price').\
            reindex(index=self.date_idx, columns=self.symbol_idx)
        data.to_pickle(os.path.join(self.target_dir, 'book_to_price.pkl'))

    def calc_leverage(self):
        barra_df = pd.read_pickle('barra_factor.pkl')
        data = pd.pivot(barra_df, index='date', columns='stock_code', values='leverage').\
            reindex(index=self.date_idx, columns=self.symbol_idx)
        data.to_pickle(os.path.join(self.target_dir, 'leverage.pkl'))

    def calc_liquidity(self):
        barra_df = pd.read_pickle('barra_factor.pkl')
        data = pd.pivot(barra_df, index='date', columns='stock_code', values='liquidity').\
            reindex(index=self.date_idx, columns=self.symbol_idx)
        data.to_pickle(os.path.join(self.target_dir, 'liquidity.pkl'))

    def calc_non_linear_size(self):
        barra_df = pd.read_pickle('barra_factor.pkl')
        data = pd.pivot(barra_df, index='date', columns='stock_code', values='non_linear_size').\
            reindex(index=self.date_idx, columns=self.symbol_idx)
        data.to_pickle(os.path.join(self.target_dir, 'non_linear_size.pkl'))

if __name__ == '__main__':
    factor_list = pd.read_excel('因子池.xlsx')['数据字段'].to_list()
    fc = FactorCalculator('D:/20241/bigdata/factor_data')
    for factor in factor_list:
        method_name = f'calc_{factor}'
        if hasattr(fc, method_name):
            method = getattr(fc, method_name)
            method()  # 调用方法并传入参数
            print(f"Finish factor {factor}")
        else:
            print(f"Method {method_name} not found")
