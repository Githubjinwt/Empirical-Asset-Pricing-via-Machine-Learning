import pandas as pd
import numpy as np
import os
import re
import bottleneck as bn
from tqdm import tqdm

from ysquant.api.stock.daily_sharry_reader import DailySharryReader
import backtrader as bt

import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置显示中文字体
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

# 定义回测策略
class GroupStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or bt.num2date(self.datas[0].datetime[0])
        print('{}, {}'.format(dt.isoformat(), txt))

    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Rejected:
            self.log(f"Rejected : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Margin:
            self.log(f"Margin : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Cancelled:
            self.log(f"Concelled : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Partial:
            self.log(f"Partial : order_ref:{order.ref}  data_name:{order.p.data._name}")

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f" BUY : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}")

            else:
                self.log(f" SELL : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}")

    def notify_trade(self, trade):
        # 一个trade结束的时候输出信息
        if trade.isclosed:
            self.log('closed symbol is : {} , total_profit : {} , net_profit : {}' .format(trade.getdataname(),trade.pnl, trade.pnlcomm))
            # self.trade_list.append([self.datas[0].datetime.date(0),trade.getdataname(),trade.pnl,trade.pnlcomm])

        if trade.isopen:
            self.log('open symbol is : {} , price : {} ' .format(trade.getdataname(),trade.price))

    def __init__(self, group_data):
        """
        初始化策略。
        group_data (pd.DataFrame): 筛选后的分组数据，包含日期和股票代码
        """
        self.group_data = group_data  # 分组数据
        self.current_holdings = set()  # 当前持有的股票
    
    def prenext(self):
        self.next()

    def next(self):
        # 获取当前日期
        current_date = self.datas[0].datetime.date(0).strftime('%Y-%m-%d') # 日频
        # current_date = self.datas[0].datetime.date(0).strftime('%Y-%m') # 月频
        # current_date = pd.to_datetime(current_date).to_period('M')

        # 获取当前日期的股票列表
        if current_date in self.group_data.index:
            current_stocks = self.group_data.loc[current_date]['stock_code'].tolist()
        else:
            current_stocks = []  # 如果当前日期没有数据，则为空
        
        # 平仓：卖出上个月持有但本月不再持有的股票
        for stock_code in list(self.current_holdings):
            if stock_code not in current_stocks:
                data = self.getdatabyname(stock_code)
                self.close(data)  # 平仓
                self.current_holdings.remove(stock_code)

        # 处理需要降低仓位的股票
        if current_stocks:
            weight = 0.95 / len(current_stocks)  # 平均分配资金，0.95是防止小数点溢出
            target_value = self.broker.getvalue() * weight  # 每只股票的目标价值

            # 计算当前持仓的价值
            current_values = {}
            for stock_code in self.current_holdings:
                data = self.getdatabyname(stock_code)
                current_values[stock_code] = data.close[0] * self.getposition(data).size

            # 卖出需要降低仓位的股票
            for stock_code, value in current_values.items():
                if value > target_value:
                    data = self.getdatabyname(stock_code)
                    target_size = int(target_value / data.close[0])
                    self.order_target_size(data, target_size)

            # 买入当前组别的其他股票，position=target_size的时候不会交易
            for stock_code in current_stocks:
                data = self.getdatabyname(stock_code)
                target_size = int(weight * self.broker.getvalue() / data.close[0])
                self.order_target_size(data, target_size)
                self.current_holdings.add(stock_code)

# 读结果数据
dsr = DailySharryReader()
adj_close = dsr.get_field_data('adj_factor') * dsr.get_field_data('close')
adj_close = adj_close.unstack().reset_index(name='close').dropna(subset='close')
adj_close = adj_close.rename(columns={'level_0': 'stock_code', 'level_1': 'date'})
adj_close['date'] = pd.to_datetime(adj_close['date'].astype(str), format='%Y%m%d')
adj_close = adj_close.sort_values(by=['date', 'stock_code'], ignore_index=True)

# 计算年化收益率
def annualized_return(nav):
    s, e = nav.index[0].strftime('%Y-%m-%d'), nav.index[-1].strftime('%Y-%m-%d')
    length = len(adj_close[(adj_close['stock_code'] == '000001') & (adj_close['date'] >= s) & (adj_close['date'] <= e)])
    return pow(nav[-1] / nav[0], 252 / length) - 1

# 回测所有结果
folder = 'results/'
files = os.listdir(folder)
files = [i for i in files if i.endswith('pkl')]
res = pd.DataFrame(columns=['model', '年化收益率', '夏普比率', '最大回撤', '交易次数', '总收益率'])
for filename in tqdm(files):
    model = filename[:-4]
    print('backtest model: ', model)
    
    results_df = pd.read_pickle(os.path.join(folder, filename))
    monthly = results_df.select_dtypes(include=['period[M]']).columns.tolist()
    if len(monthly):
        for col in monthly:
            results_df[col] = results_df[col].dt.to_timestamp()
    results_df['date'] = pd.to_datetime(results_df['date'], format='%Y-%m-%d')
    results_df.sort_values(by=['date', 'stock_code'], ignore_index=True, inplace=True)

    data = pd.merge_asof(results_df, adj_close, on=['date'], by=['stock_code'], allow_exact_matches=True, direction='forward')

    # backtrader不接受pd.Period，所以转化为datetime
    # open high low close都是必须的参数，虽然用不上，但是要有
    # data['date'] = data['date'].dt.to_timestamp()
    data['high'] = data['close']
    data['low'] = data['close']
    data['open'] = data['close']
    data['volume'] = 1000000
    data['openinterest'] = 0

    # 根据预测收益率分组
    results_df['group'] = results_df.groupby('date')['predicted_return'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    
    # 筛选指定组别的数据，做多前10%的股票
    group_num = 9
    group_data = results_df[results_df['group'] == group_num].set_index(['date', 'stock_code']).index
    group_data = group_data.to_frame(index=False).set_index('date')

    # 回测
    cerebro = bt.Cerebro()

    # 添加股票数据
    for stock_code in results_df['stock_code'].unique():
        stock_data = bt.feeds.PandasData(
            dataname=data.loc[data['stock_code'] == stock_code].set_index('date'),
            name=stock_code
        )
        cerebro.adddata(stock_data)

    # 添加策略
    cerebro.addstrategy(GroupStrategy, group_data=group_data)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='my_sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='my_returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='my_drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='my_trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.PyFolio)

    # 设置初始资金
    cerebro.broker.set_cash(100000000)
    cerebro.broker.setcommission(commission=0.001)

    # 运行回测
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
    results_bt = cerebro.run()

    # 统计结果
    sharpe_ratio = results_bt[0].analyzers.my_sharpe.get_analysis()['sharperatio']
    # annual_return = results_bt[0].analyzers.my_returns.get_analysis()['rnorm']
    max_drawdown = results_bt[0].analyzers.my_drawdown.get_analysis()["max"]["drawdown"]/100
    trade_num = results_bt[0].analyzers.my_trade_analyzer.get_analysis()['total']['total']
    returns, positions, transactions, gross_lev = results_bt[0].analyzers.getbyname('pyfolio').get_pf_items()
    annual_return = annualized_return((returns+1).cumprod())

    print(f"夏普率: {sharpe_ratio}, 年化收益率: {annual_return}, 最大回撤: {max_drawdown}, 交易次数: {trade_num}")
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    
    # 保存结果
    returns.index = returns.index.strftime(date_format='%Y-%m-%d')
    transactions.index = transactions.index.strftime(date_format='%Y-%m-%d')
    with pd.ExcelWriter(os.path.join('results', model+'.xlsx'), engine='openpyxl') as writer:
        returns.to_excel(writer, sheet_name='Returns', index=True)
        transactions.to_excel(writer, sheet_name='Transactions', index=True)
    
    # model = re.findall(r'preds_(.*)\.pkl', filename)[0]
    res.loc[len(res)] = [model, annual_return, sharpe_ratio, max_drawdown, trade_num, cerebro.broker.getvalue()/100000000]

    # 绘制回测结果，但是画的好像有问题
    # cerebro.plot(iplot=True)

res.to_csv('results/btResults.csv', index=False)