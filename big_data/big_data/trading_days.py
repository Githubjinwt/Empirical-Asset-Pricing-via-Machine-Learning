import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class TradingDay(object):
    def __init__(self):
        self.trading_days = np.load('trading_days.npy')
        self.date_map = dict(list(zip(self.trading_days, np.arange(self.length))))

    @property
    def start_date(self):
        return self.trading_days[0]

    @property
    def end_date(self):
        return self.trading_days[-1]

    @property
    def length(self):
        return len(self.trading_days)

    def distance(self, start_date, end_date):
        return self.date_map[end_date] - self.date_map[start_date]

    def get_loc(self, date_):
        return self.date_map[date_]

    def is_trading_day(self, date_):
        try:
            num = self.date_map[date_]
            return True
        except KeyError:
            return False

    def last_trading_day(self, date_):
        return self.trading_days[self.trading_days < date_][-1]

    def next_trading_day(self, date_):
        return self.trading_days[self.trading_days > date_][0]

    def day_after_last(self, date_):
        last_trading_day = self.last_trading_day(date_)
        datetime_ = datetime.strptime(str(last_trading_day), "%Y%m%d")
        datetime_ += timedelta(days=1)
        return int(datetime_.strftime("%Y%m%d"))

    def trading_day_pair(self, date_):
        if self.is_trading_day(date_):
            return date_, self.next_trading_day(date_)
        else:
            return self.last_trading_day(date_), self.next_trading_day(date_)

    def get_range(self, start_date, end_date):
        return self.trading_days[(self.trading_days >= start_date) &
                                 (self.trading_days <= end_date)]

