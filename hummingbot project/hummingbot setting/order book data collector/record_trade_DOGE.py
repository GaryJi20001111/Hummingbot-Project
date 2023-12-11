import datetime
import os
from decimal import Decimal
from operator import itemgetter

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class MicropricePMM(ScriptStrategyBase):
    # ! Configuration
    trading_pair = "DOGE-USDT"
    exchange = "kucoin_paper_trade"
    range_of_imbalance = 1  # ? Compute imbalance from [best bid/ask, +/- ticksize*range_of_imbalance)

    # ! Microprice configuration
    dt = 1
    n_imb = 6  # ? Needs to be large enough to capture shape of imbalance adjustmnts without being too large to capture noise

    # ! Advanced configuration variables
    show_data = False  # ? Controls whether current df is shown in status
    path_to_data = './data'  # ? Default file format './data/microprice_{trading_pair}_{exchange}_{date}.csv'
    interval_to_write = 60
    price_line_width = 60
    precision = 4  # ? should be the length of the ticksize
    data_size_min = 10000  # ? Seems to be the ideal value to get microprice adjustment values for other spreads
    day_offset = 1  # ? How many days back to start looking for csv files to load data from

    # ! Script variabes
    columns = ['date', 'time', 'ask_vwap20000','bid_vwap20000','ask_vwap10000','bid_vwap10000','ask_vwap5000','bid_vwap5000','bid', 'bs', 'ask', 'as','bid1', 'bs1', 'ask1', 'as1','bid2', 'bs2', 'ask2', 'as2','bid3', 'bs3', 'ask3', 'as3','bid4', 'bs4', 'ask4', 'as4']
    current_dataframe = pd.DataFrame(columns=columns)
    time_to_write = 0
    markets = {exchange: {trading_pair}}
    g_star = None
    recording_data = True
    ticksize = None
    n_spread = None

    # ! System methods
    def on_tick(self):
        # Record data, dump data, update write timestamp
        self.record_data()
        if self.time_to_write < self.current_timestamp:
            self.time_to_write = self.interval_to_write + self.current_timestamp
            self.dump_data()

    # Records a new row to the dataframe every tick
    # Every 'time_to_write' ticks, writes the dataframe to a csv file
    def record_data(self):
        # Fetch bid and ask data
        bid, ask, bid_volume, ask_volume,bid1, ask1, bid_volume1, ask_volume1,bid2, ask2, bid_volume2, ask_volume2,bid3, ask3, bid_volume3, ask_volume3,bid4, ask4, bid_volume4, ask_volume4 = itemgetter('bid', 'ask', 'bs', 'as','bid1', 'ask1', 'bs1', 'as1','bid2', 'ask2', 'bs2', 'as2','bid3', 'ask3', 'bs3', 'as3','bid4', 'ask4', 'bs4', 'as4')(self.get_bid_ask())
        # Fetch date and time in seconds
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        time = self.current_timestamp
        ask_vwap20000 = self.connectors[self.exchange].get_quote_volume_for_base_amount(self.trading_pair, True, 20000)
        bid_vwap20000 = self.connectors[self.exchange].get_quote_volume_for_base_amount(self.trading_pair, False, 20000)
        ask_vwap10000 = self.connectors[self.exchange].get_quote_volume_for_base_amount(self.trading_pair, True, 10000)
        bid_vwap10000 = self.connectors[self.exchange].get_quote_volume_for_base_amount(self.trading_pair, False, 10000)
        ask_vwap5000 = self.connectors[self.exchange].get_quote_volume_for_base_amount(self.trading_pair, True, 5000)
        bid_vwap5000 = self.connectors[self.exchange].get_quote_volume_for_base_amount(self.trading_pair, False, 5000)
        data = [[date, time, ask_vwap20000,bid_vwap20000,ask_vwap10000,bid_vwap10000,ask_vwap5000,bid_vwap5000,bid, bid_volume, ask, ask_volume,bid1, ask1, bid_volume1, ask_volume1,bid2, ask2, bid_volume2, ask_volume2,bid3, ask3, bid_volume3, ask_volume3,bid4, ask4, bid_volume4, ask_volume4]]
        self.current_dataframe = self.current_dataframe.append(pd.DataFrame(data, columns=self.columns), ignore_index=True)
        return

    def dump_data(self):
        if len(self.current_dataframe) < 2 * self.range_of_imbalance:
            return
        # Dump data to csv file
        csv_path = f'{self.path_to_data}/microprice_{self.trading_pair}_{self.exchange}_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv'
        try:
            data = pd.read_csv(csv_path, index_col=[0])
        except Exception as e:
            self.logger().info(e)
            self.logger().info(f'Creating new csv file at {csv_path}')
            data = pd.DataFrame(columns=self.columns)

        data = data.append(self.current_dataframe.iloc[:-self.range_of_imbalance], ignore_index=True)
        data.to_csv(csv_path)
        self.current_dataframe = self.current_dataframe.iloc[-self.range_of_imbalance:]
        return

# ! Data methods
    def get_csv_path(self):
        # Get all files in self.path_to_data directory
        files = os.listdir(self.path_to_data)
        for i in files:
            if i.startswith(f'microprice_{self.trading_pair}_{self.exchange}'):
                len_data = len(pd.read_csv(f'{self.path_to_data}/{i}', index_col=[0]))
                if len_data > self.data_size_min:
                    return f'{self.path_to_data}/{i}'

        # Otherwise just return today's file
        return f'{self.path_to_data}/microprice_{self.trading_pair}_{self.exchange}_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv'

    def get_bid_ask(self):
        bids, asks = self.connectors[self.exchange].get_order_book(self.trading_pair).snapshot
        # if size > 0, return average of range
        best_ask = asks.iloc[0].price
        ask_volume = asks.iloc[0].amount
        best_bid = bids.iloc[0].price
        bid_volume = bids.iloc[0].amount
        #level2
        best_ask1 = asks.iloc[1].price
        ask_volume1 = asks.iloc[1].amount
        best_bid1 = bids.iloc[1].price
        bid_volume1 = bids.iloc[1].amount
        #level3
        best_ask2 = asks.iloc[2].price
        ask_volume2 = asks.iloc[2].amount
        best_bid2 = bids.iloc[2].price
        bid_volume2 = bids.iloc[2].amount
        #level4
        best_ask3 = asks.iloc[3].price
        ask_volume3 = asks.iloc[3].amount
        best_bid3 = bids.iloc[3].price
        bid_volume3 = bids.iloc[3].amount
        #level5
        best_ask4 = asks.iloc[4].price
        ask_volume4 = asks.iloc[4].amount
        best_bid4 = bids.iloc[4].price
        bid_volume4 = bids.iloc[4].amount
        return {'bid': best_bid, 'ask': best_ask, 'bs': bid_volume, 'as': ask_volume,
        'bid1': best_bid1, 'ask1': best_ask1, 'bs1': bid_volume1, 'as1': ask_volume1,
        'bid2': best_bid2, 'ask2': best_ask2, 'bs2': bid_volume2, 'as2': ask_volume2,
        'bid3': best_bid3, 'ask3': best_ask3, 'bs3': bid_volume3, 'as3': ask_volume3,
        'bid4': best_bid4, 'ask4': best_ask4, 'bs4': bid_volume4, 'as4': ask_volume4}


    def get_df(self):
        csv_path = self.get_csv_path()
        try:
            df = pd.read_csv(csv_path, index_col=[0])
            df = df.append(self.current_dataframe)
        except Exception as e:
            self.logger().info(e)
            df = self.current_dataframe

        df['time'] = df['time'].astype(float)
        df['bid'] = df['bid'].astype(float)
        df['ask'] = df['ask'].astype(float)
        df['bs'] = df['bs'].astype(float)
        df['as'] = df['as'].astype(float)

        df['bid1'] = df['bid1'].astype(float)
        df['ask1'] = df['ask1'].astype(float)
        df['bs1'] = df['bs1'].astype(float)
        df['as1'] = df['as1'].astype(float)

        df['bid2'] = df['bid2'].astype(float)
        df['ask2'] = df['ask2'].astype(float)
        df['bs2'] = df['bs2'].astype(float)
        df['as2'] = df['as2'].astype(float)

        df['bid3'] = df['bid3'].astype(float)
        df['ask3'] = df['ask3'].astype(float)
        df['bs3'] = df['bs3'].astype(float)
        df['as3'] = df['as3'].astype(float)

        df['bid4'] = df['bid4'].astype(float)
        df['ask4'] = df['ask4'].astype(float)
        df['bs4'] = df['bs4'].astype(float)
        df['as4'] = df['as4'].astype(float)
        df['mid'] = (df['bid'] + df['ask']) / float(2)
        df['imb'] = df['bs'] / (df['bs'] + df['as'])
        return df

    def compute_imbalance(self) -> Decimal:
        if self.get_df().empty or self.current_dataframe.empty:
            self.logger().info('No data to compute imbalance, recording data')
            self.recording_data = True
            return Decimal(-1)
        bid_size = self.current_dataframe['bs'].sum()
        ask_size = self.current_dataframe['as'].sum()
        return round(Decimal(bid_size) / Decimal(bid_size + ask_size), self.precision * 2)


