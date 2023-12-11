import logging
from decimal import Decimal
from typing import List
from collections import deque
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from scipy.integrate import quad
import numpy as np
import pandas as pd
class SimplePMM(ScriptStrategyBase):
    # Optimal Parameter Dictionary
    param_dict = {
        'BTC-USDT': [[28.51, np.inf, np.inf, np.inf], [28.75, 20, np.inf, np.inf], [28.75, 10, 2.89, np.inf], [28.75, 10, 2.89, 20000]],
        'ETH-USDT': [[1.9, np.inf, np.inf, np.inf], [2.0, 20, np.inf, np.inf], [2.1, 20, 2.1, np.inf], [2.1, 20, 2.1, 20000]],
        'DOGE-USDT': [[0.00014, np.inf, np.inf, np.inf], [0.0001, 120, np.inf, np.inf], [0.00008, 120, 0.00017, np.inf], [0.00008, 120, 0.00017, 5000]],
        'CYBER-USDT': [[0.024, np.inf, np.inf, np.inf], [0.02, 120, np.inf, np.inf], [0.023, 120, 0.028, np.inf], [0.023, 120, 0.028, 500]],
    }

    order_amount_dict = {
        'BTC-USDT':  0.023,
        'ETH-USDT': 0.42,
        'DOGE-USDT': 10000,
        'CYBER-USDT': 145
    }

    algo = 3
    trading_pair = "BTC-USDT"
    order_amount = order_amount_dict[trading_pair]
    x, l, k, alpha = param_dict[trading_pair][algo]

    order_refresh_time = 60
    create_timestamp = 0

    token = trading_pair.split('-')[0]
    base = trading_pair.split('-')[1]
    exchange = "kucoin_paper_trade"
    price_source = PriceType.MidPrice
    markets = {"kucoin_paper_trade": {trading_pair}}

    price = []
    vol_fresh = 60 if (l==np.inf) else 60*l
    vol = 1
    cur_vol = 1
    q = 0

    def on_tick(self):
        self.exchange = "kucoin_paper_trade"
        ref_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
        self.price.append(float(ref_price))
        if self.create_timestamp <= self.current_timestamp and len(self.price)>60:
            self.cancel_all_orders()
            self.cur_vol = float(np.std(pd.Series(self.price[-60:]).pct_change().dropna())) * np.sqrt(365 * 24 * 60 * 60)
            if len(self.price)>=self.vol_fresh:
                self.vol = float(np.std(pd.Series(self.price[(-self.vol_fresh):]).pct_change().dropna()))*np.sqrt(365*24*60*60)
            else:
                self.vol = self.cur_vol
            proposal: List[OrderCandidate] = self.create_proposal(ref_price)
            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)
            self.create_timestamp = self.order_refresh_time + self.current_timestamp

    def create_proposal(self,ref_price) -> List[OrderCandidate]:


        if self.algo == 3:
            askvamp = self.connectors[self.exchange].get_vwap_for_volume(self.trading_pair, True, Decimal(self.alpha / float(ref_price))).result_price
            bidvamp = self.connectors[self.exchange].get_vwap_for_volume(self.trading_pair, False, Decimal(self.alpha / float(ref_price))).result_price
            fair_price = float((askvamp+bidvamp) / 2)
        else:
            fair_price = float(ref_price)


        volatility_adjustment = self.cur_vol / self.vol if (self.vol != 0) else 1 # For algo 0, always equal to 1
        buy_price = fair_price - self.x * volatility_adjustment
        sell_price = fair_price + self.x * volatility_adjustment

        if (self.algo == 2) or (self.algo == 3):
            balance_df = self.get_balance_df()
            quote_amount = balance_df[balance_df['Asset'] == self.token]['Total Balance'].values[0]
            base_amount = balance_df[balance_df['Asset'] == self.base]['Total Balance'].values[0]
            self.q = quote_amount * float(ref_price) / base_amount - 1
            if self.q >= 0:
                buy_price -= self.k * self.q * volatility_adjustment
            else:
                sell_price -= self.k * self.q * volatility_adjustment

        buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                   order_side=TradeType.BUY, amount=Decimal(self.order_amount), price=Decimal(buy_price))

        sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.SELL, amount=Decimal(self.order_amount), price=Decimal(sell_price))

        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)

        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = (f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)