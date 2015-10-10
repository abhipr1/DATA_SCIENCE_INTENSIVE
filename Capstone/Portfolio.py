import pandas as pd

class MarketIntradayPortfolio:
    def __init__(self, symbol, bars, signals, initial_capital=1000000.0, shares=500):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.shares = int(shares)
        self.positions = self.generate_positions()

    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index)
        positions[self.symbol] = self.shares*self.signals['signal']
        return positions

    def backtest_portfolio(self):
        portfolio = pd.DataFrame(index=self.positions.index)
        pos_diff = self.positions.diff()
        portfolio['price_diff'] = self.bars['Adj Close']-self.bars['Open']
        #portfolio['price_diff'][0:2] = 0.0
        portfolio['profit'] = self.positions[self.symbol] * portfolio['price_diff']
        portfolio['total'] = self.initial_capital + portfolio['profit'].cumsum()
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio

