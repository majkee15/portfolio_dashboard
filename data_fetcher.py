import os
from base import Base
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import yfinance as yf
import outils


class Cursor(Base):

    def __init__(self, db_file):
        super().__init__(__class__.__name__)
        self.db_path = db_file

    def __enter__(self):
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()
        self.cursor.close()
        self.connection.close()
        return


class DataFetcher(Base):

    def __init__(self, datadir: str):
        super().__init__(__class__.__name__)
        self.datadir = datadir
        self.cursor = Cursor(self.datadir)
        conn = sqlite3.connect(self.cursor.db_path)
        self.all_tickers = pd.read_sql("SELECT ticker, id FROM security", conn)
        self.ticker_index = dict(self.all_tickers.to_dict('split')['data'])
        self.tickers = list(self.ticker_index.keys())
        conn.close()

    def update_symbol(self, ticker, ticker_id):
        conn = sqlite3.connect(self.cursor.db_path)
        # Get last date
        sql = f"SELECT price_date FROM daily_price WHERE ticker_id={ticker_id}"
        dates = pd.read_sql(sql, conn)
        last_date = dates.iloc[-1, 0]
        self.logger.info(f"Updating symbol: {ticker.info['symbol']} from date: {last_date}.")
        df = ticker.history(start=last_date)
        for row in df.itertuples():
            sql = "INSERT OR IGNORE INTO daily_price (data_vendor_id, ticker_id, price_date, open_price, high_price," \
                  "low_price, close_price,  volume) VALUES (?, ?, ?, ?, ?, ? ,?, ?)"
            conn.cursor().execute(sql, (outils.YAHOO_VENDOR_ID, ticker_id, row.Index.date(), row.Open, row.High, row.Low,
                                 row.Close, row.Volume))
        conn.close()

    def process_symbol_yf(self, symbol: str):
        try:
            ticker = yf.Ticker(symbol)
            self.logger.info(f'Symbol: {symbol} successfully loaded.')
            df = ticker.history(period="max", auto_adjust = True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            with self.cursor as cursor:
                exchange = ticker.info['exchange']
                currency = ticker.info['currency']
                classType = ticker.info['quoteType']
                exchangeName = ticker.info['fullExchangeName']
                country = ticker.info['region']
                name = ticker.info['longName']
                symbol = ticker.info['symbol']

                # First check if symbol is already in the database
                # sql_check_for_existence = "SELECT ticker FROM daily_price"
                # already_present = cursor.execute("SELECT EXISTS(select 1 FROM security WHERE ticker=?)",
                #                                  (symbol,)).fetchall()[0][0]

                if symbol not in self.tickers:
                    self.logger.info('Downloading new symbol into the database.')
                    # Insert exchange into the table if not already there
                    sql_exchange = "INSERT OR IGNORE INTO exchanges (name, currency, country, exchange) " \
                        "VALUES (?, ?, ?, ?)"
                    cursor.execute(sql_exchange, (exchangeName, currency, country, exchange))

                    # retreive exchange id for ticker security
                    sql_find_exchange_id = "SELECT id FROM exchanges WHERE name=? AND exchange=?"
                    exchange_id = cursor.execute(sql_find_exchange_id, (exchangeName, exchange)).fetchall()[0][0]

                    # save the security to the database - security table
                    sql_security = "INSERT OR IGNORE INTO security(exchange_id, ticker, name, security_class)" \
                                   "VALUES (?, ?, ?, ?)"
                    cursor.execute(sql_security, (exchange_id, symbol, name, classType))
                    sql_find_ticker_id = "SELECT id FROM security WHERE ticker=?"
                    ticker_id = cursor.execute(sql_find_ticker_id, (symbol, )).fetchall()[0][0]

                    #write daily data into the daily prices table
                    for row in df.itertuples():
                        sql = "INSERT OR IGNORE INTO daily_price (data_vendor_id, ticker_id, price_date, open_price, high_price," \
                              "low_price, close_price,  volume) VALUES (?, ?, ?, ?, ?, ? ,?, ?)"
                        cursor.execute(sql, (outils.YAHOO_VENDOR_ID,ticker_id, row.Index.date(), row.Open, row.High, row.Low,
                                             row.Close, row.Volume))
                else:
                    self.logger.info('Symbol already existing, updating instead.')
                    sql_find_ticker_id = "SELECT id FROM security WHERE ticker=?"
                    ticker_id = cursor.execute(sql_find_ticker_id, (symbol,)).fetchall()[0][0]
                    self.update_symbol(ticker, ticker_id)

        except Exception as e:
            self.logger.warning(f'An exception caught during loading symbol: {symbol}. Message: {e}')

    def process_symbols(self, symbols):
        for symbol in symbols:
            self.process_symbol_yf(symbol)

    def fetch_security(self, symbol):
        try:
            ticker_sql = f"SELECT * FROM security WHERE ticker='{symbol}'"
            conn = sqlite3.connect(self.cursor.db_path)
            security = pd.read_sql(ticker_sql, conn)
        except Exception as e:
            self.logger.warning(f'Exception caught while fetching symbol: {symbol}. Message: {e.args[0]}')
            security = None
        finally:
            conn.close()
        return security

    def fetch_price_data_single(self, symbol, date_from=0, date_to=0):
        security = self.fetch_security(symbol)
        ticker_id = security.id.values[0]
        try:
            if (date_from == 0)&(date_to == 0):
                prices_sql = f"SELECT * FROM daily_price WHERE ticker_id='{ticker_id}'"
            elif date_from == 0:
                prices_sql = f"SELECT * FROM daily_price WHERE ticker_id='{ticker_id}' and price_date<='{date_to}'"
            elif date_to == 0:
                prices_sql = f"SELECT * FROM daily_price WHERE ticker_id='{ticker_id}' and price_date>='{date_from}'"
            conn = sqlite3.connect(self.cursor.db_path)
            prices = pd.read_sql(prices_sql, conn)
            prices.index = prices['price_date']
        except Exception as e:
            self.logger.warning(f'Exception caught while fetching symbol: {symbol}. Message: {e.args[0]}')
            prices = None
        finally:
            conn.close()
        return prices

    def fetch_price_data(self, symbols , date_from=0, date_to=0, pricetype="close_price"):
        df = pd.DataFrame(data=None, columns=symbols)
        for symbol in symbols:
            series = self.fetch_price_data_single(symbol, date_from, date_to)
            df[symbol] = series[pricetype]
        return df



if __name__ == '__main__':

    fetcher = DataFetcher('stock_prices_eod.sqlite3')
    # symbols = ['SPY', 'IAU', 'APPLE', 'QLD', 'NNN', 'PIMIX']
    symbols = outils.benchmark + outils.rp_all_weather + outils.american_rocket
    # fetcher.process_symbols(symbols=symbols)
    print(fetcher.fetch_security('SPY'))
    print(fetcher.fetch_price_data(['SPY', 'IAU'], '2019-07-23'))
    # fetcher.process_symbol_yf('SPY')