import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.ib.marketdata import Barsize
from lib.marketsmith import MarketSmith
from lib.y import read_earnings, read_financials
from strategies.pattern_search.dataprep import get_market_data

from lib.marketsmith import read_financials as ms_read_financials
from strategies.pattern_search.features import feature_adr, feature_atr, feature_vwap, feature_MACD


def merge(symbol_, df1, df2):
    index = df1.index
    df2 = df2.loc[df2['symbol'] == symbol_]
    df1 = df1.merge(df2, left_index=True, right_index=True, how='left')
    # Drop column starting with symbol_
    df1 = df1.loc[:, ~df1.columns.str.startswith('symbol_')]
    df1.index = index
    return df1


def get_data(symbol_, exchange_, start_):
    df_ = get_market_data(symbol_, exchange_, Barsize.day_1).copy()

    # Nb of years from today
    now = pd.Timestamp.now()
    years = (now - pd.Timestamp(start_)).days / 365

    # Remove data before 2013
    df_ = df_.loc[df_.index > start_]
    if len(df_) < years * 150:  # At least 150 bars per year
        raise ValueError(f'Not enough data for {symbol_}')

    # print("Read Yahoo earnings...")
    df_ = merge(symbol_, df_, read_earnings())
    # print("Read Yahoo financials...")
    df_ = merge(symbol_, df_, read_financials())
    # print("Read Market Smith financials...")
    df_ = merge(symbol_, df_, ms_read_financials())

    # Create a new column with the previous number of funds
    df_.loc[df_['numberOfFunds'].notnull(), 'numberOfFunds_prev'] = (
        df_.loc[df_['numberOfFunds'].notnull(), 'numberOfFunds'].shift(1))

    # Forward fill numberOfFunds and total_revenue
    df_ = df_.ffill()
    df_ = df_.replace('', np.nan)

    df_ = df_.replace([np.inf, -np.inf], 0)

    # print("Cast to numeric")
    for column in df_.columns:
        df_[column] = pd.to_numeric(df_[column], errors='ignore')

    # Keep columns
    # df_ = df_[['open', 'high', 'low', 'close', 'volume', 'reported_eps', 'surprise_pct',
    #           'total_revenue', 'numberOfFunds', 'marketCapitalizationPrimary', 'numberOfFunds_prev']]

    return df_


def process(symbol_, exchange_, start_='2022-01-01'):
    # Get market data and earnings
    df_ = get_data(symbol_, exchange_, start_)

    # Move the datetime index to a column
    df_ = df_.reset_index()

    # ********************
    # SETUP CONDITIONS
    # ********************
    # 1. Pct gap between current open and previous close
    df_['gap'] = ((df_['open'] - df_['close'].shift(1)) / df_['close'].shift(1) * 100).where(
        df_['open'] > df_['close'].shift(1), np.nan)
    # df_['gap'] = ((df_['open'] - df_['low'].shift(1)) / df_['low'].shift(1) * 100).where(df_['open'] < df_[
    # 'low'].shift(1), df_['gap'])

    # Earnings Acceleration
    # Check if there are any non-null values in the 'reported_eps' column
    if df_['reported_eps'].notnull().any():
        # If there are non-null values, calculate the percentage change
        df_.loc[df_['reported_eps'].notnull(), 'earnings_acceleration'] = df_.loc[df_['reported_eps'].notnull()][
                                                                              'reported_eps'].pct_change() * 100
    else:
        # If there are no non-null values, set 'earnings_acceleration' to NaN
        df_['earnings_acceleration'] = np.nan

    # Sales Acceleration
    # Check if there are any non-null values in the 'reported_eps' column
    if df_['total_revenue'].notnull().any():
        # If there are non-null values, calculate the percentage change
        df_.loc[df_['total_revenue'].notnull(), 'sales_acceleration'] = df_.loc[df_['total_revenue'].notnull()][
                                                                            'total_revenue'].pct_change() * 100
        # Forward fill where the sales acceleration is 0
        df_['sales_acceleration'] = df_['sales_acceleration'].replace(0, np.nan).ffill()
    else:
        # If there are no non-null values, set 'earnings_acceleration' to NaN
        df_['sales_acceleration'] = np.nan

    # Data Cleaning: Set gap to nan if the previous row has a volume of 0
    # df_.loc[df_['volume'].shift(1) == 0, 'gap'] = np.nan
    # df_['volume_close'] = df_['volume'] * df_['close']

    # 2. Volume average
    # df_['volume_avg'] = df_['volume'].rolling(10).mean()

    df_ = feature_adr(df_, 20)
    df_ = feature_atr(df_, 20)
    df_ = feature_vwap(df_)
    df_[f'close_ma_{10}'] = df_['close'].rolling(window=10).mean()
    df_ = feature_MACD(df_)
    df_['close_ma_20'] = df_['close'].rolling(window=20).mean()
    # Avg Volume 10 days
    df_['volume_avg_10'] = df_['volume'].rolling(window=10).mean()
    # *******************
    # Compute performance
    # *******************

    indexer_1 = pd.api.indexers.FixedForwardWindowIndexer(window_size=21 * 1)
    # df_['lowest_1'] = df_['low'].rolling(window=indexer_1, min_periods=1).min()
    # Get the index of the lowest value
    # df_['lowest_1_index'] = df_['low'].rolling(window=indexer_1, min_periods=1).apply(np.argmin, raw=True)

    df_['highest_1'] = df_['high'].rolling(window=indexer_1, min_periods=1).max()
    # Get the index of the highest value
    df_['highest_1_index'] = df_['high'].rolling(window=indexer_1, min_periods=1).apply(np.argmax, raw=True)

    # Find the lowest low between the current day and the lowest low to the highest high index
    df_['lowest_1'] = df_.apply(lambda x: df_.loc[x.name:x.name + x['highest_1_index'], 'low'].min(), axis=1)

    # indexer_3 = pd.api.indexers.FixedForwardWindowIndexer(window_size=21 * 3)
    # df_['lowest_3'] = df_['low'].rolling(window=indexer_3, min_periods=1).min()
    # df_['highest_3'] = df_['high'].rolling(window=indexer_3, min_periods=1).max()
    # indexer_6 = pd.api.indexers.FixedForwardWindowIndexer(window_size=21 * 6)
    # df_['lowest_6'] = df_['low'].rolling(window=indexer_6, min_periods=1).min()
    # df_['highest_6'] = df_['high'].rolling(window=indexer_6, min_periods=1).max()

    # Compute the down pct change from today's close to the lowest
    df_['drawdown_open_1'] = ((df_['lowest_1'] - df_['open']) / df_['open'] * 100)
    # df_['drawdown_open_3'] = ((df_['lowest_3'] - df_['open']) / df_['open'] * 100)
    # df_['drawdown_open_6'] = ((df_['lowest_6'] - df_['open']) / df_['open'] * 100)
    # Compute the down pct change from today's close to the lowest
    df_['drawdown_low_1'] = ((df_['lowest_1'] - df_['low']) / df_['low'] * 100)
    # df_['drawdown_low_3'] = ((df_['lowest_3'] - df_['low']) / df_['low'] * 100)
    # df_['drawdown_low_6'] = ((df_['lowest_6'] - df_['low']) / df_['low'] * 100)

    # Compute the down pct change from today's close to the lowest
    df_['maxrunup_open_1'] = ((df_['highest_1'] - df_['open']) / df_['open'] * 100)

    # df_['maxrunup_open_3'] = ((df_['highest_3'] - df_['open']) / df_['open'] * 100)
    # df_['maxrunup_open_6'] = ((df_['highest_6'] - df_['open']) / df_['open'] * 100)
    # Compute the down pct change from today's close to the lowest
    df_['maxrunup_close_1'] = ((df_['highest_1'] - df_['close']) / df_['close'] * 100)
    # df_['maxrunup_close_3'] = ((df_['highest_3'] - df_['close']) / df_['close'] * 100)
    # df_['maxrunup_close_6'] = ((df_['highest_6'] - df_['close']) / df_['close'] * 100)

    # Compute the pct change in forward_period time
    df_['win1'] = ((df_['close'].shift(-21 * 1) - df_['close']) / df_['close'] * 100)
    # df_['win3'] = ((df_['close'].shift(-21 * 3) - df_['close']) / df_['close'] * 100)
    # df_['win6'] = ((df_['close'].shift(-21 * 6) - df_['close']) / df_['close'] * 100)

    # Set the date as index again
    df_ = df_.set_index('date')

    return df_


def get_fundamentals(symbol_):
    ms = MarketSmith()
    r = ms.get_company_info(symbol_)
    # MA
    last_earnings = r['fundamentalDataBlock']['quarterlyBlock']['content'][-1]['epsPercChange']['value']
    last_sales = r['fundamentalDataBlock']['quarterlyBlock']['content'][-1]['salesPercChange']
    # G: N/A
    # N
    numberOfFunds = r['fundamentalDataBlock']['miscBlock']['numberOfFunds'][-1]['value']
    neglected = numberOfFunds < 100 if numberOfFunds else np.nan
    # A
    previous_sales = r['fundamentalDataBlock']['quarterlyBlock']['content'][-1]['salesPercChange']
    sales_acceleration = last_sales > 100 or (last_sales > 29 and previous_sales > 29) if previous_sales else np.nan

    # 5
    shortInterestRatio = r['companyInfo']['shortInterestRatio']
    m_5 = shortInterestRatio >= 5 if shortInterestRatio else np.nan
    # Cap 10
    marketCapitalizationPrimary = r['companyInfo']['marketCapitalizationPrimary']
    cap10 = marketCapitalizationPrimary < 10000000000 if marketCapitalizationPrimary else np.nan

    # Convert to a pandas series
    s = pd.Series({
        'symbol': symbol_,
        'last_earnings': last_earnings,
        'last_sales': last_sales,
        'previous_sales': previous_sales,
        'numberOfFunds': numberOfFunds,
        'shortInterestRatio': shortInterestRatio,
        'marketCapitalizationPrimary': marketCapitalizationPrimary,
        'N': neglected,
        'A': sales_acceleration,
        '5': m_5,
        'CAP 10': cap10
    })
    return s
