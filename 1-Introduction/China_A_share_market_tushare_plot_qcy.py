
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from IPython import display

display.set_matplotlib_formats("svg")

from meta import config
from meta.data_processor import DataProcessor
from main import check_and_make_directories
from meta.data_processors.tushare import Tushare, ReturnPlotter
from meta.env_stock_trading.env_stocktrading_China_A_shares import (
    StockTradingEnv,
)
from agents.stablebaselines3_models import DRLAgent
import os
from typing import List
from argparse import ArgumentParser
from meta import config
from meta.config_tickers import DOW_30_TICKER
from meta.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
)
import pyfolio
from pyfolio import timeseries

pd.options.display.max_columns = None

print("ALL Modules have been imported!")


### Create folders

import os

"""
use check_and_make_directories() to replace the following

if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models")
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log")
if not os.path.exists("./results"):
    os.makedirs("./results")
"""

check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
)


### Download data, cleaning and feature engineering

ticker_list = [
    "600000.SH",
    "600009.SH",
    "600016.SH",
    "600028.SH",
    "600030.SH",
    "600031.SH",
    "600036.SH",
    "600050.SH",
    "600104.SH",
    "600196.SH",
    "600276.SH",
    "600309.SH",
    "600519.SH",
    "600547.SH",
    "600570.SH",
]
# ticker_list = ['600000.XSHG', '600009.XSHG', '600016.XSHG', '600028.XSHG', '600030.XSHG', '600031.XSHG', '600036.XSHG', '600050.XSHG', '600104.XSHG', '600196.XSHG', '600276.XSHG', '600309.XSHG', '600519.XSHG', '600547.XSHG', '600570.XSHG']


TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2019-08-01"
TRADE_START_DATE = "2019-08-01"
TRADE_END_DATE = "2020-01-03"


TIME_INTERVAL = "1d"
import pickle

result_dir_path = rf'/Users/qinchaoyi/workspace/Ultron/code/AI4Fin/FinRL-Tutorials/1-Introduction/results'

### Backtest
with open(rf'{result_dir_path}/result_data.pkl', 'rb') as f:
    result_data = pickle.load(f)

    df_account_value = result_data['df_account_value']
    trade = result_data['trade']


# matplotlib inline
plotter = ReturnPlotter(df_account_value, trade, TRADE_START_DATE, TRADE_END_DATE)
# plotter.plot_all()

plotter.plot()

# matplotlib inline
# # ticket: SSE 50：000016
# plotter.plot("000016")

#### Use pyfolio

# # CSI 300
# baseline_df = plotter.get_baseline("000300.SH")


daily_return = plotter.get_return(df_account_value)
# daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

perf_func = timeseries.perf_stats
perf_stats_all = perf_func(
    returns=daily_return,
    # factor_returns=daily_return_base,
    factor_returns = None,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)
print("==============DRL Strategy Stats===========")
print(f"perf_stats_all: {perf_stats_all}")


daily_return = plotter.get_return(df_account_value)
# daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

# perf_func = timeseries.perf_stats
# perf_stats_all = perf_func(
#     returns=daily_return_base,
#     # factor_returns=daily_return_base,
#     factor_returns=None,
#     positions=None,
#     transactions=None,
#     turnover_denom="AGB",
# )
# print("==============Baseline Strategy Stats===========")

# print(f"perf_stats_all: {perf_stats_all}")
