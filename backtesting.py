from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from collections import OrderedDict
from trading_bot import AlpacaClient, KronosForecaster
from model import Kronos, KronosTokenizer, KronosPredictor
from dotenv import load_dotenv
import json
import os
import sys
import time
from datetime import datetime

env_file = './alpaca.env'
load_dotenv(env_file)
config_file = './configs/test.json'
with open(config_file) as f:
    CONFIG = json.load(f)

# weighted average
# downside deviation
# relative strength index
# average directional index
# VAR
def rss_to_se(rss:float, n:int):
  rmse = rss/(n-2)
  xs = list(range(n))
  ssx = sum([(x - np.mean(xs))**2 for x in xs])
  se = np.sqrt(rmse/ssx)
  return se

def calc_slope(arr, n_future):
  xs = list(range(n_future))
  slope, rss, _, _, _ = np.polyfit(xs, arr, 1, full = True)
  se = rss_to_se(rss, n_future)
  return slope[0], se

def plot_trends(xs, ys):
    fig = go.Figure()
    y_obs = [(x* obs_moments[0]/100) +ys[0] for x in xs]
    y_sdu = [x*(obs_moments[0]+obs_moments[1])/100 +ys[0] for x in xs]
    y_sdl = [x*(obs_moments[0]-obs_moments[1])/100 +ys[0] for x in xs]

    y_bar = [(x* pred_moments[0][0]/100) +ys[0] for x in xs]
    y_upper = [x*(pred_moments[0][0]+pred_moments[0][1])/100 +ys[0] for x in xs]
    y_lower = [x*(pred_moments[0][0]-pred_moments[0][1])/100 +ys[0] for x in xs]

    fig.add_trace(go.Scatter(x=xs, y=ys, name='actual'))
    fig.add_trace(go.Scatter(x=xs, y = y_obs, mode = 'lines', name = 'estimated'))
    fig.add_trace(go.Scatter(x=xs, y = y_sdu, mode = 'lines', name = 'upper'))
    fig.add_trace(go.Scatter(x=xs, y = y_sdl, mode = 'lines', name = 'lower', fill = 'tonexty'))
    fig.add_trace(go.Scatter(x=xs, y = y_bar, mode = 'lines', name = 'predicted'))
    fig.add_trace(go.Scatter(x=xs, y = y_upper, mode = 'lines', name = 'upper'))
    fig.add_trace(go.Scatter(x=xs, y = y_lower, mode = 'lines', name = 'lower', fill = 'tonexty'))
    fig.show()

class BackTester():
    def __init__(self, cfg):
        load_dotenv(cfg['env'])
        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        paper      = os.getenv("PAPER_TRADING", "true").lower() != "false"

        if not api_key or not secret_key:
            sys.exit(
                "\n❌  No API keys found.\n"
                "    Create a .env file with:\n"
                "        ALPACA_API_KEY=your_key\n"
                "        ALPACA_SECRET_KEY=your_secret\n"
                "        PAPER_TRADING=true\n"
            )

        self.alpaca_client = AlpacaClient(api_key, secret_key, paper=paper)
        self.predictor = KronosForecaster(cfg).predictor
        self.symbols = cfg['symbols']
        self.context = cfg['context_bars']
        self.max_context = cfg['max_context']
        self.horizon = cfg['forecast_steps']
        self.n_samples = cfg['n_samples']
        self.T = cfg['T']
        self.take_limit = cfg['take_profit_pct']
        self.stop_loss = cfg['stop_loss_pct']
        self.buy_threshold = cfg['buy_threshold']

    @staticmethod
    def get_bars(alpaca_client, symbol, max_context) -> pd.DataFrame:
        """Fetch recent daily OHLCV bars via Alpaca (free IEX feed).""" 
        df = alpaca_client.get_bars(symbol, max_context, adjustment = 'all')
        return df
    
    @staticmethod
    def make_y_ts(start, horizon, freq):
        y_ts = pd.Series(
            pd.bdate_range(
                start = start,
                periods = horizon,
                # freq    = f"{freq}D",
                freq = f"{freq}B" # b for business day
                )
            )
        return y_ts
    
    @staticmethod
    def make_backtest_data(bars, horizon, context):
        """create lists of ohlcv, x_timestamps, and y_timstamps
        compatible with KronosPredictor.batch_predict

        Parameters
        ---
        bars: pd.DataFrame
            ohlcv data returned by get_bars. must have datetimeindex
        horizon: int
            number of trading days to forecast
        context: int
            number of previous trading days to use as history
        """
        bars.sort_index(axis = 0, ascending = True, inplace = True)
        indices = list(range(len(bars) - horizon - context + 1))
        
        ohlc = ["open", "high", "low", "close"]
        ctx_df = bars[ohlc].copy(deep = True)
        if "volume" in bars.columns:
            ctx_df["volume"] = bars["volume"].values
            ctx_df["amount"] = (bars["volume"] * bars["close"]).values
        
        cfx_dfs = [ctx_df.iloc[i:i+context].copy() for i in indices]
        x_ts = [bars['timestamp'].iloc[i:i+context] for i in indices]
        freq = len(pd.bdate_range(ctx_df.index[-2], ctx_df.index[-1])) - 1
        y_ts = [BackTester.make_y_ts(bars['timestamp'].iloc[i+context], horizon, freq) for i in indices]
        # TODO: create assert statement to ensure data is as expected
        return OrderedDict({'cfx':cfx_dfs, 'xs':x_ts, 'ys': y_ts})

    @staticmethod
    def make_backtest_preds(
        predictor:KronosPredictor,
        dat:OrderedDict,
        horizon: int,
        T:float,
        n_samples:int):
        # create predictions
        pred_df_list = predictor.predict_batch(
            df_list = dat['cfx'],
            x_timestamp_list= dat['xs'],
            y_timestamp_list = dat['ys'],
            pred_len = horizon,
            T=T,
            top_k=0,
            top_p=0.9,
            sample_count=n_samples,
            verbose=True)   

        return pred_df_list

    @staticmethod
    def summarize_preds(pred_df):
        # organize predictions
        # slope, se = calc_slope(pred_df['close'].values, len(pred_df))
        summary_df = pd.DataFrame.from_dict(
            {
            "timestamp": pred_df.index[0:1],
            "low" : pred_df['low'].min(),
            "high" : pred_df['high'].max(),
            "close" : pred_df['close'].iloc[0],
            "expected": pred_df['close'].mean(),
            "var": pred_df['close'].var()
            },
            orient = 'columns')
        summary_df.set_index('timestamp', inplace = True)
        
        return summary_df
    
    @staticmethod
    def aggregate_preds(pred_df_list):
        df = pd.concat(
            [BackTester.summarize_preds(pred_df) for pred_df in pred_df_list],
            axis = 0,
            ignore_index = False)
        return df
        
    @staticmethod
    def plot_predictions(true_df:pd.DataFrame, pred_df:pd.DataFrame, metric:str='close') -> None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
        ax.plot(true_df[metric], label='Ground Truth', color='blue', linewidth=1.5)
        ax.plot(pred_df[metric], label='Prediction', color='red', linewidth=1.5)
        ax.set_ylabel(f'{metric.capitalize()} Price', fontsize=14)
        ax.legend(loc='lower left', fontsize=12)
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def combine(preds:pd.DataFrame, bars:pd.Dataframe):
        """Combine predicted and observed OHLCV data for a ticker
        Args:
            preds (pd.DataFrame): predicted OHLCV data output of aggregate_preds
            bars (pd.DataFrame): observed OHLCV data output from get_bars
        
        Returns:
            pd.DataFrame: merged predicted and observed data
        """
        combined = pd.merge_asof(
            left = bars,
            left_index = True,
            right =  preds,
            right_index = True,
            suffixes = [None, '_p'],
            tolerance = pd.Timedelta(12, unit='h'),
            direction = 'backward')
        
        # for calculating percent change its useful to have the
        # real previous day close in a row
        combined['shifted_close'] = combined['close'].shift()   

        return combined
    
    # def close_position(ledger, type, )

    @staticmethod
    def backtest_single(combined: pd.DataFrame, funds, buy_threshold, take_limit, stop_loss):
        fresh_row = {
            'entry_date':None,
            'position':None, # 'long' or 'short'
            'entry_price':None, # transaction price
            'exit_date':None, # 'open' or 'close'
            'exit_price':None,
            'days':0
        }

        result = pd.DataFrame(fresh_row, index = [0])

        for i, row in combined.iterrows():
            last_row = result.index[-1]
            position = result.iloc[last_row]
            if position['position'] is None: # if we're not holding
                expected = row['expected']
                var = row['var']
                prev = row['shifted_o']
                open = row['open']
                expected_p_min = ((expected - var) - prev)/prev
                expected_p_max = ((expected + var) - prev)/prev

                if expected_p_min >= buy_threshold:
                    print(f"buying at {open}")
                    result.at[last_row, 'position']='long'
                    result.at[last_row, 'entry_price']=open
                    result.at[last_row, 'days']=1
                    result.at[last_row, 'entry_date']=row['timestamp']
                elif expected_p_max <= -buy_threshold:
                    print(f"shorting at {open}")
                    result.at[last_row,'position']='short'
                    result.at[last_row,'entry_price']=open
                    result.at[last_row,'days']=1
                    result.at[last_row,'entry_date']=row['timestamp']

            elif position['position'] == 'short': # if we have a short position
            # see if they trigger take limits
                anchor = position['entry_price']
                high_pct = (row['high'] - anchor)/anchor
                low_pct = (row['low'] - anchor)/anchor
                prev_close_pct = (row['shifted_o'] - anchor)/anchor
                
                if position['days']==5: # if we've held for 5 days
                    print(f'closing short position for {-prev_close_pct} return')
                    funds += (funds*-prev_close_pct)
                    result.at[last_row,'exit_date'] = row['timestamp']
                    result.at[last_row,'exit_price'] = row['shifted_o']
                    result.at[last_row,'change'] = -prev_close_pct
                    result.at[last_row,'funds'] = funds
                    result.loc[len(result)] = fresh_row
                
                elif -low_pct >= take_limit: # if we hit our take limit
                    print(f'closing long position for {-low_pct} return')
                    funds += (funds*-low_pct)
                    result.at[last_row,'exit_date'] = row['timestamp']
                    result.at[last_row,'exit_price'] = row['low']
                    result.at[last_row,'change'] = -low_pct
                    result.at[last_row,'funds'] = funds
                    result.loc[len(result)] = fresh_row
                    
                elif high_pct >= stop_loss:
                    print(f'closing long position for {-high_pct} return')
                    funds += (funds*-high_pct)
                    result.at[last_row,'exit_date'] = row['timestamp']
                    result.at[last_row,'exit_price'] = row['high']
                    result.at[last_row,'change'] = -high_pct
                    result.at[last_row,'funds'] = funds
                    result.loc[len(result)] = fresh_row                      
                
                else:
                    result.at[last_row,'days'] += 1

            elif position['position'] == 'long': # if we have a long position
            # see if they trigger take limits
                anchor = position['entry_price']
                high_pct = (row['high'] - anchor)/anchor
                low_pct = (row['low'] - anchor)/anchor
                prev_close_pct = (row['shifted_o'] - anchor)/anchor
                
                if position['days']==5: # if we've held for 5 days
                    print(f'closing long position for {prev_close_pct} return')
                    funds += (funds*prev_close_pct)
                    result['exit_date'] = row['timestamp']
                    result['exit_price'] = row['shifted_o']
                    result['change'] = prev_close_pct
                    result['funds'] = funds
                    result.loc[len(result)] = fresh_row
                
                elif high_pct >= take_limit: # if we hit our take limit
                    print(f'closing long position for {high_pct} return')
                    funds += (funds*high_pct)
                    result.at[last_row,'exit_date'] = row['timestamp']
                    result.at[last_row,'exit_price'] = row['high']
                    result.at[last_row,'change'] = high_pct
                    result.at[last_row,'funds'] = funds
                    result.loc[len(result)] = fresh_row
                    
                elif low_pct <= -stop_loss:
                    print(f'closing long position for {low_pct} return')
                    funds += (funds*low_pct)
                    result.at[last_row,'exit_date'] = row['timestamp']
                    result.at[last_row,'exit_price'] = row['low']
                    result.at[last_row,'change'] = low_pct
                    result.at[last_row,'funds'] = funds
                    result.loc[len(result)] = fresh_row                      
                
                else:
                    result.at[last_row,'days'] += 1
            
        return result
    
    @staticmethod
    def backtest_multiple(preds: xr.DataArray, funds, n_hold, buy_threshold, short_threshold, take_limit, stop_loss):
        # get all the tickers we're monitoring
        tickers = preds['ticker'].values
        portfolio = {
            ticker:{
                'funds':None,
                'days':None,
                'entry_price':None,
                'entry_date':None,
                'position':None,
                'exit_price':None,
                'exit_date':None} for ticker in tickers}
        ledger = []
        
        # for each new 'day' in backtest period...
        for i, t in enumerate(preds['timestamp'].values):
            group = preds.isel(timestamp = i)
            df = group['preds'].to_pandas().transpose()
            df['exp_p_min'] = ((df["expected"] - df["var"]) - df["shifted_close"])/df["shifted_close"]
            df['exp_p_max'] = ((df["expected"] + df["var"]) - df["shifted_close"])/df["shifted_close"]
            df['exp_p'] = abs(df[['exp_p_min', 'exp_p_max']]).min(axis = 1)
            new_longs = df['exp_p_min'] >= buy_threshold 
            new_shorts = df['exp_p_max'] <= short_threshold # boolean vectors
            df['thresholds'] = new_longs + new_shorts
            df.sort_values(['thresholds', 'exp_p'], ascending = False, inplace = True)
            df['row_number'] = range(len(df))

            # get reference data needed to calculate exit prices
            highs = df['high']
            lows = df['low']
            opens = df['open']
            prev_closes = df['shifted_close']
            tickers = df.index.values
            
            # first, check which positions should be closed
            day_five = [ticker for ticker, value in portfolio.items() if value['days'] == 5]
            exits = prev_closes[df.index.isin(day_five)] # positions to close b/c we held for 5 days
            anchors = [portfolio[ticker]['entry_price'] for ticker in exits.index]
            exit_pct = (exits - anchors)/anchors
            profits = (1+exit_pct)*[portfolio[ticker]['funds'] for ticker in exits.index]
            
            new_row = pd.DataFrame({
                'funds': [portfolio[ticker]['funds'] for ticker in exits.index],
                'entry_date': [portfolio[ticker]['entry_date'] for ticker in exits.index],
                'entry_price': [portfolio[ticker]['entry_price'] for ticker in exits.index],
                'position': [portfolio[ticker]['position'] for ticker in exits.index],
                'exit_date':t,
                'exit_price':exits,
                'profit':profits})
            if not new_row.empty:
                ledger.append(new_row)
                print(f'closing {len(new_row)} positions held 5 days for {sum(profits)}') 
                funds += np.nansum(profits)
            # update portfolio to reflect these closings
            portfolio |= {ticker:{'funds': None, 'days':None, 'entry_date': None, 'entry_price':None, 'position':None, 'exit_date':None, 'exit_price':None} for ticker in day_five}
            del(exits, profits, day_five, new_row)

            # next, check remaining positions for stop-loss and take-limit
            shorts = [ticker for ticker, value in portfolio.items() if value['position'] == 'short'] # if we have a short position
            short_highs = highs[df.index.isin(shorts)]
            short_lows = lows[df.index.isin(shorts)]
            short_opens = opens[df.index.isin(shorts)]
            anchors = [portfolio[ticker]['entry_price'] for ticker in short_highs.index]
            assert(len(anchors) == len(short_lows) == len(short_highs))
            # see if high/low prices trigger exits
            open_pct = -(short_opens - anchors)/anchors
            low_pct = -(short_lows - anchors)/anchors
            high_pct = -(short_highs - anchors)/anchors
            
            # if the day opened higher than stop loss, sale would be triggered there
            open_losses = (open_pct < stop_loss)*short_opens
            # if we open above our take limit, sale triggered at open price
            open_takes = (open_pct > take_limit)*short_opens 
            # anything triggered at open supercedes subsequent action
            open_exits = open_losses + open_takes # these are mutually exclusive

            # subsequent highs that trigger would be sold at stop loss
            losses = (high_pct < stop_loss) * [(1-stop_loss) * anchor for anchor in anchors]
            # only update those that weren't already sold at open
            losses[open_exits > 0.0] = open_exits[open_exits>0.0]
            
            
            # if low is below open and triggers take limit, sale would be triggered at take limit
            takes = (low_pct > take_limit)*[(1-take_limit) * anchor for anchor in anchors]
            # to be conservative, assume any stop losses got triggered first. only update others
            takes[losses > 0.0] = losses[losses>0.0]
            # takes = np.array([open_takes, low_takes])
            # take_limits = np.min(takes, axis = 0, where = takes>0, initial = np.inf)
            # if we have both stop loss and take, conservatively assume stop loss
            exits = takes

            # take_limits = loss_pct >= take_limit # boolean array of which tickers to close due to stop loss
            # stop_losses = take_pct <= stop_loss # boolean array of which tickers to close due to take limit
            # exits = (short_lows*take_limits) + (short_highs*stop_losses)
            profits = (1+(-(exits - anchors)/anchors)) * [portfolio[ticker]['funds'] for ticker in exits.index]
            # increase all by one day
            [portfolio[ticker].update({'days': portfolio[ticker]['days'] + 1}) for ticker in shorts]
            # record exit price for those exiting 
            new_row = pd.DataFrame({
                'funds': [portfolio[ticker]['funds'] for ticker in exits[exits>0].index],
                'entry_date': [portfolio[ticker]['entry_date'] for ticker in exits[exits>0].index],
                'entry_price': [portfolio[ticker]['entry_price'] for ticker in exits[exits>0].index],
                'position': 'short',
                'exit_date':t,
                'exit_price':exits[exits>0],
                'profit':profits[exits>0]})
            if not new_row.empty:
                ledger.append(new_row)  
                print(f'closing {len(new_row)} short positions for {np.nansum(profits[exits>0])}')          
                funds += np.nansum(profits[exits>0])
            # reset portfolio for closed positions
            portfolio |= {ticker:{'funds': None, 'days':None, 'entry_date': None, 'entry_price':None, 'position':None, 'exit_date':None, 'exit_price':None} for ticker in exits[exits>0].index}
            del(profits, new_row, exits, takes, losses, open_exits)

            longs = [ticker for ticker, value in portfolio.items() if value['position'] == 'long'] # if we have a short position
            long_highs = highs[df.index.isin(longs)]
            long_lows = lows[df.index.isin(longs)]
            long_opens = opens[df.index.isin(longs)]
            anchors = [portfolio[ticker]['entry_price'] for ticker in long_highs.index]
            assert(len(anchors) == len(long_lows) == len(long_highs))
            # see if high/low prices trigger exits
            low_pct = (long_lows - anchors)/anchors
            high_pct = (long_highs - anchors)/anchors
            open_pct = (long_opens - anchors)/anchors

            # if the day opened lower than stop loss, sale triggered at open price
            open_losses = (open_pct < stop_loss)*long_opens
            # if the day opened above take limit, sale triggered at open price
            open_takes = (open_pct > take_limit)*long_opens 
            # anything triggered at open supercedes subsequent action
            open_exits = open_losses + open_takes # these are mutually exclusive

            # subsequent lows that trigger would be sold at stop loss
            losses = (low_pct < stop_loss) * [(1+stop_loss) * anchor for anchor in anchors]
            # only update those that weren't already sold at open
            losses[open_exits > 0.0] = open_exits[open_exits>0.0]
            
            
            # if high is above open and triggers take limit, sale triggered at take limit
            takes = (high_pct > take_limit)*[(1+take_limit) * anchor for anchor in anchors]
            # to be conservative, assume any stop losses got triggered first. only update others
            takes[losses > 0.0] = losses[losses>0.0]
            # takes = np.array([open_takes, low_takes])
            # take_limits = np.min(takes, axis = 0, where = takes>0, initial = np.inf)
            # if we have both stop loss and take, conservatively assume stop loss
            exits = takes

            # take_limits = loss_pct >= take_limit # boolean array of which tickers to close due to stop loss
            # stop_losses = take_pct <= stop_loss # boolean array of which tickers to close due to take limit
            # exits = (short_lows*take_limits) + (short_highs*stop_losses)
            profits = (1+(exits - anchors)/anchors) * [portfolio[ticker]['funds'] for ticker in exits.index]
            # increase all by one day
            new_row = pd.DataFrame({
                'funds': [portfolio[ticker]['funds'] for ticker in exits[exits>0].index],
                'entry_date': [portfolio[ticker]['entry_date'] for ticker in exits[exits>0].index],
                'entry_price': [portfolio[ticker]['entry_price'] for ticker in exits[exits>0].index],
                'position': 'long',
                'exit_date':t,
                'exit_price':exits[exits>0],
                'profit':profits[exits>0]})
            if not new_row.empty:
                ledger.append(new_row)
                print(f'closing {len(new_row)} long positions for {np.nansum(profits[exits>0])}')
                # update funds
                funds += np.nansum(profits[exits>0])
            # reset portfolio for closed positions
            portfolio |= {ticker:{'funds': None, 'days':None, 'entry_date': None, 'entry_price':None, 'position':None, 'exit_date':None, 'exit_price':None} for ticker in exits[exits>0].index}
            del(profits, new_row, exits, takes, losses, open_exits)

            # Now check for new positions to open
            # get our remaining holdings
            print(funds)
            if funds > 0:
                holding = [ticker for ticker, values in portfolio.items() if values['position'] is not None]
                n_target = n_hold - len(holding) # number of positions to fill
                new_longs = ((df['exp_p_min'] >= buy_threshold) & ~(df.index.isin(holding)) & (df['row_number'] < n_target)) * df['thresholds']
                new_shorts = ((df['exp_p_max'] <= short_threshold) & ~(df.index.isin(holding)) & (df['row_number'] < n_target)) * df['thresholds']           
                long_entries = opens[new_longs] # array of open prices
                short_entries = opens[new_shorts] # array of closing prices
                n_add = (len(long_entries) + len(short_entries))
                assert(n_add <= n_target)
                # update portfolio with long openings
                portfolio |= {ticker: {'funds': funds/n_add, 'days':1, 'entry_date': t, 'entry_price': long_entries.iloc[i], 'position':'long', 'exit_date':None, 'exit_price':None} for i, ticker in enumerate(tickers[new_longs])}
                # update portfolio with short openings
                portfolio |= {ticker: {'funds': funds/n_add, 'days':1, 'entry_date': t, 'entry_price': short_entries.iloc[i], 'position':'short', 'exit_date':None, 'exit_price':None} for i, ticker in enumerate(tickers[new_shorts])}
                del(df)
                if n_add > 0:
                    funds = 0
            else:
                print('no liquidity, moving to next day')
                next

        return pd.concat(ledger, axis = 0)
    
    def run_single_backtest(self, symbol, funds):
        today = datetime.strftime(pd.Timestamp.today(), '%d%B%Y')
        bars = self.get_bars(self.alpaca_client, symbol, self.max_context)
        backtest_data = self.make_backtest_data(bars, self.horizon, self.context)
        pred_df_list = self.make_backtest_preds(
            predictor = self.predictor,
            dat = backtest_data,
            horizon = self.horizon,
            T = self.T,
            n_samples = self.n_samples)
        preds = self.aggregate_preds(pred_df_list)
        combined = self.combine(preds = preds, bars = bars)
        combined.to_csv(f'./data/backtests/{symbol}_{today}.csv')
        result = self.backtest_single(combined = combined, funds = funds, buy_threshold = self.buy_threshold, take_limit = self.take_limit, stop_loss = self.stop_loss)
        return result

# GATHER OHLCV DATA FOR A BUNCH OF TICKERS
backtester = BackTester(CONFIG)
nasdaq = pd.read_csv('./data/NASDAQ.csv')
nas_tickers = list(nasdaq['Symbol'].values)
nas_tickers = [t for t in nas_tickers if type(t) == str]
nas_tickers = [t for t in nas_tickers if len(t) <= 4]
import random
subset = random.sample(nas_tickers, k = 500)

ohlcv = ['open', 'high', 'low', 'close', 'volume']
for i, t in enumerate(subset):
    print(f'getting bars for {t}')
    df = BackTester.get_bars(backtester.alpaca_client, symbol = t, max_context = 360)
    # make sure there's enough data for backtesting
    if len(df) < 360:
        print(f'skipping {t} with {len(df)} rows')
        next
    else:
        da = xr.DataArray(
            data = df[ohlcv],
            dims = ['timestamp', 'metrics'],
            coords = {'timestamp':df['timestamp'], 'metrics':ohlcv},
            name = 'ohlcv'
        )
        
        print(f'writing data to zarr')
        if os.path.exists('./nasdaq'):
            z = xr.open_zarr('./nasdaq')
            aligned, z_aligned = xr.align(da, z, join = 'outer', fill_value = np.nan )
            # if time dimensions don't match
            if z_aligned['timestamp'].shape > z['timestamp'].shape:
                diff = list(set(z_aligned['timestamp'].values) ^ set(z['timestamp'].values))
                z.close()
                z_aligned.sel(timestamp=slice(diff[0], diff[-1]+1)).to_zarr('./nasdaq', append_dim = 'timestamp', align_chunks = True)
            z.close()
            aligned.expand_dims({'ticker':[t]}, axis = -1).chunk({'ticker':1}).to_zarr('./nasdaq', append_dim = 'ticker', align_chunks = True)

        else:
            da.expand_dims({'ticker':[t]}, axis = -1).chunk({'ticker':1}).to_zarr(store = './nasdaq')
    time.sleep(2)


def make_historic_preds(zs:str, backtester:BackTester):
    """generate daily predictions from historic ohlcv data
    Parameters
    ---
    z: str
        path to zarr store containing historic ohlcv ticker data
    backtester: BackTester
        backtesting object providing predictive model and strategy parameters
    """
    today = datetime.strftime(pd.Timestamp.today(), '%d%B%Y')

    z = xr.open_zarr(zs)

    for ticker in z['ticker'].values:
        print(ticker)
        df = z['ohlcv'].sel(ticker = ticker).to_pandas().dropna(how = 'any', axis = 0)
        df['timestamp'] = df.index.values
        backtest_data = backtester.make_backtest_data(df, horizon = 5, context = 120)
        pred_df_list = backtester.make_backtest_preds(
            predictor = backtester.predictor,
            dat = backtest_data,
            horizon = backtester.horizon,
            T = backtester.T,
            n_samples = backtester.n_samples)
        del(backtest_data)
        preds = backtester.aggregate_preds(pred_df_list)
        combined = backtester.combine(preds = preds, bars = df)
        del(preds)
        da = xr.DataArray(
            data = combined[['open', 'high', 'low', 'low_p','high_p','close_p','expected','var','shifted_close']],
            dims = ['timestamp', 'metrics'],
            coords = {'timestamp':df['timestamp'], 'metrics':['open', 'high', 'low', 'low_p','high_p','close_p','expected','var','shifted_close']},
            name = 'preds'
        )
        del(combined)
        if os.path.exists('./data/nasdaq_preds'):
            
        #     aligned, _ = xr.align(da, z, join = 'outer', fill_value = np.nan )
        #     aligned.sel({'metrics':['low_p','high_p','close_p','expected','var','shifted_o']})
        #     .assign_coords({'metrics':['low_p', 'high_p', 'cl_p', 'exp', 'var', 'shift']})
            
            da.expand_dims({'ticker':[ticker]}, axis = -1).chunk({'ticker':1}).to_zarr('./data/nasdaq_preds', append_dim = 'ticker', align_chunks = True)
            
        else:
            da.expand_dims({'ticker':[ticker]}, axis = -1).chunk({'ticker':1}).to_zarr('./data/nasdaq_preds')
            # combined.to_csv(f'./data/backtests/{ticker}_{today}.csv')



    # result = backtester.backtest_single(combined = combined, funds = funds, buy_threshold = self.buy_threshold, take_limit = self.take_limit, stop_loss = self.stop_loss)

results = []
for ticker in preds['ticker'].values:
    pred = preds['preds'].sel(ticker=ticker).drop_vars('ticker').to_pandas()
    pred['timestamp'] = pred.index.values
    his = hist['ohlcv'].sel(ticker = ticker).drop_vars('ticker').to_pandas()
# his['timestamp'] = his.index.values
    combined = pd.merge(pred.drop('open', axis = 1), his, how = 'inner', left_index = True, right_index = True)
    results.append(BackTester.backtest_single(combined, 10000, 0.02, 0.02, 0.05))