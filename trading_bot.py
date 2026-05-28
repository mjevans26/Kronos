"""
Kronos × Alpaca  —  Automated Trading Bot
==========================================
Uses Kronos (OHLCV foundation model) to generate signals and
Alpaca's free paper-trading API to execute orders automatically.

──────────────────────────────────────────
SETUP  (one-time)
──────────────────────────────────────────
1. Clone Kronos and install deps:
       git clone https://github.com/shiyu-coder/Kronos.git
       cd Kronos
       pip install -r requirements.txt
       pip install alpaca-py yfinance python-dotenv schedule

2. Sign up free at https://app.alpaca.markets
   → switch to "Paper Trading" account
   → generate API keys  (Settings → API Keys)

3. Create a .env file in the same directory:
       ALPACA_API_KEY=your_paper_key_here
       ALPACA_SECRET_KEY=your_paper_secret_here
       PAPER_TRADING=true        # set to false for live (careful!)

4. Place this file inside the cloned Kronos/ directory, then:
       python trading_bot.py

──────────────────────────────────────────
ARCHITECTURE
──────────────────────────────────────────
  ┌─────────────┐    OHLCV      ┌──────────────┐
  │  Alpaca     │──────────────▶│   Kronos      │
  │  Market     │   last N bars │   Forecaster  │
  │  Data API   │               └──────┬────────┘
  └─────────────┘                      │ BUY / SELL / HOLD
                                        ▼
  ┌─────────────┐   orders      ┌──────────────┐
  │  Alpaca     │◀──────────────│   Risk        │
  │  Trading    │               │   Manager     │
  │  API        │               └──────────────┘
  └─────────────┘

The bot runs every market day at a configurable time (default: 09:35 ET).
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime, timedelta, date

warnings.filterwarnings("ignore")

# ── third-party ────────────────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import schedule
    import yfinance as yf
    from dotenv import load_dotenv
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n"
             "Run: pip install alpaca-py yfinance python-dotenv schedule numpy pandas")

# ── Alpaca SDK ─────────────────────────────────────────────────────────────
try:
    from alpaca.trading.client   import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    from alpaca.trading.enums    import OrderClass, OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.data.historical  import StockHistoricalDataClient
    from alpaca.data.requests    import StockBarsRequest
    from alpaca.data.timeframe   import TimeFrame
except ImportError:
    sys.exit("alpaca-py not found.\nRun: pip install alpaca-py")

# ── Kronos model ───────────────────────────────────────────────────────────
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    sys.exit(
        "\n❌  Cannot import Kronos.\n"
        "    Run this script from inside the cloned Kronos repo:\n"
        "        git clone https://github.com/shiyu-coder/Kronos.git\n"
        "        cd Kronos && python trading_bot.py\n"
    )

# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Watchlist ──────────────────────────────────────────────────────────
    "symbols": ["AAPL", "MSFT", "NVDA", "SPY"],   # stocks to trade

    # ── Kronos ────────────────────────────────────────────────────────────
    "model_id":       "NeoQuasar/Kronos-small",
    "tokenizer_id":   "NeoQuasar/Kronos-Tokenizer-base",
    "device":         "cpu",         # "cuda:0" if you have a GPU
    "max_context":    512,
    "forecast_steps": 5,
    "context_bars":   120,           # candles fed to Kronos as context
    "T": 0.7,
    "n_samples":10,

    # ── Signal thresholds ─────────────────────────────────────────────────
    "buy_threshold":  0.005,         # predicted return > +0.5 % → BUY
    "sell_threshold": -0.005,        # predicted return < -0.5 % → SELL

    # ── Risk / position sizing ────────────────────────────────────────────
    "max_position_pct":  0.20,       # max 20 % of equity per symbol
    "max_portfolio_pct": 0.80,       # keep 20 % cash reserve
    "stop_loss_pct":     0.03,       # close position if loss > 3 %
    "take_profit_pct":   0.08,       # close position if gain > 8 %

    # ── Scheduler ─────────────────────────────────────────────────────────
    # Runs once per day shortly after market open (ET).
    # The bot also runs a midday check and an end-of-day risk sweep.
    "run_times": ["09:35", "12:00", "15:45"],   # ET  (HH:MM, 24 h)

    # ── Misc ──────────────────────────────────────────────────────────────
    "dry_run": True,    # True = log orders but don't submit them
    "log_file": "./logs/trading_bot.log",
}

# ══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CONFIG["log_file"]),
    ],
)
log = logging.getLogger("KronosBot")


# ══════════════════════════════════════════════════════════════════════════
#  ALPACA CLIENT WRAPPER
# ══════════════════════════════════════════════════════════════════════════
class AlpacaClient:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.paper = paper
        self.trader = TradingClient(api_key, secret_key, paper=paper)
        self.data   = StockHistoricalDataClient(api_key, secret_key)
        mode = "PAPER" if paper else "⚠️  LIVE"
        log.info(f"Alpaca connected  [{mode}]")

    # ── Account ────────────────────────────────────────────────────────────
    def get_equity(self) -> float:
        return float(self.trader.get_account().equity)

    def get_cash(self) -> float:
        return float(self.trader.get_account().cash)

    def get_buying_power(self) -> float:
        return float(self.trader.get_account().buying_power)

    # ── Positions ──────────────────────────────────────────────────────────
    def get_positions(self) -> dict:
        """Returns {symbol: position_object}"""
        return {p.symbol: p for p in self.trader.get_all_positions()}

    def get_position(self, symbol: str):
        try:
            return self.trader.get_open_position(symbol)
        except Exception:
            return None

    # ── Orders ─────────────────────────────────────────────────────────────
    def cancel_open_orders(self, symbol: str):
        req    = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        orders = self.trader.get_orders(filter=req)
        for o in orders:
            try:
                self.trader.cancel_order_by_id(o.id)
                log.info(f"  cancelled order {o.id} for {symbol}")
            except Exception as e:
                log.warning(f"  could not cancel {o.id}: {e}")

    def bracket_order(self, symbol: str, position: str, notional: float, take_limit:float, stop_loss:float) -> bool:
        """Buy $notional worth of symbol (fractional shares supported)."""
        if notional < 1:
            log.warning(f"  {symbol}: notional ${notional:.2f} too small, skipping")
            return False
        
        if position == 'long':
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class = OrderClass.BRACKET,
                take_profit = {'limit_price':take_limit},
                stop_loss = {'stop_price':stop_loss}
            )

        elif position == 'short':
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class = OrderClass.BRACKET,
                take_profit = {'limit_price':stop_loss},
                stop_loss = {'stop_price':take_limit}
            )            

        if CONFIG["dry_run"]:
            log.info(f"  [DRY RUN] BUY ${notional:.2f} of {symbol}")
            return True
        try:
            order = self.trader.submit_order(req)
            log.info(f"  ✅ BUY  {symbol}  ${notional:.2f}  order_id={order.id}")
            return True
        except Exception as e:
            log.error(f"  ❌ BUY {symbol} failed: {e}")
            return False


    def market_buy(self, symbol: str, notional: float) -> bool:
        """Buy $notional worth of symbol (fractional shares supported)."""
        if notional < 1:
            log.warning(f"  {symbol}: notional ${notional:.2f} too small, skipping")
            return False
        
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        if CONFIG["dry_run"]:
            log.info(f"  [DRY RUN] BUY ${notional:.2f} of {symbol}")
            return True
        try:
            order = self.trader.submit_order(req)
            log.info(f"  ✅ BUY  {symbol}  ${notional:.2f}  order_id={order.id}")
            return True
        except Exception as e:
            log.error(f"  ❌ BUY {symbol} failed: {e}")
            return False

    def market_sell_all(self, symbol: str) -> bool:
        """Liquidate entire position in symbol."""
        pos = self.get_position(symbol)
        if pos is None:
            return False
        qty = float(pos.qty)
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        if CONFIG["dry_run"]:
            log.info(f"  [DRY RUN] SELL {qty} shares of {symbol}")
            return True
        try:
            order = self.trader.submit_order(req)
            log.info(f"  ✅ SELL {symbol}  qty={qty}  order_id={order.id}")
            return True
        except Exception as e:
            log.error(f"  ❌ SELL {symbol} failed: {e}")
            return False

    # ── Market data ────────────────────────────────────────────────────────
    def get_bars(self, symbol: str, n_bars: int = 150, adjustment = None) -> pd.DataFrame:
        """Fetch recent daily OHLCV bars via Alpaca (free IEX feed)."""
        end   = datetime.now()
        start = end - timedelta(days=int(n_bars * 1.6))  # buffer for weekends
        req   = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=adjustment
        )
        bars = self.data.get_stock_bars(req).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level="symbol")
        bars = bars.rename(columns=str.lower)
        bars.index = pd.to_datetime(bars.index)
        bars['timestamp'] = bars.index.values
        if isinstance(bars.index, pd.DatetimeTZLocaware if hasattr(pd, "DatetimeTZLocaware") else type(bars.index)):
            bars.index = bars.index.tz_localize(None) if bars.index.tzinfo is None else bars.index.tz_convert(None)
        return bars.tail(n_bars)


# ══════════════════════════════════════════════════════════════════════════
#  KRONOS FORECASTER
# ══════════════════════════════════════════════════════════════════════════
class KronosForecaster:
    def __init__(self, cfg: dict):
        log.info(f"Loading Kronos  [{cfg['model_id']}]  device={cfg['device']}")
        tokenizer     = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
        model         = Kronos.from_pretrained(cfg["model_id"])
        self.predictor = KronosPredictor(
            model, tokenizer,
            device=cfg["device"],
            max_context=cfg["max_context"]
        )
        self.horizon  = cfg["forecast_steps"]
        self.T = cfg["T"]
        self.samples = cfg["n_samples"]
        log.info("Kronos ready ✅")

    def forecast(self, bars: pd.DataFrame) -> float | None:
        """
        Feed OHLCV bars to Kronos.
        Returns the predicted close-price at step 1 (next bar),
        or None if prediction fails.
        """
        ohlc = ["open", "high", "low", "close"]
        ctx_df = bars[ohlc].copy()
        if "volume" in bars.columns:
            ctx_df["volume"] = bars["volume"].values
            ctx_df["amount"] = (bars["volume"] * bars["close"]).values

        x_ts     = bars['timestamp'] #bars.index
        # freq     = (bars.timestamp[-1] - bars.timestamp[-2]).days or 1 #(bars.index[-1] - bars.index[-2]).days or 1
        freq = len(pd.bdate_range(bars.index[-2], bars.index[-1])) - 1
        y_ts     = pd.Series(pd.bdate_range(
            # start   = bars.index[-1] + timedelta(days=freq),
            start = ['timestamp'].iloc[-1] + pd.offsets.BDay(freq),
            periods = self.horizon,
            # freq    = f"{freq}D",
            freq = f"{freq}B", # b for business day
        ))

        try:
            fc = self.predictor.predict(
                df=ctx_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len = self.horizon,
                T=self.T,
                top_p=0.9,
                sample_count=self.n_samples)
            # if hasattr(fc, "median"):
            #     return float(fc.median(axis = 0).values[0])
            # elif isinstance(fc, np.ndarray):
            #     return float(np.median(fc[0]))
            # return float(fc[0])
            return fc[['open', 'high', 'low', 'close']].values
        except Exception as e:
            log.warning(f"  Kronos forecast error: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════
#  RISK MANAGER
# ══════════════════════════════════════════════════════════════════════════
class RiskManager:
    def __init__(self, cfg: dict):
        self.max_pos_pct  = cfg["max_position_pct"]
        self.max_port_pct = cfg["max_portfolio_pct"]
        self.stop_loss    = cfg["stop_loss_pct"]
        self.take_profit  = cfg["take_profit_pct"]

    def position_size(self, equity: float, cash: float, current_price: float) -> float:
        """Return dollar amount to invest in a new BUY order."""
        max_by_equity  = equity * self.max_pos_pct
        max_by_cash    = cash   * self.max_port_pct
        return min(max_by_equity, max_by_cash, cash * 0.99)

    def should_stop(self, position) -> tuple[bool, str]:
        """Check if a position should be closed by stop-loss or take-profit."""
        if position is None:
            return False, ""
        unrealized_pct = float(position.unrealized_plpc)
        if unrealized_pct <= -self.stop_loss:
            return True, f"stop-loss ({unrealized_pct:.2%})"
        if unrealized_pct >= self.take_profit:
            return True, f"take-profit ({unrealized_pct:.2%})"
        return False, ""


# ══════════════════════════════════════════════════════════════════════════
#  TRADING BOT
# ══════════════════════════════════════════════════════════════════════════
class TradingBot:
    def __init__(self):
        load_dotenv()
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

        self.alpaca    = AlpacaClient(api_key, secret_key, paper=paper)
        self.kronos    = KronosForecaster(CONFIG)
        self.risk      = RiskManager(CONFIG)
        self.symbols   = CONFIG["symbols"]
        self.ctx_bars  = CONFIG["context_bars"]
        self.buy_thr   = CONFIG["buy_threshold"]
        self.short_thr  = CONFIG["short_threshold"]
        self.stop_loss = CONFIG['stop_loss_pct']
        self.take_limit = CONFIG['take_profit_pct']

    # ── Core loop ──────────────────────────────────────────────────────────
    def run_once(self):
        now = datetime.now()
        log.info("=" * 60)
        log.info(f"  Cycle start  {now:%Y-%m-%d %H:%M:%S}")
        log.info("=" * 60)

        equity     = self.alpaca.get_equity()
        cash       = self.alpaca.get_cash()
        positions  = self.alpaca.get_positions()

        log.info(f"  Equity: ${equity:,.2f}  |  Cash: ${cash:,.2f}")

        for symbol in self.symbols:
            log.info(f"\n── {symbol} ──")
            self._process_symbol(symbol, equity, cash, positions)

        log.info("\n  Cycle complete.\n")

    def _process_symbol(self, symbol: str, equity: float, cash: float, positions: dict):
        # ── 1. Fetch bars ────────────────────────────────────────────────
        try:
            bars = self.alpaca.get_bars(symbol, n_bars=self.ctx_bars)
        except Exception as e:
            log.warning(f"  Could not fetch bars for {symbol}: {e}")
            log.info(f"  Falling back to yfinance for {symbol}")
            try:
                bars = self._yfinance_bars(symbol, self.ctx_bars)
            except Exception as e2:
                log.error(f"  yfinance also failed for {symbol}: {e2}")
                return

        if len(bars) < self.ctx_bars:
            log.warning(f"  Not enough bars ({len(bars)}) for {symbol}, skipping")
            return

        current_close = float(bars["close"].iloc[-1])

        # ── 2. Risk check on existing position ──────────────────────────
        pos = positions.get(symbol)
        if pos is not None:
            hit, reason = self.risk.should_stop(pos)
            if hit:
                log.info(f"  {symbol}: closing position  [{reason}]")
                self.alpaca.cancel_open_orders(symbol)
                self.alpaca.market_sell_all(symbol)
                return

        # ── 3. Kronos forecast ───────────────────────────────────────────
        pred_close = self.kronos.forecast(bars)
        if pred_close is None:
            log.warning(f"  {symbol}: no forecast, skipping")
            return

        pred_return = (pred_close - current_close) / current_close
        log.info(
            f"  {symbol}  current={current_close:.2f}  "
            f"predicted={pred_close:.2f}  ret={pred_return:+.3f}"
        )

        # ── 4. Execute signal ────────────────────────────────────────────
        if pred_return > self.buy_thr:
            if pos is not None:
                log.info(f"  {symbol}: already holding, skip BUY")
            else:
                take_limit = self.take_limit*current_close
                stop_loss = self.stop_loss*current_close
                notional = self.risk.position_size(equity, cash, current_close)
                log.info(f"  {symbol}: 🟢 BUY  ${notional:.2f}")
                if self.alpaca.bracket_order(symbol, 'long', notional, take_limit, stop_loss):
                # if self.alpaca.market_buy(symbol, notional):
                    cash -= notional   # update local cash estimate

        elif pred_return < self.short_thr:
            if pos is not None:
                log.info(f"  {symbol}: already shorting, skip BUY")
                # log.info(f"  {symbol}: 🔴 SELL  closing position")
                # self.alpaca.cancel_open_orders(symbol)
                # self.alpaca.market_sell_all(symbol)
            else:
                # TODO: Create short sale order?
                log.info(f"  {symbol}: SELL signal and no position, shorting")
                notional = self.risk.position_size(equity, cash, current_close)
                log.info(f"  {symbol}: 🟢 SHORT SELL  ${notional:.2f}")
                if self.alpaca.bracket_order(symbol, 'short', notional, take_limit, stop_loss):
                # if self.alpaca.market_buy(symbol, notional):
                    cash -= notional   # update local cash estimate
        else:
            log.info(f"  {symbol}: ⬜ HOLD")

    # ── Fallback data source ───────────────────────────────────────────────
    def _yfinance_bars(self, symbol: str, n: int) -> pd.DataFrame:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        df.columns = [c.lower() for c in df.columns.get_level_values(0)]
        df.dropna(inplace=True)
        return df.tail(n)

    # ── Status report ──────────────────────────────────────────────────────
    def print_status(self):
        equity    = self.alpaca.get_equity()
        cash      = self.alpaca.get_cash()
        positions = self.alpaca.get_positions()
        log.info("\n📊  PORTFOLIO STATUS")
        log.info(f"  Equity : ${equity:,.2f}")
        log.info(f"  Cash   : ${cash:,.2f}")
        if positions:
            log.info("  Open positions:")
            for sym, p in positions.items():
                pnl = float(p.unrealized_pl)
                pct = float(p.unrealized_plpc) * 100
                log.info(f"    {sym:<6}  qty={float(p.qty):.4f}  "
                          f"mkt=${float(p.market_value):,.2f}  "
                          f"P&L={pnl:+,.2f} ({pct:+.2f}%)")
        else:
            log.info("  No open positions.")

    # ── Scheduler ──────────────────────────────────────────────────────────
    def start(self):
        log.info("🤖  Kronos × Alpaca Trading Bot starting")
        log.info(f"    Symbols   : {self.symbols}")
        log.info(f"    Run times : {CONFIG['run_times']} ET")
        log.info(f"    Paper     : {self.alpaca.paper}")
        log.info(f"    Dry run   : {CONFIG['dry_run']}")

        self.print_status()

        for t in CONFIG["run_times"]:
            schedule.every().day.at(t).do(self.run_once)
            log.info(f"    Scheduled at {t}")

        log.info("\nWaiting for next scheduled run… (Ctrl-C to stop)\n")

        # Run immediately on first start so you don't wait until next window
        self.run_once()

        while True:
            schedule.run_pending()
            time.sleep(30)


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kronos × Alpaca Trading Bot")
    parser.add_argument("--once",   action="store_true", help="Run one cycle then exit")
    parser.add_argument("--status", action="store_true", help="Print portfolio status and exit")
    parser.add_argument("--dry",    action="store_true", help="Override config: dry_run=True")
    args = parser.parse_args()

    if args.dry:
        CONFIG["dry_run"] = True

    bot = TradingBot()

    if args.status:
        bot.print_status()
    elif args.once:
        bot.run_once()
        bot.print_status()
    else:
        bot.start()