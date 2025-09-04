import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from typing import Dict, Optional
import matplotlib.dates as mdates

plt.style.use('default')


class PriceAnalyzer:
    def __init__(self, symbol: str = "SCBGHC",
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 rsi_overbought: int = 70,
                 rsi_oversold: int = 30,
                 volume_multiplier: float = 1.5):
        self.symbol = symbol
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_multiplier = volume_multiplier
        self.base_url = "https://www.finnomena.com/fn3/api/fund/v2/public/tv/history"

    def get_timestamps(self, days_back: int = 100) -> tuple[int, int]:
        yesterday = datetime.now() - timedelta(days=1)
        from_date = yesterday - timedelta(days=days_back)
        return int(from_date.timestamp()), int(yesterday.timestamp())

    def fetch_price_data(self, days_back: int = 100) -> Optional[pd.DataFrame]:
        from_ts, to_ts = self.get_timestamps(days_back)
        params = {
            'symbol': self.symbol,
            'resolution': '1D',
            'from': from_ts,
            'to': to_ts,
            'currencyCode': 'THB'
        }
        headers = {'User-Agent': 'Mozilla/5.0'}

        for attempt in range(3):
            try:
                response = requests.get(
                    self.base_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data.get('s') != 'ok':
                    continue
                df = pd.DataFrame({
                    'timestamp': data['t'],
                    'close': data['c'],
                    'volume': data['v']
                })
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('datetime', inplace=True)
                return df
            except Exception as e:
                print(f"⚠️ Fetch attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        return None

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sma'] = df['close'].rolling(window=self.bb_period).mean()
        df['std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['sma'] + (df['std'] * self.bb_std)
        df['bb_lower'] = df['sma'] - (df['std'] * self.bb_std)
        return df

    def check_signals(self, df: pd.DataFrame, idx: int) -> Dict:
        if len(df) < max(self.bb_period, self.rsi_period) + 1:
            return {'alert': False, 'signal': 'HOLD', 'message': 'Insufficient data'}

        latest = df.iloc[idx]
        if pd.isna(latest['sma']):
            return {'alert': False, 'signal': 'HOLD', 'message': 'Indicators not ready'}

        price = latest['close']
        sma = latest['sma']
        volume_spike = latest['volume_spike']

        # Strategy: Buy when price > SMA, Sell when price < SMA
        sell_cond = price < sma
        buy_cond = price > sma

        msg = f"✅ HOLD. Price {price:.4f} | SMA {sma:.4f}"
        signal = 'HOLD'
        if sell_cond:
            msg = f"🚨 SELL! Price {price:.4f} < SMA {sma:.4f}"
            if volume_spike:
                msg += " + Volume Spike 🔥"
            signal = 'SELL'
        elif buy_cond:
            msg = f"🟢 BUY! Price {price:.4f} > SMA {sma:.4f}"
            if volume_spike:
                msg += " + Volume Spike ⚡"
            signal = 'BUY'

        return {
            'alert': sell_cond or buy_cond,
            'signal': signal,
            'current_price': price,
            'sma': sma,
            'volume_spike': volume_spike,
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'message': msg,
            'timestamp': latest.name
        }

    def calculate_percentage_differences(self, df):
        """Calculate percentage differences between current price and prices from 1, 3, 5, 7, 14, 30 days ago"""
        if len(df) < 30:
            return None

        current_price = df['close'].iloc[-1]
        current_date = df.index[-1]

        # Find the closest available dates to our target dates
        def find_closest_date(target_date):
            # Get all dates in the DataFrame
            all_dates = df.index
            # Find the closest date (before or equal to target)
            mask = all_dates <= target_date
            if not any(mask):
                return None
            return all_dates[mask][-1]

        # Calculate target dates
        date_1d = current_date - timedelta(days=1)
        date_3d = current_date - timedelta(days=3)
        date_5d = current_date - timedelta(days=5)
        date_7d = current_date - timedelta(days=7)
        date_14d = current_date - timedelta(days=14)
        date_30d = current_date - timedelta(days=30)

        # Find closest dates in the data
        closest_1d = find_closest_date(date_1d)
        closest_3d = find_closest_date(date_3d)
        closest_5d = find_closest_date(date_5d)
        closest_7d = find_closest_date(date_7d)
        closest_14d = find_closest_date(date_14d)
        closest_30d = find_closest_date(date_30d)

        # Get prices at these dates
        price_1d = df.loc[closest_1d,
                          'close'] if closest_1d is not None else None
        price_3d = df.loc[closest_3d,
                          'close'] if closest_3d is not None else None
        price_5d = df.loc[closest_5d,
                          'close'] if closest_5d is not None else None
        price_7d = df.loc[closest_7d,
                          'close'] if closest_7d is not None else None
        price_14d = df.loc[closest_14d,
                           'close'] if closest_14d is not None else None
        price_30d = df.loc[closest_30d,
                           'close'] if closest_30d is not None else None

        # Calculate percentage differences
        pct_diff_1d = ((current_price - price_1d) / price_1d *
                       100) if price_1d is not None else None
        pct_diff_3d = ((current_price - price_3d) / price_3d *
                       100) if price_3d is not None else None
        pct_diff_5d = ((current_price - price_5d) / price_5d *
                       100) if price_5d is not None else None
        pct_diff_7d = ((current_price - price_7d) / price_7d *
                       100) if price_7d is not None else None
        pct_diff_14d = ((current_price - price_14d) /
                        price_14d * 100) if price_14d is not None else None
        pct_diff_30d = ((current_price - price_30d) /
                        price_30d * 100) if price_30d is not None else None

        # Create result dictionary
        result = {
            'current_date': current_date,
            'current_price': current_price,
            '1_day': {
                'date': closest_1d,
                'price': price_1d,
                'pct_change': pct_diff_1d
            },
            '3_days': {
                'date': closest_3d,
                'price': price_3d,
                'pct_change': pct_diff_3d
            },
            '5_days': {
                'date': closest_5d,
                'price': price_5d,
                'pct_change': pct_diff_5d
            },
            '7_days': {
                'date': closest_7d,
                'price': price_7d,
                'pct_change': pct_diff_7d
            },
            '14_days': {
                'date': closest_14d,
                'price': price_14d,
                'pct_change': pct_diff_14d
            },
            '30_days': {
                'date': closest_30d,
                'price': price_30d,
                'pct_change': pct_diff_30d
            }
        }

        return result

    def print_percentage_differences_table(self, pct_diffs):
        """Print a simple table of percentage differences"""
        if not pct_diffs:
            print("Insufficient data to calculate percentage differences")
            return

        print("\nPERCENTAGE DIFFERENCE ANALYSIS")
        print(f"Symbol: {self.symbol}")
        print(
            f"Current Price ({pct_diffs['current_date'].strftime('%Y-%m-%d')}): {pct_diffs['current_price']:.4f} THB")
        print()

        # Print header
        print(f"{'Period':<8} {'Date':<12} {'Price':<10} {'Change':<10}")
        print("-" * 45)

        # 1 day row
        if pct_diffs['1_day']['date'] is not None:
            change_1d = pct_diffs['1_day']['pct_change']
            trend_1d = "↑" if change_1d > 0 else "↓" if change_1d < 0 else "→"
            print(
                f"{'1 Day':<8} {pct_diffs['1_day']['date'].strftime('%Y-%m-%d'):<12} {pct_diffs['1_day']['price']:>9.4f} {change_1d:>+9.2f}% {trend_1d}")

        # 3 days row
        if pct_diffs['3_days']['date'] is not None:
            change_3d = pct_diffs['3_days']['pct_change']
            trend_3d = "↑" if change_3d > 0 else "↓" if change_3d < 0 else "→"
            print(
                f"{'3 Days':<8} {pct_diffs['3_days']['date'].strftime('%Y-%m-%d'):<12} {pct_diffs['3_days']['price']:>9.4f} {change_3d:>+9.2f}% {trend_3d}")

        # 5 days row
        if pct_diffs['5_days']['date'] is not None:
            change_5d = pct_diffs['5_days']['pct_change']
            trend_5d = "↑" if change_5d > 0 else "↓" if change_5d < 0 else "→"
            print(
                f"{'5 Days':<8} {pct_diffs['5_days']['date'].strftime('%Y-%m-%d'):<12} {pct_diffs['5_days']['price']:>9.4f} {change_5d:>+9.2f}% {trend_5d}")

        # 7 days row
        if pct_diffs['7_days']['date'] is not None:
            change_7d = pct_diffs['7_days']['pct_change']
            trend_7d = "↑" if change_7d > 0 else "↓" if change_7d < 0 else "→"
            print(
                f"{'7 Days':<8} {pct_diffs['7_days']['date'].strftime('%Y-%m-%d'):<12} {pct_diffs['7_days']['price']:>9.4f} {change_7d:>+9.2f}% {trend_7d}")

        # 14 days row
        if pct_diffs['14_days']['date'] is not None:
            change_14d = pct_diffs['14_days']['pct_change']
            trend_14d = "↑" if change_14d > 0 else "↓" if change_14d < 0 else "→"
            print(
                f"{'14 Days':<8} {pct_diffs['14_days']['date'].strftime('%Y-%m-%d'):<12} {pct_diffs['14_days']['price']:>9.4f} {change_14d:>+9.2f}% {trend_14d}")

        # 30 days row
        if pct_diffs['30_days']['date'] is not None:
            change_30d = pct_diffs['30_days']['pct_change']
            trend_30d = "↑" if change_30d > 0 else "↓" if change_30d < 0 else "→"
            print(
                f"{'30 Days':<8} {pct_diffs['30_days']['date'].strftime('%Y-%m-%d'):<12} {pct_diffs['30_days']['price']:>9.4f} {change_30d:>+9.2f}% {trend_30d}")

    def analyze_data(self, days_back: int = 200):
        df = self.fetch_price_data(days_back)
        if df is None or df.empty:
            return {'error': 'No data'}

        df = self.calculate_bollinger_bands(df)
        df['rsi'] = self.calculate_rsi(df, self.rsi_period)
        df['avg_volume'] = df['volume'].rolling(window=self.bb_period).mean()
        df['volume_spike'] = df['volume'] > (
            df['avg_volume'] * self.volume_multiplier)

        # Check signals for each day
        signals = []
        for i in range(len(df)):
            if i < self.bb_period:
                continue

            signal = self.check_signals(df, i)
            signals.append(signal)

        # Calculate percentage differences
        pct_diffs = self.calculate_percentage_differences(df)

        return {
            'data': df,
            'signals': signals,
            'pct_diffs': pct_diffs
        }

    def plot_analysis(self, df, signals):
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # Plot price and indicators
        ax1.plot(df.index, df['close'],
                 label='Close Price', color='blue', linewidth=2)
        ax1.plot(df.index, df['sma'], label='SMA',
                 color='orange', linewidth=1.5)
        ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'],
                         color='gray', alpha=0.2, label='Bollinger Bands')

        # Highlight areas where price is above/below SMA
        ax1.fill_between(df.index, df['close'], df['sma'], where=df['close'] >= df['sma'],
                         color='green', alpha=0.1, label='Price > SMA (Buy Zone)')
        ax1.fill_between(df.index, df['close'], df['sma'], where=df['close'] < df['sma'],
                         color='red', alpha=0.1, label='Price < SMA (Sell Zone)')

        ax1.set_title(f'{self.symbol} - Price and SMA with Buy/Sell Signals',
                      fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (THB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Mark buy/sell signals with text annotations
        for i, signal in enumerate(signals):
            if i < self.bb_period:
                continue

            date = signal['timestamp']
            price = signal['current_price']
            signal_type = signal['signal']

            # Add text annotation
            if signal_type == 'BUY':
                ax1.annotate('BUY', xy=(date, price), xytext=(date, price * 1.02),
                             ha='center', va='bottom', fontsize=14, fontweight='bold',
                             color='green',
                             arrowprops=dict(arrowstyle='->',
                                             color='green', lw=2),
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", edgecolor="green", alpha=0.9))

            elif signal_type == 'SELL':
                ax1.annotate('SELL', xy=(date, price), xytext=(date, price * 0.98),
                             ha='center', va='top', fontsize=14, fontweight='bold',
                             color='red',
                             arrowprops=dict(arrowstyle='->',
                                             color='red', lw=2),
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", edgecolor="red", alpha=0.9))

        # Plot RSI
        ax2.plot(df.index, df['rsi'], label='RSI', color='purple', linewidth=2)
        ax2.axhline(y=self.rsi_overbought, color='red',
                    linestyle='--', label='Overbought (70)')
        ax2.axhline(y=50, color='gray', linestyle=':',
                    alpha=0.7, label='Neutral (50)')
        ax2.axhline(y=self.rsi_oversold, color='green',
                    linestyle='--', label='Oversold (30)')
        ax2.fill_between(df.index, self.rsi_overbought, df['rsi'], where=df['rsi'] >= self.rsi_overbought,
                         color='red', alpha=0.2)
        ax2.fill_between(df.index, self.rsi_oversold, df['rsi'], where=df['rsi'] <= self.rsi_oversold,
                         color='green', alpha=0.2)
        ax2.set_title('RSI Indicator', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def print_current_signal(self, signals):
        if not signals:
            print("No signals available")
            return

        latest_signal = signals[-1]
        print(
            f"\nCURRENT SIGNAL FOR {self.symbol} ({latest_signal['timestamp'].strftime('%Y-%m-%d')})")
        print(latest_signal['message'])
        print(f"Current Price: {latest_signal['current_price']:.4f}")
        print(f"SMA: {latest_signal['sma']:.4f}")

    def print_recent_signals(self, signals, num_signals=10):
        if not signals:
            print("No signals available")
            return

        print(f"\nRECENT SIGNALS FOR {self.symbol} (Last {num_signals})")

        # Get the most recent signals
        recent_signals = signals[-num_signals:] if len(
            signals) > num_signals else signals

        for signal in recent_signals:
            date_str = signal['timestamp'].strftime('%Y-%m-%d')
            price = signal['current_price']
            sma = signal['sma']
            signal_type = signal['signal']

            if signal_type == 'BUY':
                print(f"🟢 {date_str}: BUY at {price:.4f} (SMA: {sma:.4f})")
            elif signal_type == 'SELL':
                print(f"🔴 {date_str}: SELL at {price:.4f} (SMA: {sma:.4f})")
            else:
                print(f"⚪ {date_str}: HOLD at {price:.4f} (SMA: {sma:.4f})")


def main():
    # Initialize price analyzer with SCBGHC symbol
    analyzer = PriceAnalyzer(symbol="KTFIXPLUS-A")

    # Analyze data
    results = analyzer.analyze_data(days_back=200)

    if 'error' in results:
        print("❌ Error:", results['error'])
        return

    # Print current signal
    analyzer.print_current_signal(results['signals'])

    # Print percentage differences table
    analyzer.print_percentage_differences_table(results['pct_diffs'])

    # Print recent signals
    # analyzer.print_recent_signals(results['signals'])

    # Plot analysis
    analyzer.plot_analysis(results['data'], results['signals'])


if __name__ == "__main__":
    main()
