import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
from typing import Dict, Optional


class PriceAlertSystem:
    def __init__(self, symbol: str = "KTFIXPLUS-A",
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
                response = requests.get(self.base_url, params=params, headers=headers, timeout=10)
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

    def check_signals(self, df: pd.DataFrame) -> Dict:
        if len(df) < max(self.bb_period, self.rsi_period) + 1:
            return {'alert': False, 'message': 'Insufficient data'}

        df = self.calculate_bollinger_bands(df)
        df['rsi'] = self.calculate_rsi(df, self.rsi_period)
        df['avg_volume'] = df['volume'].rolling(window=self.bb_period).mean()
        df['volume_spike'] = df['volume'] > (df['avg_volume'] * self.volume_multiplier)

        latest = df.iloc[-1]
        if pd.isna(latest['sma']) or pd.isna(latest['rsi']):
            return {'alert': False, 'message': 'Indicators not ready'}

        price = latest['close']
        sma = latest['sma']
        rsi = latest['rsi']
        volume_spike = latest['volume_spike']

        sell_cond = price <= sma and rsi >= self.rsi_overbought
        buy_cond = price >= sma and rsi <= self.rsi_oversold

        msg = f"✅ HOLD. Price {price:.4f} | SMA {sma:.4f} | RSI {rsi:.2f}"
        if sell_cond:
            msg = f"🚨 SEALERT! Price {price:.4f} ≤ SMA {sma:.4f}, RSI={rsi:.2f}"
            if volume_spike:
                msg += " + Volume Spike 🔥"
        elif buy_cond:
            msg = f"🟢 BUY ALERT! Price {price:.4f} ≥ SMA {sma:.4f}, RSI={rsi:.2f}"
            if volume_spike:
                msg += " + Volume Spike ⚡"

        return {
            'alert': sell_cond or buy_cond,
            'signal': 'SELL' if sell_cond else 'BUY' if buy_cond else 'HOLD',
            'current_price': price,
            'sma': sma,
            'rsi': round(rsi, 2),
            'volume_spike': volume_spike,
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'message': msg,
            'timestamp': latest.name
        }

    def plot_chart(self, df: pd.DataFrame):
        plt.figure(figsize=(12,6))
        plt.plot(df.index, df['close'], label="Close", color="blue")
        plt.plot(df.index, df['sma'], label="SMA", color="orange")
        plt.fill_between(df.index, df['bb_upper'], df['bb_lower'], color="gray", alpha=0.2)
        plt.title(f"{self.symbol} Price & Bollinger Bands")
        plt.xlabel("Date")
        plt.ylabel("Price (THB)")
    
        # Only annotate the most recent price
        latest = df.iloc[-1]
        rsi = latest['rsi']
        price = latest['close']
        sma = latest['sma']
    
        signal = "HOLD"
        if price <= sma and rsi >= self.rsi_overbought:
            signal = "SELL"
        elif price >= sma and rsi <= self.rsi_oversold:
            signal = "BUY"
    
        plt.text(df.index[-1], price + 0.02, f"{signal}", ha="center", fontweight='bold', fontsize=12, color="red" if signal=="SELL" else "green" if signal=="BUY" else "black")
    
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    

    def print_last7days_table(self, df: pd.DataFrame):
        # Get the latest date in the data
        latest_date = df.index[-1]
        
        # Calculate the dates for comparison
        date_7d = latest_date - timedelta(days=7)
        date_14d = latest_date - timedelta(days=14)
        date_30d = latest_date - timedelta(days=30)
        
        # Find the closest available dates to our target dates
        def find_closest_date(target_date):
            # Get all dates in the DataFrame
            all_dates = df.index
            # Find the closest date (before or equal to target)
            mask = all_dates <= target_date
            if not any(mask):
                return None
            return all_dates[mask][-1]
        
        closest_7d = find_closest_date(date_7d)
        closest_14d = find_closest_date(date_14d)
        closest_30d = find_closest_date(date_30d)
        
        # Get the prices at these dates
        current_price = df['close'].iloc[-1]
        price_7d = df.loc[closest_7d, 'close'] if closest_7d is not None else None
        price_14d = df.loc[closest_14d, 'close'] if closest_14d is not None else None
        price_30d = df.loc[closest_30d, 'close'] if closest_30d is not None else None
        
        # Calculate percentage differences
        pct_diff_7d = ((current_price - price_7d) / price_7d * 100) if price_7d is not None else None
        pct_diff_14d = ((current_price - price_14d) / price_14d * 100) if price_14d is not None else None
        pct_diff_30d = ((current_price - price_30d) / price_30d * 100) if price_30d is not None else None
        
        # Create a summary DataFrame
        summary_data = {
            'Period': ['7 days', '14 days', '30 days'],
            'Comparison Date': [
                closest_7d.strftime('%Y-%m-%d') if closest_7d else 'N/A',
                closest_14d.strftime('%Y-%m-%d') if closest_14d else 'N/A',
                closest_30d.strftime('%Y-%m-%d') if closest_30d else 'N/A'
            ],
            'Price at Comparison': [
                f"{price_7d:.4f}" if price_7d is not None else 'N/A',
                f"{price_14d:.4f}" if price_14d is not None else 'N/A',
                f"{price_30d:.4f}" if price_30d is not None else 'N/A'
            ],
            'Current Price': [f"{current_price:.4f}"] * 3,
            '% Difference': [
                f"{pct_diff_7d:.2f}%" if pct_diff_7d is not None else 'N/A',
                f"{pct_diff_14d:.2f}%" if pct_diff_14d is not None else 'N/A',
                f"{pct_diff_30d:.2f}%" if pct_diff_30d is not None else 'N/A'
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        print(f"\n📊 Percentage Difference Analysis (as of {latest_date.strftime('%Y-%m-%d')}):")
        print(summary_df.to_string(index=False))
        
        # Also show the last 7 days with daily changes
        df7 = df.tail(7).copy()
        df7['% Daily Change'] = df7['close'].pct_change() * 100
        df7['% Daily Change'] = df7['% Daily Change'].round(2)
        df7['close'] = df7['close'].round(4)
        df7 = df7.reset_index()[['datetime', 'close', '% Daily Change']]
        df7.columns = ['Date', 'Close', '% Daily Change']
        
        print("\n📅 Last 7 Days Closing Prices:\n")
        print(df7.to_string(index=False))
        
        # Save files
        #summary_df.to_csv("percentage_differences.csv", index=False)
        #df7.to_csv("last7days.csv", index=False)
        
        #print("\n📂 Saved as percentage_differences.csv and last7days.csv")
        
        return summary_df, df7
        
    def run_analysis(self, days_back: int = 100) -> Dict:
        df = self.fetch_price_data(days_back)
        if df is None or df.empty:
            return {'error': 'No data'}

        df = self.calculate_bollinger_bands(df)
        df['rsi'] = self.calculate_rsi(df, self.rsi_period)
        df['avg_volume'] = df['volume'].rolling(window=self.bb_period).mean()
        df['volume_spike'] = df['volume'] > (df['avg_volume'] * self.volume_multiplier)

        signals = self.check_signals(df)
        return {'signals': signals, 'data': df}

def main():
    alert_system = PriceAlertSystem(symbol="KTFIXPLUS-A")
    results = alert_system.run_analysis(days_back=200)

    if 'error' in results:
        print("❌ Error:", results['error'])
        return

    sig = results['signals']
    print(f"\n📊 FINAL REPORT {sig['timestamp']:%Y-%m-%d}:")
    print(sig['message'])

    # Print percentage differences and last 7 days table
    alert_system.print_last7days_table(results['data'])

    # Plot chart (optional)
    alert_system.plot_chart(results['data'])


if __name__ == "__main__":
    main()