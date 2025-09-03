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
            msg = f"🚨 SELL ALERT! Price {price:.4f} ≤ SMA {sma:.4f}, RSI={rsi:.2f}"
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
        df7 = df.tail(7).copy()
        df7['% Diff'] = df7['close'].pct_change() * 100
        df7['% Diff'] = df7['% Diff'].round(2)
        df7['close'] = df7['close'].round(2)
        df7 = df7.reset_index()[['datetime', 'close', '% Diff']]
        df7.columns = ['Date', 'Close', '% Diff']

        print("\n📅 Last 7 Days Closing Prices:\n")
        print(df7.to_string(index=False))

        # save CSV only
        df7.to_csv("last7days.csv", index=False)
        #print("📂 Saved as last7days.csv")

        return df7
        
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

    # export all data
    #results['data'].to_csv("analysis_output.csv")
    #print("📂 Full data exported to analysis_output.csv")

    # last 7 days table
    alert_system.print_last7days_table(results['data'])

    # plot chart
    alert_system.plot_chart(results['data'])


if __name__ == "__main__":
    main()
