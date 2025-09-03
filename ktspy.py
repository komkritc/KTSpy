import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional


class PriceAlertSystem:
    def __init__(self, symbol: str = "KTFIXPLUS-A",
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 rsi_period: int = 14,
                 volume_multiplier: float = 1.5):
        """
        Initialize Price Alert System with SMA, RSI, and Volume confirmation.

        Args:
            symbol: Fund symbol to monitor
            bb_period: Bollinger Bands period (also used for SMA)
            bb_std: Standard deviation for Bollinger Bands
            rsi_period: Period for RSI calculation (default 14)
            volume_multiplier: Volume must exceed avg_volume * this multiplier (e.g., 1.5x)
        """
        self.symbol = symbol
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.volume_multiplier = volume_multiplier
        self.base_url = "https://www.finnomena.com/fn3/api/fund/v2/public/tv/history"

    def get_timestamps(self, days_back: int = 100) -> tuple[int, int]:
        yesterday = datetime.now() - timedelta(days=1)
        from_date = yesterday - timedelta(days=days_back)
        return int(from_date.timestamp()), int(yesterday.timestamp())

    def fetch_price_data(self, days_back: int = 100) -> Optional[pd.DataFrame]:
        try:
            from_ts, to_ts = self.get_timestamps(days_back)
            params = {
                'symbol': self.symbol,
                'resolution': '1D',
                'from': from_ts,
                'to': to_ts,
                'currencyCode': 'THB'
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                              '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            response = requests.get(
                self.base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data.get('s') != 'ok':
                print(f"API error: {data.get('errmsg', 'Unknown')}")
                return None

            required = ['t', 'c', 'v']
            if not all(k in data for k in required):
                print("Missing data fields")
                return None
            if len(set(map(len, [data['t'], data['c'], data['v']]))) > 1:
                print("Length mismatch in API response")
                return None

            df = pd.DataFrame({
                'timestamp': data['t'],
                'close': data['c'],
                'volume': data['v']
            })
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI using close prices."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sma'] = df['close'].rolling(window=self.bb_period).mean()
        df['std'] = df['close'].rolling(window=self.bb_period).std()
        df['bb_upper'] = df['sma'] + (df['std'] * self.bb_std)
        df['bb_lower'] = df['sma'] - (df['std'] * self.bb_std)
        df['distance_from_sma_pct'] = abs(
            (df['close'] - df['sma']) / df['sma'] * 100)
        return df

    def check_sell_alert(self, df: pd.DataFrame) -> Dict:
        if len(df) < max(self.bb_period, self.rsi_period) + 1:
            return {'alert': False, 'message': 'Insufficient data for analysis'}

        # Calculate indicators
        df = self.calculate_bollinger_bands(df)
        df['rsi'] = self.calculate_rsi(df, self.rsi_period)
        df['avg_volume'] = df['volume'].rolling(window=self.bb_period).mean()
        df['volume_spike'] = df['volume'] > (
            df['avg_volume'] * self.volume_multiplier)

        latest = df.iloc[-1]

        if pd.isna(latest['sma']) or pd.isna(latest['rsi']):
            return {'alert': False, 'message': 'Indicator data not ready'}

        current_price = latest['close']
        sma = latest['sma']
        rsi = latest['rsi']
        volume_spike = latest['volume_spike']

        # 🔴 COMBINED SELL SIGNAL
        price_condition = current_price <= sma
        rsi_condition = rsi >= 70
        volume_condition = volume_spike

        strong_sell = price_condition and rsi_condition and volume_condition
        weak_sell = price_condition and rsi_condition  # without volume

        distance_pct = abs((current_price - sma) / sma * 100)

        alert_info = {
            'alert': strong_sell,
            'weak_alert': weak_sell and not strong_sell,
            'current_price': current_price,
            'sma': sma,
            'rsi': round(rsi, 2),
            'volume_spike': volume_spike,
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'distance_from_sma_pct': distance_pct,
            'timestamp': latest.name,
            'conditions': {
                'price_le_sma': price_condition,
                'rsi_ge_70': rsi_condition,
                'volume_spike': volume_condition
            },
            'message': ''
        }

        if strong_sell:
            alert_info['message'] = (
                f"🚨🔥 STRONG SELL ALERT! "
                f"Price {current_price:.4f} ≤ SMA {sma:.4f}, "
                f"RSI={rsi:.2f} (>70), "
                f"Volume spike detected"
            )
        elif weak_sell:
            alert_info['message'] = (
                f"🟡 WEAK SELL ALERT! "
                f"Price {current_price:.4f} ≤ SMA {sma:.4f}, "
                f"RSI={rsi:.2f} – Watch volume"
            )
        else:
            alert_info['message'] = (
                f"✅ HOLD. Price {current_price:.4f} | SMA {sma:.4f} | RSI {rsi:.2f}"
            )

        return alert_info

    def get_market_analysis(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {}

        latest = df.iloc[-1]
        if 'rsi' not in df or pd.isna(latest.get('rsi')):
            rsi_status = "N/A"
        elif latest['rsi'] >= 70:
            rsi_status = "Overbought (≥70)"
        elif latest['rsi'] <= 30:
            rsi_status = "Oversold (≤30)"
        else:
            rsi_status = "Neutral (30-70)"

        # Bollinger Band position
        if latest['close'] > latest['bb_upper']:
            bb_pos = "Above upper band"
        elif latest['close'] < latest['bb_lower']:
            bb_pos = "Below lower band"
        else:
            bb_pos = "Within bands"

        trend = "Upward" if df['close'].iloc[-1] > df['close'].iloc[-5] else "Downward"

        return {
            'bb_position': bb_pos,
            'rsi_status': rsi_status,
            'trend': trend,
            'current_volume': latest['volume'],
            'avg_volume': latest['avg_volume'],
            'volume_ratio': round(latest['volume'] / latest['avg_volume'], 2) if latest['avg_volume'] > 0 else 0,
            'days_analyzed': len(df)
        }

    def prepare_data_for_export(self, df: pd.DataFrame, num_records: int = 10) -> List[Dict]:
        recent_df = df.tail(num_records).copy()
        data_list = []
        for idx, row in recent_df.iterrows():
            record = {
                'date': idx.strftime('%Y-%m-%d'),
                'close': round(row['close'], 4),
                'volume': int(row['volume']),
                'sma': round(row['sma'], 4) if not pd.isna(row['sma']) else None,
                'rsi': round(row['rsi'], 2) if not pd.isna(row['rsi']) else None,
                'volume_spike': bool(row['volume_spike']) if not pd.isna(row['volume_spike']) else False,
                'bb_upper': round(row['bb_upper'], 4) if not pd.isna(row['bb_upper']) else None,
                'bb_lower': round(row['bb_lower'], 4) if not pd.isna(row['bb_lower']) else None
            }
            data_list.append(record)
        return data_list

    def run_analysis(self, days_back: int = 100) -> Dict:
        print(f"📊 Fetching data for {self.symbol}...")
        df = self.fetch_price_data(days_back)
        if df is None or df.empty:
            return {'error': 'Failed to fetch price data'}

        print("📈 Calculating indicators (SMA, RSI, Volume Avg, Bollinger Bands)...")
        df = self.calculate_bollinger_bands(df)
        df['rsi'] = self.calculate_rsi(df, self.rsi_period)
        df['avg_volume'] = df['volume'].rolling(window=self.bb_period).mean()
        df['volume_spike'] = df['volume'] > (
            df['avg_volume'] * self.volume_multiplier)

        print("🔍 Checking combined sell signal (SMA + RSI + Volume)...")
        alert_info = self.check_sell_alert(df)

        print("📊 Generating market analysis...")
        market_analysis = self.get_market_analysis(df)

        data_for_export = self.prepare_data_for_export(df, 10)

        return {
            'alert_info': alert_info,
            'market_analysis': market_analysis,
            'data': data_for_export,
            'symbol': self.symbol,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def send_alert_notification(self, alert_info: Dict):
        """Placeholder for real notifications (Telegram, Email, etc.)"""
        print("\n🔔🔔 SELL ALERT TRIGGERED!")
        print(f"📌 Symbol: {self.symbol}")
        print(f"💰 Price: {alert_info['current_price']:.4f} THB")
        print(f"📊 SMA({self.bb_period}): {alert_info['sma']:.4f} THB")
        print(f"📈 RSI: {alert_info['rsi']:.2f}")
        print(f"📦 Volume Spike: {alert_info['volume_spike']}")
        print(f"📜 Conditions Met: {alert_info['conditions']}")
        print("-" * 50)

    def monitor_continuously(self, check_interval_minutes: int = 30):
        print(f"🚀 Starting continuous monitoring: {self.symbol}")
        print(f"⏰ Interval: {check_interval_minutes} min")
        print(
            f"🎯 Sell Signal: Price ≤ SMA AND RSI ≥ 70 AND Volume > {self.volume_multiplier}x avg")
        print("-" * 60)

        while True:
            try:
                results = self.run_analysis()
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                if 'error' in results:
                    print(f"[{now}] ❌ Error: {results['error']}")
                else:
                    msg = results['alert_info']['message']
                    print(f"[{now}] {msg}")

                    if results['alert_info']['alert']:
                        self.send_alert_notification(results['alert_info'])

                jitter = np.random.randint(30, 120)
                time.sleep(check_interval_minutes * 60 + jitter)

            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped.")
                break
            except Exception as e:
                print(f"⚠️ Error in loop: {e}")
                time.sleep(60)


# ======================
# 🧪 Example Usage
# ======================
def main():
    alert_system = PriceAlertSystem(
        symbol="KTFIXPLUS-A",
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        volume_multiplier=1.5  # Volume > 1.5x average
    )

    print("🔍 Running single analysis...")
    results = alert_system.run_analysis(days_back=100)

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return

    a = results['alert_info']
    m = results['market_analysis']

    print(f"\n📊 FINAL REPORT: {results['symbol']}")
    print(f"🕒 {results['analysis_timestamp']}")
    print(f"\n🚨 ALERT: {a['message']}")
    print(f"📌 Conditions → Price≤SMA: {a['conditions']['price_le_sma']}, "
          f"RSI≥70: {a['conditions']['rsi_ge_70']}, "
          f"Vol Spike: {a['conditions']['volume_spike']}")
    print(f"\n📈 RSI: {m['rsi_status']} | Trend: {m['trend']}")
    print(f"📦 Volume: {m['current_volume']:,} (Avg: {m['avg_volume']:,.0f}, "
          f"Ratio: {m['volume_ratio']}x)")
    print(f"📉 Bollinger: {m['bb_position']}")

    print(f"\n📋 Recent Data (Last 5 Days):")
    for record in results['data'][-5:]:
        rsi_str = f"{record['rsi']:.2f}" if record['rsi'] else "N/A"
        vol_alert = "💥" if record['volume_spike'] else ""
        print(f"  {record['date']} | {record['close']} | RSI={rsi_str} | "
              f"Vol={record['volume']:,} {vol_alert}")
    print(f"\n🚨 ALERT: {a['message']}")

    # 🔄 Uncomment to start live monitoring
    # alert_system.monitor_continuously(check_interval_minutes=30)


if __name__ == "__main__":
    main()
