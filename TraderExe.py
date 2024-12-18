import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from indicators import calculate_indicators,calculate_dynamic_sl_tp,calculate_fibonacci_tp,calculate_atr
import matplotlib.pyplot as plt
import pandas as pd

def plot_indicators(df,save_path,num_samples=100):
    """
    Plots all the key indicators for the trading bot: RSI, SMA, Momentum, and Stochastic RSI.

    Parameters:
        df (pd.DataFrame): DataFrame containing price and indicator columns.
            Expected columns: ['timestamp', 'close', 'RSI', 'SMA_short', 'SMA_long', 'Momentum', 'Stochastic_RSI']
    """
    # Set up the figure
    fig, axs = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
    df=df.iloc[-num_samples:]
    #recent_high = df['high'].max()
    #recent_low = df['low'].min()
    #fib_38 = recent_low + 0.382 * (recent_high - recent_low)
    #fib_61 = recent_low + 0.618 * (recent_high - recent_low)
    latest_rsi = df['RSI'].iloc[-1]
    latest_stochastic = df['Stochastic_RSI'].iloc[-1]
    title = f"Trading Indicators | RSI: {latest_rsi}, Stochastic RSI: {latest_stochastic}"
    fig.suptitle(title, fontsize=16)
    # Plot Close Price and SMA
    axs[0].plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    axs[0].plot(df['timestamp'], df['SMA_short'], label='SMA Short', color='orange', linestyle='--')
    axs[0].plot(df['timestamp'], df['SMA_long'], label='SMA Long', color='green', linestyle='--')
    #axs[0].axhline(y=fib_38, color='purple', linestyle='--', label='Fib 38.2%')
    #axs[0].axhline(y=fib_61, color='brown', linestyle='--', label='Fib 61.8%')
    axs[0].set_title('Price and Moving Averages')
    axs[0].legend()
    axs[0].set_ylabel('Price')
    #buy_signals = df[df['signal'] == 1]
    #sell_signals = df[df['signal'] == -1]
    #axs[0].scatter(buy_signals['timestamp'], buy_signals['close'], label='Buy Signal', marker='^', color='green', alpha=1)
    #axs[0].scatter(sell_signals['timestamp'], sell_signals['close'], label='Sell Signal', marker='v', color='red', alpha=1)


    # Plot RSI
    axs[1].plot(df['timestamp'], df['RSI'], label='RSI', color='purple')
    axs[1].axhline(y=70, color='red', linestyle='--', label='Overbought')
    axs[1].axhline(y=30, color='green', linestyle='--', label='Oversold')
    axs[1].set_title(f'Relative Strength Index (RSI) {latest_rsi:.2f}')
    axs[1].legend()
    axs[1].set_ylabel('RSI')

    # Plot Stochastic RSI
    axs[2].plot(df['timestamp'], df['Stochastic_RSI'], label='Stochastic RSI', color='magenta')
    axs[2].axhline(y=80, color='red', linestyle='--', label='Overbought')
    axs[2].axhline(y=20, color='green', linestyle='--', label='Oversold')
    axs[2].set_title(f'Stochastic RSI {latest_stochastic}')
    axs[2].legend()
    axs[2].set_ylabel('Stochastic RSI')
    axs[2].set_xlabel('Timestamp')

    # Improve layout
#    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=300)
        logging.info(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
import pandas as pd
import numpy as np
from indicators import calculate_atr

def trading_strategy_with_atr(df, rsi_period=14, stochastic_period=5, macd_params=(12, 26, 9), 
                              atr_period=14, breakout_window=20, atr_multiplier=2.0):
    """
    Trading strategy integrating ATR breakout filters and ATR bands.

    Parameters:
        df (pd.DataFrame): OHLC data with 'high', 'low', 'close'.
        rsi_period (int): RSI period.
        stochastic_period (int): Stochastic RSI period.
        macd_params (tuple): MACD parameters (short, long, signal).
        atr_period (int): ATR period.
        breakout_window (int): Window for breakout levels.
        atr_multiplier (float): Multiplier for ATR bands.

    Returns:
        pd.DataFrame: Updated DataFrame with signals.
    """
    from indicators import calculate_rsi, calculate_stochastic_rsi, calculate_macd

    # Calculate indicators
    df['RSI'] = calculate_rsi(df, rsi_period)
    df['Stochastic_RSI'] = calculate_stochastic_rsi(df, stochastic_period)
    df = calculate_macd(df, *macd_params)
    df = add_atr_bands(df, atr_period, atr_multiplier)
    df = add_atr_breakout_filter(df, atr_period, breakout_window, atr_multiplier)

    # Combine signals
    df['signal'] = np.where(
        (df['RSI'] < 30) & (df['close'] < df['ATR_Lower_Band']) & (df['MACD_Line'] > df['Signal_Line']), 1,
        np.where(
            (df['RSI'] > 70) & (df['close'] > df['ATR_Upper_Band']) & (df['MACD_Line'] < df['Signal_Line']), -1, 0
        )
    )

    return df


def add_trailing_stop(df, atr_period=7, atr_multiplier=1.0):
    df['Trailing_Stop_Buy'] = df['high'] - (df['ATR'] * atr_multiplier)
    df['Trailing_Stop_Sell'] = df['low'] + (df['ATR'] * atr_multiplier)
    return df
def add_atr_bands(df, atr_period=7, atr_multiplier=1.5):
    df['ATR_Upper_Band'] = df['close'] + (atr_multiplier * df['ATR'])
    df['ATR_Lower_Band'] = df['close'] - (atr_multiplier * df['ATR'])
    return df

def apply_fib_strategy(df,symbol,oversold_threshold,overbought_threshold,stochastic_sell_threshold,stochastic_buy_threshold):
    df = calculate_indicators(df)
    df['ATR'] = calculate_atr(df)
    
    df = df.dropna(subset=['RSI', 'Momentum', 'SMA_short', 'SMA_long', 'Stochastic_RSI']).copy()
    #df['atr'] = df['atr'].fillna(0)
    
    
    if symbol =='USDCHF':window=12
    else:window=24
    recent_high = df['high'].rolling(window=20).max()
    recent_low = df['low'].rolling(window=20).min()
    fib_23 = recent_low + 0.236 * (recent_high - recent_low)
    fib_38 = recent_low + 0.382 * (recent_high - recent_low)
    fib_50 = recent_low + 0.5 * (recent_high - recent_low)
    fib_61 = recent_low + 0.618 * (recent_high - recent_low)
    fib_76 = recent_low + 0.764 * (recent_high - recent_low)

    #df['avg_volume'] = df['volume'].rolling(window=window).mean()
    #df['avg_volume'] = df['avg_volume'].bfill()
    #df['volume_filter'] = np.where(df['volume'] > df['avg_volume'], 1, 0)
    #assert df['avg_volume'].isna().sum() == 0, "NaN values found in avg_volume"
    #assert df['volume_filter'].isna().sum() == 0, "NaN values found in volume_filter"
    #logging.info(f'last vol avg{df['avg_volume'].iloc[-1]}vol filter {df['volume_filter'].iloc[-1]}')
    df = add_atr_bands(df, atr_period=9, atr_multiplier=1.5)
    #df = add_atr_breakout_filter(df, atr_period=9, breakout_window=50, atr_multiplier=1.5)

    # Combine signals
    
    # Allow signals only if ATR is rising
    df = df.dropna().reset_index(drop=True)
    logging.info(f"After dropping NaN values, DataFrame shape: {df.shape}")
    df['buy_signal'] = np.where(
        #(df['close'] > df['EMA5']) & (df['EMA5'] > df['EMA10'])&
    #    (df['volume_filter'] == 1)&
        #(df['Momentum'] > 0) &
        (df['RSI'] < 30) &
         (df['close'] < df['ATR_Lower_Band']) &
        #(df['SMA_short'] > df['SMA_long']) &
        (df['Stochastic_RSI'] < 20)&
        (df['MACD_Line'] > df['Signal_Line']),
        #(df['breakout_filter'] == 1),
        #((df['close'] > fib_38) | (df['close'] > fib_23))  # Include 23.6% level
        1, 0
    )
    df['sell_signal'] = np.where(
        #(df['close'] < df['EMA5']) & (df['EMA5'] < df['EMA10']) &
    #    (df['volume_filter'] == 1)&
        #(df['Momentum'] < 0) &
        (df['RSI'] > 70) &
        (df['close'] > df['ATR_Upper_Band']) &
        #(df['breakout_filter'] == 1)&
        #(df['SMA_short'] < df['SMA_long']) &
        (df['MACD_Line'] < df['Signal_Line'])&
        (df['Stochastic_RSI'] > 80) ,
        #((df['close'] < fib_61) | (df['close'] < fib_76)) # Include 76.4% level
    -1, 0
    )
    df['signal'] = df['buy_signal'] + df['sell_signal']
    #df = add_trailing_stop(df, atr_period=7, atr_multiplier=1.0)
    return df

def add_atr_breakout_filter(df, atr_period=7, breakout_window=10, atr_multiplier=1.5):
    df['Recent_High'] = df['high'].rolling(window=breakout_window).max()
    df['Recent_Low'] = df['low'].rolling(window=breakout_window).min()
    df['Upper_Breakout'] = df['Recent_High'] + (df['ATR'] * atr_multiplier)
    df['Lower_Breakout'] = df['Recent_Low'] - (df['ATR'] * atr_multiplier)

    # Filter signals
    df['breakout_filter'] = np.where(
        (df['close'] > df['Upper_Breakout']) | (df['close'] < df['Lower_Breakout']), 1, 0
    )
    return df

def add_candlestick_filter(df):
    df['bullish_engulfing'] = np.where(
        (df['close'] > df['open']) &  # Current candle is bullish
        (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
        (df['close'] > df['open'].shift(1)) &  # Current close > previous open
        (df['open'] < df['close'].shift(1)), 1, 0  # Current open < previous close
    )

    # Identify bearish engulfing pattern
    df['bearish_engulfing'] = np.where(
        (df['close'] < df['open']) &  # Current candle is bearish
        (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
        (df['close'] < df['open'].shift(1)) &  # Current close < previous open
        (df['open'] > df['close'].shift(1)), 1, 0  # Current open > previous close
    )

    df['signal'] = np.where(
        (df['signal'] == 1) & (df['bullish_engulfing'] == 1), 1,
        np.where((df['signal'] == -1) & (df['bearish_engulfing'] == 1), -1, 0)
    )

    return df
def add_volatility_volume_filter(df, atr_multiplier=1.5, min_volume_threshold=100000):
    # Calculate ATR
    df['ATR'] = calculate_atr(df)

    # Volatility filter: ensure the current range exceeds ATR
    df['volatility_filter'] = np.where(
        (df['high'] - df['low']) > (atr_multiplier * df['ATR']), 1, 0
    )

    # Volume filter: ensure volume exceeds the threshold
    df['volume_filter'] = np.where(df['volume'] > min_volume_threshold, 1, 0)

    # Combine filters
    df['signal'] = np.where(
        (df['signal'] != 0) & (df['volatility_filter'] == 1) & (df['volume_filter'] == 1),
        df['signal'], 0
    )

    return df
