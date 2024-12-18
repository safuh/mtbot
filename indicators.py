import numpy as np
import pandas as pd
rsi_thresholds={'oversold_threshold':30,
                'overbought_threshold': 70,
                'stochastic_buy_threshold':20,
                'stochastic_sell_threshold': 80}
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    df['EMA_short'] = df['close'].ewm(span=short_period, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=long_period, adjust=False).mean()
    df['MACD_Line'] = df['EMA_short'] - df['EMA_long']
    df['Signal_Line'] = df['MACD_Line'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']
    return df

def calculate_fibonacci_levels(df, lookback_period=21):
    df['High_Max'] = df['high'].rolling(window=lookback_period).max()
    df['Low_Min'] = df['low'].rolling(window=lookback_period).min()

    df['Fib_23.6'] = df['Low_Min'] + (df['High_Max'] - df['Low_Min']) * 0.236
    df['Fib_38.2'] = df['Low_Min'] + (df['High_Max'] - df['Low_Min']) * 0.382
    df['Fib_50.0'] = df['Low_Min'] + (df['High_Max'] - df['Low_Min']) * 0.5
    df['Fib_61.8'] = df['Low_Min'] + (df['High_Max'] - df['Low_Min']) * 0.618
    df['Fib_78.6'] = df['Low_Min'] + (df['High_Max'] - df['Low_Min']) * 0.786
    return df

# Calculate indicators
def calculate_indicators(df,rsi_period = 14,sma_short_window = 5,sma_long_window = 20,stochastic_period = 5,momentum_period = 1):
    #df['atr']=calculate_atr(df)
    df['SMA_short'] = df['close'].rolling(window=sma_short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=sma_long_window).mean()
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['RSI'] = calculate_rsi(df, rsi_period)
    df['Momentum'] = df['close'].diff(momentum_period)
    df['Stochastic_RSI'] = calculate_stochastic_rsi(df, stochastic_period)
    df = calculate_macd(df)
    
    # Fibonacci Levels
    df = calculate_fibonacci_levels(df)
    return df
def calculate_fibonacci_tp(df, last_price, signal):
    """
    Calculate take-profit levels based on Fibonacci retracement levels.

    Parameters:
    - df (pd.DataFrame): Dataframe with historical data.
    - last_price (float): Entry price.
    - signal (int): 1 for buy, -1 for sell.

    Returns:
    - float: Fibonacci-based TP price.
    """
    high = df['high'].max()
    low = df['low'].min()

    if signal == 1:  # Buy
        tp_price = last_price + (high - low) * 0.618
    elif signal == -1:  # Sell
        tp_price = last_price - (high - low) * 0.618
    else:
        raise ValueError("Invalid signal: must be 1 (buy) or -1 (sell).")

    return tp_price
def calculate_atr(df, period=14):
    """
    Calculates the Average True Range (ATR) indicator.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' columns.
        period (int): Period for ATR calculation.

    Returns:
        pd.Series: ATR values.
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr



def calculate_rsi(df, period):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic_rsi(df, period):
    rsi = calculate_rsi(df, period)
    stoch_rsi = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
    return stoch_rsi * 100
def calculate_position_size(account_balance, risk_percentage, stop_loss_pips, pip_value,symbol,broker):
    risk_amount = account_balance * risk_percentage
    raw_position_size = risk_amount / (stop_loss_pips * pip_value)
    symbol_info=broker.get_symbol_info(symbol)
    # Adjust volume to match broker constraints
    min_volume = symbol_info.volume_min
    max_volume = symbol_info.volume_max
    step_volume = symbol_info.volume_step
    pip_value = broker.get_pip_value(symbol, 1.0)
    # volume within broker's limits
    position_size = max(min_volume, min(raw_position_size, max_volume))
    # Align to the nearest step size
    position_size = round(position_size / step_volume) * step_volume
    if position_size < min_volume or position_size > max_volume:
        raise ValueError(f"Adjusted volume {position_size} is outside allowable range for {symbol}.")
    return position_size
def calculate_atr_tr(df, period=9):
    """
    Calculate the Average True Range (ATR) for the given data.

    Parameters:
    - df (pd.DataFrame): Dataframe with 'high', 'low', and 'close' columns.
    - period (int): Number of periods for ATR calculation (default: 14).

    Returns:
    - float: Latest ATR value.
    """
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    return df['tr'].rolling(window=period).mean()
def calculate_dynamic_sl_tp(df, last_price, signal, atr_multiplier=1.5):
    df=calculate_atr(df)
    atr = df['atr'].iloc[-1]  # Use your indicators to calculate ATR
    if signal == 1:  # Buy signal
        sl_price = last_price - (atr * atr_multiplier)
        tp_price = last_price + (atr * atr_multiplier)
    elif signal == -1:  # Sell signal
        sl_price = last_price + (atr * atr_multiplier)
        tp_price = last_price - (atr * atr_multiplier)
    else:
        raise ValueError("Invalid signal: must be 1 (buy) or -1 (sell).")
    return sl_price, tp_price

def calculate_dynamic_stop_loss(df, atr_multiplier=1.5):
    """
    Calculate the best stop loss in pips using ATR.

    Parameters:
    - df (pd.DataFrame): Dataframe with market data.
    - atr_multiplier (float): Multiplier for ATR to determine stop loss.

    Returns:
    - int: Best stop loss in pips.
    """
    atr = calculate_atr(df)
    stop_loss_pips = atr * atr_multiplier * 10000  # Convert ATR (in price terms) to pips
    return int(round(stop_loss_pips))
