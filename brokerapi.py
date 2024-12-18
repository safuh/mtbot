import MetaTrader5 as mt5 # type: ignore
import logging
import time
import pandas as pd
from indicators import rsi_thresholds
from TraderExe import apply_fib_strategy
import threading
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize(server="EGMSecurities-Live",login = 9224868,password = "RozzL3!4."):
    if not mt5.initialize():
        error = mt5.last_error()
        raise ConnectionError(f"Failed to initialize MetaTrader5. Error: {error}")
    else:
        print(mt5.terminal_info())
        print(mt5.version())
        auth=mt5.login(login, password=password,server=server)
        if auth:
            #print(mt5.account_info())
            account_info_dict = mt5.account_info()._asdict()
            print("Trade allowed={}".format(account_info_dict['trade_allowed']))
        else:
            print("failed to connect at account #{}, error code: {}".format(login, mt5.last_error()))
        logging.info("MetaTrader5 initialized successfully.")
    
def shutdown():
    """
    Shutdown the MetaTrader5 connection.
    """
    mt5.shutdown()
    logging.info("MetaTrader5 connection closed.")

def get_ticker(symbol):
    """
    Get the current bid and ask prices for a Forex symbol.
    Parameters:
        symbol (str): The Forex symbol (e.g., 'EURUSD').
    Returns:
        dict: Contains 'bid' and 'ask' prices.
    """
    selected=mt5.symbol_select(symbol,True)
    if not selected:
        print("Failed to select symbol")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        error = mt5.last_error()
        raise ValueError(f"Failed to retrieve ticker for {symbol}. Error: {error}")
    return {"bid": tick.bid, "ask": tick.ask}
def get_symbol_info(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        raise ValueError(f"Symbol {symbol} information is unavailable.")
    return symbol_info
def get_symbol_positions(symbol=''):
    if not symbol:return []
    else:return mt5.positions_get(symbol=symbol)    
    
def get_open_positions():
    """Fetch all open positions."""
    positions = mt5.positions_get()
    open_pos=[]
    if positions is None:
        print("No open positions:", mt5.last_error())
        return []
    for position in positions:
        open_pos.append(position)
    open_pos
    return open_pos
def update_order_stop_loss(position, new_stop_loss):
    """
    Modify an existing position to set a new stop-loss value.
    """
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "symbol": position.symbol,
        "sl": new_stop_loss,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
        #"tp": position.tp,  # Keep the same take-profit level
        "deviation":1
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Failed to update stop-loss for position {position.ticket}: {result.retcode}")
    else:
        logging.info(f"Updated stop-loss for position {position.ticket} to {new_stop_loss}")
    return result
closed_positions_list=[]
def close_thrd(order,profit_trigger,trailing_stop_percent):
    #global closed_positions_list
    symbol = order.symbol
    #quantity = float(order.volume)
    current_price = float(order.price_current)
    logging.info(f"Current price for {symbol}: {current_price}")
    logging.info(f"Order ID: {order.ticket}, Profit/Loss: {order.profit}")
    if order.profit >=profit_trigger:
        if order.type == mt5.ORDER_TYPE_BUY:
            new_stop_loss = current_price * (1 - trailing_stop_percent)
            update_order_stop_loss(order, new_stop_loss)
        else:
            new_stop_loss = current_price * (1 + trailing_stop_percent)
            update_order_stop_loss(order, new_stop_loss)
        logging.info(f"Updating order {order.ticket} due to profit.")
    #if order.profit <= -1.89:
    #    stop_trail=trailing_stop_percent+0.00003
    #    if order.type==mt5.ORDER_TYPE_BUY:
    #        new_stop_loss= current_price *(1-stop_trail)
    #        update_order_stop_loss(order,new_stop_loss)
    #    else:
    #        new_stop_loss = current_price * (1 + stop_trail)
    #        update_order_stop_loss(order, new_stop_loss)
    #    logging.info(f"Updating order {order.ticket} due to profit.")
def close_positions(stop_percent=0.00125,take_percent=0.005,trailing_stop_percent=0.00002, profit_trigger=0.102):
    positions=get_open_positions()
    global closed_positions_list
    for order in positions.copy():
        try:
            thread_close=threading.Thread(target=close_thrd,args=(order,profit_trigger,trailing_stop_percent),name=f'{order.symbol}')
            thread_close.start()
            logging.info(f"{thread_close.name} thread <-> {thread_close.is_alive()}")
            #close_thrd(order,profit_trigger,trailing_stop_percent)
        except Exception as e:
            logging.info(f'closing positions error{str(e)}')
    return closed_positions_list
def get_pos_type(position):
    typ=mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
    return typ
def get_current_price(symbol,order_type):
    return mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
def get_order_current_price(position):
    return mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask
def close_order(position):
    """Close an open order."""
    symbol = position.symbol
    ticket = position.ticket
    lot = position.volume
    curr_price=position.price_current
    action_type = mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
    # Get the current price for the symbol
    price = mt5.symbol_info_tick(symbol).ask if action_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).bid
    # Create a close request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": action_type,
        "position":ticket,
        "price": curr_price,
        "deviation": 1,
        "magic": position.magic,  # Optional: Use an identifier for the order
        "type_filling": mt5.ORDER_FILLING_IOC,
        "comment": "Closed by bot",
    }
    # Send the request
    print(request)
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"close order {ticket}: {result}", result.retcode)
    else:
        print(f"Closing Order {ticket} closed successfully") 
        logging.info(f"result:{result}")
    return result

def create_market_order(symbol, action, volume,sl,tp):
    """
    Place a market order (buy or sell).

    Parameters:
        symbol (str): The Forex symbol (e.g., 'EURUSD').
        action (str): 'buy' or 'sell'.
        volume (float): The volume of the order.

    Returns:
        dict: Result of the order execution.
    """
    if action not in ['buy', 'sell']:
        raise ValueError("Action must be 'buy' or 'sell'.")
    selected=mt5.symbol_select(symbol,True)
    if not selected:
        print("Failed to select symbol")
    symbol_info = get_symbol_info(symbol)
    stop_level =  symbol_info.trade_stops_level * symbol_info.point if symbol_info.trade_stops_level else 0
    order_type = mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if action == 'buy' else mt5.symbol_info_tick(symbol).bid
    if sl and abs(price - sl) < stop_level:
        logging.info(f"Invalid SL:->{sl} Must be at least {stop_level} away from price.")
    if tp and abs(tp - price) < stop_level:
        logging.info(f"Invalid TP:->{tp} Must be at least {stop_level} away from price.")
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        #"sl":sl,
        #"tp":tp,
        "deviation": 1,
        "magic": 234000,
        "comment": f"Auto-trade {action}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    logging.info(f"Order request: {request}")
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        error = mt5.last_error()
        logging.info(f"Order failed. Retcode: {result.retcode}, Error: {error}")

    logging.info(f"Order executed successfully: {result}")
    return result
def fetch_forex_data(symbol, timeframe,n_bars=1000):
    timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
    }
    # Check if the timeframe is valid
    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, n_bars)
    if rates is None:
        raise ValueError(f"Failed to retrieve data for symbol: {symbol}. Error: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert timestamp to datetime
    df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        })
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]    
    return df
def execute_trade(symbol, signal,stop_loss_percent,tp_price):
    try:
        ticker = get_ticker(symbol)
        last_price = ticker['bid'] if signal == -1 else ticker['ask']
        pointer = get_symbol_info(symbol)
        point=pointer.point
        #position_size = calculate_position_size(account_balance, risk_percentage, stop_loss_pips, pip_value,symbol)
        position_size=0.01
        #sl_price, _ = calculate_dynamic_sl_tp(self.df, last_price, signal)
        #tp_price = calculate_fibonacci_tp(self.df, last_price, signal)
        #sl_price, tp_price = calculate_dynamic_sl_tp(self.df, last_price, signal, atr_multiplier=1.5)
        if signal == 1:
            sl_price = last_price * (1 - stop_loss_percent)
            #tp_price = last_price * (1 + take_profit_percent)
            #sl_price = last_price - (stop_loss_pips * point)
            #tp_price = last_price + (stop_loss_pips * take_profit_ratio * point)
            logging.info(f"Placing Buy order: {symbol}, Size: {position_size}")
            create_market_order(symbol, 'buy', position_size,sl_price,tp_price)
            #exit_price = last_price + (take_profit_pips * pip_value)
            #trade_profit = (exit_price - last_price) * position_size
            #logging.info(f"Simulated Buy: Entry={last_price}, Exit={exit_price}, Profit=${trade_profit:.2f}")
        elif signal == -1:
            sl_price = last_price * (1 + stop_loss_percent)
            #tp_price = last_price * (1 - take_profit_percent)
            #sl_price = last_price + (stop_loss_pips * point)
            #tp_price = last_price - (stop_loss_pips * take_profit_ratio * point)
            logging.info(f"Placing Sell order: {symbol}, Size: {position_size}")
            create_market_order(symbol, 'sell', position_size,sl_price,tp_price)
            #exit_price = last_price - (take_profit_pips * pip_value)
            #trade_profit = (last_price - exit_price) * position_size
            #logging.info(f"Simulated Sell: Entry={last_price}, Exit={exit_price}, Profit=${trade_profit:.2f}")
        #total_profit += trade_profit
        #trade_count += 1
        #if trade_profit > 0:
        #    winning_trades += 1
        #else:
        #    losing_trades += 1
    except Exception as e:
        logging.error(f"Trade execution failed: {e}")
def exec_counter_trade(symbol, signal,stop_loss_percent,tp):
    try:
        ticker = get_ticker(symbol)
        last_price = ticker['bid'] if signal == -1 else ticker['ask']
        pointer = get_symbol_info(symbol)
        point=pointer.point
        position_size=0.01
        if signal == 1:
            sl_price = last_price * (1 - stop_loss_percent)
            logging.info(f"Placing Buy order: {symbol}, Size: {position_size}")
            create_market_order(symbol, 'buy', position_size,sl_price,tp)
        elif signal == -1:
            sl_price = last_price * (1 + stop_loss_percent)
            logging.info(f"Placing Sell order: {symbol}, Size: {position_size}")
            create_market_order(symbol, 'sell', position_size,sl_price,tp)
    except Exception as e:
        logging.error(f"Trade execution failed: {e}")
def execute_trade_with_fibonacci(symbol, signal, stop_loss_percent, take_profit_percent):
    try:
        ticker = get_ticker(symbol)
        last_price = ticker['bid'] if signal == -1 else ticker['ask']
        fib_result = apply_fib_strategy(
            fetch_forex_data(symbol, timeframe="H1"), **rsi_thresholds
        )
        point = get_symbol_info(symbol).point
        position_size = 0.01  # Adjust to your needs
        
        # Calculate Fibonacci levels
        high = fib_result['high'].max()
        low = fib_result['low'].min()
        fib_38 = low + 0.382 * (high - low)
        fib_61 = low + 0.618 * (high - low)

        # Set SL/TP based on Fibonacci
        if signal == 1:  # Buy
            sl_price = max(last_price * (1 - stop_loss_percent), fib_38)
            tp_price = min(last_price * (1 + take_profit_percent), fib_61)
            create_market_order(symbol, 'buy', position_size, sl_price, tp_price)
        elif signal == -1:  # Sell
            sl_price = min(last_price * (1 + stop_loss_percent), fib_61)
            tp_price = max(last_price * (1 - take_profit_percent), fib_38)
            create_market_order(symbol, 'sell', position_size, sl_price, tp_price)
    except Exception as e:
        logging.error(f"Trade execution with Fibonacci failed: {e}")
