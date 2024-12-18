import time
import logging
from brokerapi import *
from indicators import rsi_thresholds,calculate_fibonacci_tp
import MetaTrader5 as mt5 # type: ignore
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

timeframe='M15'
symbols=['XAUUSD','UKOILRoll']
forex_symbols = [ 'USDCHF','EURGBP', 'GBPUSD','USDCAD','EURUSD','AUDUSD']
#broker=get_broker()
from TraderExe import plot_indicators
import threading
global_signals={}
data_lock=threading.Lock()
trade_lock=threading.Lock()

def positions_closer():
    global trade_count
    while trade_count > 0 or len(get_open_positions())>0:close_positions()
def fetch_and_trade(symbol):
    global trade_count
    while trade_count>=1:# and len(get_symbol_positions(symbol=symbol))<1:
        with trade_lock:
            if trade_count <= 0:break
        #try:
            df=apply_fib_strategy(fetch_forex_data(symbol,timeframe),symbol,**rsi_thresholds)
            latest_signal = df['signal'].iloc[-1]
            with data_lock:global_signals[symbol]=df
            logging.info(f"Latest signal for {symbol}: {latest_signal}")
            #logging.info(f"Latest open positions for {symbol}: {len(get_symbol_positions(symbol=symbol))}")
            if latest_signal != 0:
                with trade_lock:
                    if trade_count>0:
                        logging.info(f"Latest signal for {symbol}: {latest_signal}")
                        execute_trade(symbol, latest_signal,stop_loss_percent=0.004,tp_price=0.008)    
                        trade_count -=1
                        break
                    else:break
        #except Exception as e:logging.info(f'Exception at trader thread:->{str(e)}')
import os
from pathlib import Path
threads=[]
base_dir= Path(__file__).resolve().parent
def main():
    global threads
    global global_signals
    for symbol in forex_symbols:
        thread = threading.Thread(target=fetch_and_trade, args=(symbol,))
        threads.append(thread)
        thread.start()
    for thread in threads:thread.join()
    for key,df in global_signals.items():
        try:plot_indicators(df,os.path.join(base_dir,f'symbols\\{df['signal'].iloc[-1]}\\{key}.png'))
        except Exception as e:logging.info(f'plotting exception {str(e)}')
#if symbol == 'USOILRoll'or symbol=='UKOILRoll'or symbol=='XAUUSD':#or symbol=='EURUSD'
#server="EGMSecurities-Demo",login = 531729,password = "Airs-Gown-93"
trade_count=1
if __name__ =="__main__":
    try:
        initialize(server="EGMSecurities-Demo",login = 531729,password = "Airs-Gown-93")
        pos_thred=threading.Thread(target=positions_closer,name=f"open orders")    
        pos_thred.start()
        main()
    except Exception as e:logging.info(f"Main func exception -> {str(e)}")
    finally:pos_thred.join()
        #    logging.info(f"{thread.name} thread <-> {thread.is_alive()}")
            #posi_thrd()
        #shutdown()
#        if len(broker.closed_positions)>0:
#            logging.info(f"closed orders {broker.closed_positions}")
#        else:
#            if broker.get_open_positions() != None or len(broker.get_open_positions())>=1:
#                broker.close_positions()   