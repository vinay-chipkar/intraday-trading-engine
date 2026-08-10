import duckdb
import pandas as pd
from config import settings
SCHEMA='''
CREATE TABLE IF NOT EXISTS market_context(captured_at TIMESTAMPTZ,trading_date DATE,gift_nifty_change_pct DOUBLE,dow_change_pct DOUBLE,sp500_change_pct DOUBLE,nasdaq_change_pct DOUBLE,india_vix DOUBLE,usd_inr DOUBLE,brent DOUBLE,fii_flow DOUBLE,dii_flow DOUBLE,nifty_change_pct DOUBLE,banknifty_change_pct DOUBLE,news_count INTEGER,high_impact_news_count INTEGER,score DOUBLE,regime VARCHAR);
CREATE TABLE IF NOT EXISTS instrument_master(symbol VARCHAR PRIMARY KEY,instrument_key VARCHAR,name VARCHAR,trading_symbol VARCHAR,updated_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS candidate_events(event_time TIMESTAMPTZ,trading_date DATE,symbol VARCHAR,instrument_key VARCHAR,ltp DOUBLE,volume DOUBLE,relative_volume DOUBLE,price_change_pct DOUBLE,vwap DOUBLE,candidate_score DOUBLE,reason VARCHAR);
CREATE TABLE IF NOT EXISTS feature_snapshots(event_time TIMESTAMPTZ,trading_date DATE,symbol VARCHAR,instrument_key VARCHAR,timeframe VARCHAR,close DOUBLE,volume DOUBLE,relative_volume DOUBLE,vwap DOUBLE,rsi14 DOUBLE,ema9 DOUBLE,ema20 DOUBLE,ema50 DOUBLE,ema200 DOUBLE,atr14 DOUBLE,support DOUBLE,resistance DOUBLE,distance_to_support_pct DOUBLE,distance_to_resistance_pct DOUBLE,candle_pattern VARCHAR,trend VARCHAR,breakout BOOLEAN,breakdown BOOLEAN,feature_score DOUBLE,feature_json VARCHAR);
CREATE TABLE IF NOT EXISTS signals(signal_id BIGINT,event_time TIMESTAMPTZ,trading_date DATE,symbol VARCHAR,side VARCHAR,entry_low DOUBLE,entry_high DOUBLE,stop_loss DOUBLE,target1 DOUBLE,target2 DOUBLE,risk_reward DOUBLE,score DOUBLE,confidence DOUBLE,setup VARCHAR,reasons VARCHAR,status VARCHAR);
CREATE TABLE IF NOT EXISTS training_labels(event_time TIMESTAMPTZ,symbol VARCHAR,horizon_minutes INTEGER,entry_price DOUBLE,target_price DOUBLE,stop_price DOUBLE,target_hit_first BOOLEAN,max_favorable_excursion DOUBLE,max_adverse_excursion DOUBLE,label INTEGER);
'''
def conn():
    c=duckdb.connect(settings.duckdb_path); c.execute(SCHEMA); return c
def insert_df(table,df):
    if df is None or df.empty:return
    c=conn(); c.register('incoming_df',df); c.execute(f'INSERT INTO {table} SELECT * FROM incoming_df'); c.unregister('incoming_df'); c.close()
def upsert_instruments(df):
    if df.empty:return
    c=conn(); c.register('incoming_df',df); c.execute('INSERT OR REPLACE INTO instrument_master SELECT * FROM incoming_df'); c.unregister('incoming_df'); c.close()
def latest_market_score():
    c=conn(); r=c.execute('SELECT score FROM market_context ORDER BY captured_at DESC LIMIT 1').fetchone(); c.close(); return float(r[0]) if r else 0.0
