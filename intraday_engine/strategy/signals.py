from dataclasses import dataclass
import math
@dataclass
class Signal:
    symbol:str; event_time:object; side:str; entry_low:float; entry_high:float; stop_loss:float; target1:float; target2:float; risk_reward:float; score:float; confidence:float; reasons:str; setup:str; status:str='PAPER'
def build_signal(symbol,row,score,reasons,side):
    price=float(row.close); a=float(row.atr14)
    if not math.isfinite(price) or not math.isfinite(a) or a<=0:return None
    if side=='LONG':
        stop=min(price-1.2*a,float(row.support) if row.support==row.support else price-1.2*a); risk=price-stop; t1=price+1.5*risk; t2=price+2.5*risk; setup='LONG_BREAKOUT' if row.breakout else 'LONG_TREND'
    else:
        stop=max(price+1.2*a,float(row.resistance) if row.resistance==row.resistance else price+1.2*a); risk=stop-price; t1=price-1.5*risk; t2=price-2.5*risk; setup='SHORT_BREAKDOWN' if row.breakdown else 'SHORT_TREND'
    if risk<=0:return None
    return Signal(symbol,row.timestamp,side,price-.1*a,price+.1*a,stop,t1,t2,1.5,score,score,','.join(reasons),setup)
