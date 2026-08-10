from dataclasses import dataclass
@dataclass
class Position:
    symbol:str; side:str; quantity:int; entry:float; stop:float; target:float
class PaperBroker:
    def __init__(self,capital,risk_per_trade=.005): self.capital=capital; self.risk_per_trade=risk_per_trade; self.positions={}
    def size(self,entry,stop):
        risk=abs(entry-stop); return int(self.capital*self.risk_per_trade/risk) if risk else 0
    def open(self,symbol,side,entry,stop,target):
        q=self.size(entry,stop)
        if q<=0 or symbol in self.positions:return None
        p=Position(symbol,side,q,entry,stop,target); self.positions[symbol]=p; return p
    def mark(self,symbol,price):
        p=self.positions.get(symbol)
        if not p:return None
        hit=price<=p.stop if p.side=='LONG' else price>=p.stop; target=price>=p.target if p.side=='LONG' else price<=p.target
        if hit or target:
            pnl=(price-p.entry)*p.quantity*(1 if p.side=='LONG' else -1); self.capital+=pnl; del self.positions[symbol]; return {'symbol':symbol,'exit_price':price,'pnl':pnl,'reason':'TARGET' if target else 'STOP'}
