from pathlib import Path
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
FEATURES=['relative_volume','rsi14','ema9','ema20','ema50','atr14','distance_to_support_pct','distance_to_resistance_pct','feature_score']
def train(df,model_path='models/trade_probability.joblib'):
    d=df.dropna(subset=FEATURES+['label']).copy()
    if d.label.nunique()<2: raise ValueError('Training data must contain both label classes')
    split=int(len(d)*.8); x=d[FEATURES]; y=d.label; m=HistGradientBoostingClassifier(max_depth=4,learning_rate=.05,max_iter=200,random_state=42); m.fit(x.iloc[:split],y.iloc[:split]);
    metrics={'test_rows':len(x)-split}
    if y.iloc[split:].nunique()>1: metrics['roc_auc']=float(roc_auc_score(y.iloc[split:],m.predict_proba(x.iloc[split:])[:,1]))
    Path(model_path).parent.mkdir(parents=True,exist_ok=True); joblib.dump({'model':m,'features':FEATURES},model_path); return metrics
