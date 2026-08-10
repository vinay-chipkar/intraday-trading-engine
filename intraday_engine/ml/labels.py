def label_forward_path(df,entry,stop,target,horizon=30):
    future=df.iloc[:horizon]; th=sh=None
    for i,r in future.iterrows():
        if r.high>=target: th=i; break
        if r.low<=stop: sh=i; break
    first=th is not None and (sh is None or th<=sh)
    mfe=((future.high.max()-entry)/entry*100) if not future.empty else 0
    mae=((future.low.min()-entry)/entry*100) if not future.empty else 0
    return {'target_hit_first':bool(first),'mfe_pct':float(mfe),'mae_pct':float(mae),'label':int(first)}
