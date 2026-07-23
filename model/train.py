"""Train the FPL points model on vaastav's per-GW dataset.
Usage: python model/train.py   (regenerates model/model_export.json)
Retrain weekly in-season: newest merged_gw.csv rows are pulled automatically.
"""
import pandas as pd, numpy as np, json, math, urllib.request
SEASONS=('2023-24','2024-25','2025-26','2026-27')  # 26/27 included once rows exist
BASE='https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data'
frames=[]
for s in SEASONS:
    try:
        df=pd.read_csv(f'{BASE}/{s}/gws/merged_gw.csv',low_memory=False)
        teams=pd.read_csv(f'{BASE}/{s}/teams.csv')
        df['season']=s
        df['opponent']=df['opponent_team'].map(dict(zip(teams['id'],teams['name'])))
        frames.append(df); print(s,len(df),'rows')
    except Exception as e: print(s,'skipped:',e)
d=pd.concat(frames,ignore_index=True)
d['gw']=d['round'].astype(int)
d=d.sort_values(['season','element','gw'])
d['pos']=d['position'].replace({'GKP':'GK','AM':'MID'})
d['price']=d['value']/10
d['gf']=np.where(d['was_home'],d['team_h_score'],d['team_a_score'])
d['ga']=np.where(d['was_home'],d['team_a_score'],d['team_h_score'])
g=d.groupby(['season','element'])
d['form4']=g['total_points'].transform(lambda s:s.shift(1).rolling(4,min_periods=1).mean())
d['mins4']=g['minutes'].transform(lambda s:s.shift(1).rolling(4,min_periods=1).mean())
d['ppg']=g['total_points'].transform(lambda s:s.shift(1).expanding().mean())
d['xgi4']=g['expected_goal_involvements'].transform(lambda s:pd.to_numeric(s,errors='coerce').shift(1).rolling(4,min_periods=1).mean())
d['gws_played']=g.cumcount()
tg=d.groupby(['season','team','gw']).agg(tgf=('gf','max'),tga=('ga','max')).reset_index().sort_values(['season','team','gw'])
tgg=tg.groupby(['season','team'])
tg['team_gf6']=tgg['tgf'].transform(lambda s:s.shift(1).rolling(6,min_periods=2).mean())
tg['team_ga6']=tgg['tga'].transform(lambda s:s.shift(1).rolling(6,min_periods=2).mean())
d=d.merge(tg[['season','team','gw','team_gf6','team_ga6']],on=['season','team','gw'],how='left')
d=d.merge(tg[['season','team','gw','team_gf6','team_ga6']].rename(columns={'team':'opponent','team_gf6':'opp_gf6','team_ga6':'opp_ga6'}),on=['season','opponent','gw'],how='left')
m=d[(d['gws_played']>=2)&(d['mins4']>0)].copy()
m['home']=m['was_home'].astype(int)
for p in ('GK','DEF','MID','FWD'): m['pos_'+p]=(m['pos']==p).astype(int)
FEATS=['form4','mins4','ppg','xgi4','price','home','team_gf6','team_ga6','opp_gf6','opp_ga6','pos_GK','pos_DEF','pos_MID','pos_FWD']
m=m.dropna(subset=FEATS+['total_points'])
seasons_sorted=sorted(m['season'].unique())
last=seasons_sorted[-1]; cut=int(m[m['season']==last]['gw'].max())-12
train=m[~((m['season']==last)&(m['gw']>=cut))]; test=m[(m['season']==last)&(m['gw']>=cut)]
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
ridge=Ridge(alpha=3.0).fit(train[FEATS],train['total_points'])
pred=ridge.predict(test[FEATS])
rmse=math.sqrt(mean_squared_error(test['total_points'],pred))
mae=mean_absolute_error(test['total_points'],pred); r2=r2_score(test['total_points'],pred)
print(f'holdout ({last} GW{cut}+, n={len(test)}): RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}')
print('perfect-model band: RMSE 2.7-2.9, MAE 1.9-2.0, R2 0.12-0.17')
json.dump({'feats':FEATS,'coef':[round(float(c),6) for c in ridge.coef_],
           'intercept':round(float(ridge.intercept_),6),
           'holdout':{'rmse':round(rmse,3),'mae':round(mae,3),'r2':round(r2,3),'n':len(test)}},
          open(__file__.replace('train.py','model_export.json'),'w'),indent=1)
print('exported model_export.json')
