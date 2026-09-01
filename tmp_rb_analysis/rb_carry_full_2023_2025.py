from __future__ import annotations
import json, math, os, re, unicodedata, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import fisher_exact, chi2_contingency, beta, norm

warnings.filterwarnings('ignore')
OUT=Path('rb_carry_2023_2025_full'); OUT.mkdir(exist_ok=True)
SEASONS=[2023,2024,2025]
PROP_URL='https://raw.githubusercontent.com/gcampb41/nfl_data-/main/data/processed/football/nfl/player_props/{season}.parquet'
STATS_URL='https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv'
GAMES_URL='https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
BOOKS={15:'CONSENSUS',68:'DK',69:'FD',79:'BET365'}
REAL_BOOKS=[68,69,79]


def dl(url,path):
    path=Path(path)
    if path.exists() and path.stat().st_size>1000: return path
    r=requests.get(url,timeout=120,headers={'User-Agent':'Mozilla/5.0'})
    r.raise_for_status(); path.write_bytes(r.content); return path

def norm_name(s):
    s=unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode().lower()
    s=re.sub(r"[^a-z0-9 ]+",' ',s); s=re.sub(r'\s+',' ',s).strip()
    toks=[x for x in s.split() if x not in {'jr','sr','ii','iii','iv','v'}]
    return ' '.join(toks)

def amer_imp(o):
    try:o=float(o)
    except:return np.nan
    if not np.isfinite(o) or o==0:return np.nan
    if o>0:return 100/(o+100)
    return -o/(-o+100)

def amer_dec(o):
    try:o=float(o)
    except:return np.nan
    if not np.isfinite(o) or o==0:return np.nan
    return 1+o/100 if o>0 else 1+100/(-o)

def fair_american(p):
    if not (p>0 and p<1):return np.nan
    return (1/p-1)*100 if p<=.5 else -100*p/(1-p)

def wilson(k,n,z=1.959963984540054):
    if n==0:return (np.nan,np.nan)
    p=k/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return ctr-half,ctr+half

def bh_adjust(pvals):
    p=np.asarray(pvals,dtype=float); out=np.full(len(p),np.nan)
    ix=np.where(np.isfinite(p))[0]
    if not len(ix):return out
    s=ix[np.argsort(p[ix])]; m=len(s); vals=p[s]*m/np.arange(1,m+1)
    vals=np.minimum.accumulate(vals[::-1])[::-1]; out[s]=np.minimum(vals,1); return out

def normalize_side(x):
    s=str(x).strip().lower()
    if s in {'over','o','1'} or 'over' in s:return 'O'
    if s in {'under','u','2'} or 'under' in s:return 'U'
    return np.nan

def pick_col(df,cands):
    for c in cands:
        if c in df.columns:return c
    return None

# ---------- LOAD ----------
props=[]; audits=[]
for season in SEASONS:
    p=dl(PROP_URL.format(season=season),f'props_{season}.parquet')
    d=pd.read_parquet(p)
    d['season']=pd.to_numeric(d.get('season',season),errors='coerce').fillna(season).astype(int)
    d['book_id']=pd.to_numeric(d['book_id'],errors='coerce')
    d['value']=pd.to_numeric(d['value'],errors='coerce')
    d['odds']=pd.to_numeric(d['odds'],errors='coerce')
    d['week']=pd.to_numeric(d['week'],errors='coerce')
    d['side_norm']=d['side'].map(normalize_side)
    audits.append({'season':season,'raw_rows':len(d),'weeks':','.join(map(str,sorted(d.week.dropna().unique().astype(int)))),'rushing_attempt_rows':int((d.bet_type=='rushing_attempts').sum()),'rushing_yard_rows':int((d.bet_type=='rushing_yards').sum()),'books':json.dumps(d.book_id.value_counts().head(10).to_dict())})
    props.append(d)
props=pd.concat(props,ignore_index=True)
pd.DataFrame(audits).to_csv(OUT/'source_season_audit.csv',index=False)

# Raw schema/distributions audit
pd.DataFrame({'column':props.columns}).to_csv(OUT/'source_columns.csv',index=False)
for c in ['bet_type','period','line_type','position','position_group','book_id','side_norm']:
    if c in props.columns:
        props[c].value_counts(dropna=False).head(100).rename('count').to_csv(OUT/f'audit_{c}.csv')

# Only the two ordinary full-game markets. The archive's ingestion de-dupes latest quote per book using keys that exclude value.
p=props[props.bet_type.isin(['rushing_yards','rushing_attempts']) & props.side_norm.isin(['O','U'])].copy()
# Determine dominant period separately by market. If period is all-null, preserve null.
period_choice={}
if 'period' in p.columns:
    for m,g in p.groupby('bet_type'):
        vc=g['period'].fillna('__NULL__').value_counts()
        period_choice[m]=vc.index[0]
        val=period_choice[m]
        p.loc[(p.bet_type==m) & (p['period'].fillna('__NULL__')!=val),'_drop_period']=True
    p=p[~p.get('_drop_period',False).fillna(False)].copy()
(OUT/'period_choice.json').write_text(json.dumps(period_choice,default=str,indent=2))

# Remove synthetic OPEN (30), retain consensus and actual books.
p=p[p.book_id.isin(list(BOOKS))].copy()

# Build a paired O/U row per player-market-book-event. If line differs across sides, exclude that quote.
basekeys=['season','week','event_id','team','player_id','join_name','position','position_group','bet_type','book_id']
for c in basekeys:
    if c not in p.columns:p[c]=np.nan
agg=[]
for key,g in p.groupby(basekeys,dropna=False):
    go=g[g.side_norm=='O']; gu=g[g.side_norm=='U']
    if go.empty or gu.empty:continue
    # Latest-per-book already done upstream; if duplicates remain, use most recent if timestamp exists, otherwise last row.
    if 'last_updated' in g.columns:
        go=go.sort_values('last_updated'); gu=gu.sort_values('last_updated')
    ro=go.iloc[-1]; ru=gu.iloc[-1]
    lo=float(ro.value); lu=float(ru.value)
    if not np.isfinite(lo) or not np.isfinite(lu) or abs(lo-lu)>1e-9:continue
    rec=dict(zip(basekeys,key)); rec.update(line=lo,over_odds=ro.odds,under_odds=ru.odds)
    io,iu=amer_imp(ro.odds),amer_imp(ru.odds); s=io+iu
    rec['over_novig']=io/s if np.isfinite(s) and s>0 else np.nan
    rec['under_novig']=iu/s if np.isfinite(s) and s>0 else np.nan
    agg.append(rec)
q=pd.DataFrame(agg)
q.to_parquet(OUT/'paired_book_quotes.parquet',index=False)

# Canonical closing line: Action consensus (book 15) first; otherwise median of real books.
ckeys=['season','week','event_id','team','player_id','join_name','position','position_group','bet_type']
canon=[]
for key,g in q.groupby(ckeys,dropna=False):
    c=g[g.book_id==15]
    if len(c):
        r=c.iloc[-1]; source='CONSENSUS'; line=float(r.line); op=r.over_odds; up=r.under_odds; on=r.over_novig; un=r.under_novig; nb=int(g.book_id.nunique())
    else:
        rg=g[g.book_id.isin(REAL_BOOKS)]
        if rg.empty:continue
        line=float(rg.line.median()); source='REAL_BOOK_MEDIAN'; op=rg.over_odds.median(); up=rg.under_odds.median(); on=rg.over_novig.mean(); un=rg.under_novig.mean(); nb=int(rg.book_id.nunique())
    rec=dict(zip(ckeys,key)); rec.update(line=line,over_odds=op,under_odds=up,over_novig=on,under_novig=un,canonical_source=source,books_available=nb)
    canon.append(rec)
canon=pd.DataFrame(canon)
canon.to_parquet(OUT/'canonical_prop_lines.parquet',index=False)

# ---------- ACTUAL NFL STATS ----------
stats=[]
for season in SEASONS:
    f=dl(STATS_URL.format(season=season),f'stats_{season}.csv')
    s=pd.read_csv(f,low_memory=False); s['season']=season; stats.append(s)
stats=pd.concat(stats,ignore_index=True)
carry_col=pick_col(stats,['carries','rushing_attempts','rush_attempts'])
stat_pid=pick_col(stats,['player_id','gsis_id'])
stat_team=pick_col(stats,['team','recent_team'])
stat_name=pick_col(stats,['player_display_name','player_name','player'])
if carry_col is None:raise RuntimeError('No carries column in nflverse stats')
if stat_pid is None:raise RuntimeError('No player_id column in nflverse stats')
stats['week']=pd.to_numeric(stats['week'],errors='coerce')
stats['actual_carries']=pd.to_numeric(stats[carry_col],errors='coerce')
stats['stat_pid']=stats[stat_pid].astype(str).replace({'nan':np.nan,'None':np.nan})
stats['stat_team']=stats[stat_team] if stat_team else np.nan
stats['stat_name']=stats[stat_name] if stat_name else np.nan
stats['name_norm']=stats['stat_name'].map(norm_name)
keep=['season','week','stat_pid','stat_team','stat_name','name_norm','actual_carries']
if 'game_id' in stats.columns:keep.append('game_id')
st=stats[keep].drop_duplicates(['season','week','stat_pid'],keep='last')

canon['pid_str']=canon.player_id.astype(str).replace({'nan':np.nan,'None':np.nan,'<NA>':np.nan})
canon['name_norm']=canon.join_name.map(norm_name)
# primary player-id match
c=canon.merge(st,left_on=['season','week','pid_str'],right_on=['season','week','stat_pid'],how='left')
# fallback by normalized name+team for unmatched only
miss=c.actual_carries.isna()
if miss.any() and stat_team:
    fb=st[['season','week','stat_team','name_norm','actual_carries','game_id'] if 'game_id' in st.columns else ['season','week','stat_team','name_norm','actual_carries']].drop_duplicates(['season','week','stat_team','name_norm'])
    m=c.loc[miss,['season','week','team','name_norm']].merge(fb,left_on=['season','week','team','name_norm'],right_on=['season','week','stat_team','name_norm'],how='left')
    c.loc[miss,'actual_carries']=m.actual_carries.values
    if 'game_id' in m.columns:c.loc[miss,'game_id']=m.game_id.values
canon=c

# ---------- SCHEDULE CONTEXT ----------
games=pd.read_csv(dl(GAMES_URL,'games.csv'),low_memory=False)
games=games[games.season.isin(SEASONS)].copy()
# event -> game mapping is recovered from player stats game_id; when missing, team/week can map uniquely.
if 'game_id' not in canon.columns:canon['game_id']=np.nan
# fill missing game_id by season/week/team schedule uniqueness
map_rows=[]
for _,r in games.iterrows():
    map_rows.append({'season':r.season,'week':r.week,'team':r.home_team,'game_id_map':r.game_id})
    map_rows.append({'season':r.season,'week':r.week,'team':r.away_team,'game_id_map':r.game_id})
gm=pd.DataFrame(map_rows).drop_duplicates(['season','week','team'])
canon=canon.merge(gm,on=['season','week','team'],how='left')
canon['game_id']=canon.game_id.fillna(canon.game_id_map)

# ---------- FREEZE RB1/RB2 FROM PREGAME RUSH-YARD BOARD ----------
y=canon[canon.bet_type=='rushing_yards'].copy()
pos=y.position_group.fillna(y.position).astype(str).str.upper()
y=y[pos.eq('RB')].copy()
carry=canon[canon.bet_type=='rushing_attempts'].copy()
# Create keyed carry lookup
carry_lookup=carry.set_index(['season','week','event_id','team','pid_str'],drop=False)
# QB controls, canonical pregame
qb=canon[(canon.bet_type=='rushing_yards') & (canon.position.astype(str).str.upper().eq('QB') | canon.position_group.astype(str).str.upper().eq('QB'))].copy()

pairs=[]; exclusions=[]
for key,g in y.groupby(['season','week','event_id','team'],dropna=False):
    g=g.dropna(subset=['line']).sort_values(['line','join_name'],ascending=[False,True])
    if len(g)<2:
        exclusions.append((*key,'fewer_than_two_rb_rush_yd_lines'));continue
    if float(g.iloc[0].line)==float(g.iloc[1].line):
        exclusions.append((*key,'top_two_rush_yd_line_tie'));continue
    r1,r2=g.iloc[0],g.iloc[1]
    rec={'season':key[0],'week':key[1],'event_id':key[2],'team':key[3],
         'game_id':r1.game_id,'rb1':r1.join_name,'rb1_pid':r1.pid_str,'rb1_rush_yd_line':r1.line,
         'rb2':r2.join_name,'rb2_pid':r2.pid_str,'rb2_rush_yd_line':r2.line,
         'rb_rush_yd_gap':r1.line-r2.line,'rb_prop_count':len(g)}
    ok=True
    for label,r in [('rb1',r1),('rb2',r2)]:
        lk=(key[0],key[1],key[2],key[3],r.pid_str)
        if lk not in carry_lookup.index:
            ok=False; rec[label+'_carry_missing']=True; continue
        z=carry_lookup.loc[lk]
        if isinstance(z,pd.DataFrame):z=z.iloc[-1]
        rec[label+'_carry_line']=z.line; rec[label+'_actual_carries']=z.actual_carries
        rec[label+'_carry_over_odds']=z.over_odds; rec[label+'_carry_under_odds']=z.under_odds
        rec[label+'_carry_over_novig']=z.over_novig; rec[label+'_carry_under_novig']=z.under_novig
        rec[label+'_carry_source']=z.canonical_source; rec[label+'_carry_books']=z.books_available
    if not ok:
        exclusions.append((*key,'missing_carry_line_for_frozen_rb'));continue
    if pd.isna(rec.get('rb1_actual_carries')) or pd.isna(rec.get('rb2_actual_carries')):
        exclusions.append((*key,'missing_actual_carries_or_void'));continue
    # QB rush-yard control = highest listed QB line for team-game
    qg=qb[(qb.season==key[0])&(qb.week==key[1])&(qb.event_id==key[2])&(qb.team==key[3])]
    if len(qg):
        q0=qg.sort_values('line',ascending=False).iloc[0]; rec['qb']=q0.join_name; rec['qb_rush_yd_line']=q0.line
    pairs.append(rec)
pairs=pd.DataFrame(pairs)
pd.DataFrame(exclusions,columns=['season','week','event_id','team','reason']).to_csv(OUT/'exclusions.csv',index=False)

# Grade main lines. Push is retained but excluded from binary 2-leg analysis.
def grade(a,l):return 'O' if a>l else 'U' if a<l else 'P'
pairs['rb1_result']=[grade(a,l) for a,l in zip(pairs.rb1_actual_carries,pairs.rb1_carry_line)]
pairs['rb2_result']=[grade(a,l) for a,l in zip(pairs.rb2_actual_carries,pairs.rb2_carry_line)]
pairs['pair']=pairs.rb1_result+pairs.rb2_result
pairs['rb1_margin']=pairs.rb1_actual_carries-pairs.rb1_carry_line
pairs['rb2_margin']=pairs.rb2_actual_carries-pairs.rb2_carry_line
pairs['carry_gap']=(pairs.rb1_carry_line-pairs.rb2_carry_line).abs()
pairs['combined_line']=pairs.rb1_carry_line+pairs.rb2_carry_line
pairs['rb2_share']=pairs.rb2_carry_line/pairs.combined_line
pairs['season_phase']=np.where(pairs.week<=4,'W1-4',np.where(pairs.week<=10,'W5-10',np.where(pairs.week<=18,'W11-18','POST')))
# Schedule join
schedcols=['game_id','game_type','home_team','away_team','spread_line','total_line','gameday']
pairs=pairs.merge(games[schedcols].drop_duplicates('game_id'),on='game_id',how='left')
pairs['home_away']=np.where(pairs.team==pairs.home_team,'HOME',np.where(pairs.team==pairs.away_team,'AWAY','UNKNOWN'))
# nflverse spread_line positive = home favored. team_fav_margin positive means this team favored.
pairs['team_fav_margin']=np.where(pairs.team==pairs.home_team,pairs.spread_line,-pairs.spread_line)
pairs['fav_status']=np.where(pairs.team_fav_margin>0,'FAVORITE',np.where(pairs.team_fav_margin<0,'UNDERDOG','PICKEM'))
def spread_bucket(x):
    if pd.isna(x):return 'missing'
    if x>=7:return 'fav 7+'
    if x>=3.5:return 'fav 3.5-6.5'
    if x>0:return 'fav 0.5-3'
    if x==0:return 'pickem'
    if x>=-3:return 'dog 0.5-3'
    if x>=-6.5:return 'dog 3.5-6.5'
    return 'dog 7+'
def total_bucket(x):
    if pd.isna(x):return 'missing'
    if x<42:return '<42'
    if x<48:return '42-47.5'
    return '48+'
def qb_bucket(x):
    if pd.isna(x):return 'missing'
    if x<15:return '<15'
    if x<30:return '15-29.5'
    return '30+'
def gap_bucket(x):
    if x<=2:return '<=2'
    if x<=5:return '2.5-5'
    if x<=8:return '5.5-8'
    return '8+'
def share_bucket(x):
    if x<.30:return '<30%'
    if x<.40:return '30-39.9%'
    return '40%+'
def comb_bucket(x):
    if x<20:return '<20'
    if x<25:return '20-24.5'
    return '25+'
pairs['spread_bucket']=pairs.team_fav_margin.map(spread_bucket)
pairs['total_bucket']=pairs.total_line.map(total_bucket)
pairs['qb_bucket']=pairs.get('qb_rush_yd_line',pd.Series(np.nan,index=pairs.index)).map(qb_bucket)
pairs['carry_gap_bucket']=pairs.carry_gap.map(gap_bucket)
pairs['rb2_share_bucket']=pairs.rb2_share.map(share_bucket)
pairs['combined_line_bucket']=pairs.combined_line.map(comb_bucket)
pairs.to_csv(OUT/'full_team_game_pairs.csv',index=False)

# ---------- STATS ----------
def summary(g,label='All'):
    x=g[g.pair.isin(['UO','OU','UU','OO'])].copy(); n=len(x)
    if n==0:return {'segment':label,'n':0}
    UO=int((x.pair=='UO').sum()); OU=int((x.pair=='OU').sum()); UU=int((x.pair=='UU').sum()); OO=int((x.pair=='OO').sum())
    p1u=(UO+UU)/n; p2o=(UO+OO)/n; p2u=(OU+UU)/n; p1o=(OU+OO)/n
    euo=p1u*p2o; eou=p2u*p1o
    lo1,hi1=wilson(UO,n); lo2,hi2=wilson(OU,n)
    # independence association test
    tab=np.array([[UU,UO],[OU,OO]])
    try:OR,fp=fisher_exact(tab)
    except:OR,fp=np.nan,np.nan
    # no-vig independence expectation from archived leg prices where available
    pred_uo=(x.rb1_carry_under_novig*x.rb2_carry_over_novig).mean()
    pred_ou=(x.rb2_carry_under_novig*x.rb1_carry_over_novig).mean()
    return {'segment':label,'n':n,'push_pairs_excluded':int((~g.pair.isin(['UO','OU','UU','OO'])).sum()),
        'rb1U_rb2O_hits':UO,'rb1U_rb2O_rate':UO/n,'rb1U_rb2O_ci_low':lo1,'rb1U_rb2O_ci_high':hi1,'rb1U_rb2O_marginal_independence':euo,'rb1U_rb2O_uplift_vs_marginal':(UO/n)/euo if euo else np.nan,'rb1U_rb2O_avg_book_novig_independence':pred_uo,'rb1U_rb2O_uplift_vs_book_novig':(UO/n)/pred_uo if pred_uo else np.nan,'rb1U_rb2O_fair_american':fair_american(UO/n),
        'rb2U_rb1O_hits':OU,'rb2U_rb1O_rate':OU/n,'rb2U_rb1O_ci_low':lo2,'rb2U_rb1O_ci_high':hi2,'rb2U_rb1O_marginal_independence':eou,'rb2U_rb1O_uplift_vs_marginal':(OU/n)/eou if eou else np.nan,'rb2U_rb1O_avg_book_novig_independence':pred_ou,'rb2U_rb1O_uplift_vs_book_novig':(OU/n)/pred_ou if pred_ou else np.nan,'rb2U_rb1O_fair_american':fair_american(OU/n),
        'both_under':UU,'both_over':OO,'fisher_odds_ratio_same_direction':OR,'fisher_p':fp,
        'pearson_margin_corr':x[['rb1_margin','rb2_margin']].corr().iloc[0,1] if n>2 else np.nan,'spearman_margin_corr':x[['rb1_margin','rb2_margin']].corr(method='spearman').iloc[0,1] if n>2 else np.nan}

overall=summary(pairs,'All 2023-2025')
pd.DataFrame([overall]).to_csv(OUT/'overall_summary.csv',index=False)
# Conditional substitution
x=pairs[pairs.pair.isin(['UO','OU','UU','OO'])]
cond=pd.DataFrame([
 {'metric':'P(RB2 Over | RB1 Under)','value':(x.loc[x.rb1_result=='U','rb2_result']=='O').mean(),'n':int((x.rb1_result=='U').sum())},
 {'metric':'P(RB2 Over | RB1 Over)','value':(x.loc[x.rb1_result=='O','rb2_result']=='O').mean(),'n':int((x.rb1_result=='O').sum())},
 {'metric':'P(RB1 Over | RB2 Under)','value':(x.loc[x.rb2_result=='U','rb1_result']=='O').mean(),'n':int((x.rb2_result=='U').sum())},
 {'metric':'P(RB1 Over | RB2 Over)','value':(x.loc[x.rb2_result=='O','rb1_result']=='O').mean(),'n':int((x.rb2_result=='O').sum())},
])
cond.to_csv(OUT/'conditional_substitution.csv',index=False)

# Pre-specified segmentation
segments=[]
def add_groups(col):
    for val,g in pairs.groupby(col,dropna=False):segments.append(summary(g,f'{col}={val}'))
for ccol in ['season','game_type','season_phase','home_away','fav_status','spread_bucket','total_bucket','qb_bucket','carry_gap_bucket','rb2_share_bucket','combined_line_bucket','rb_prop_count']:
    if ccol in pairs.columns:add_groups(ccol)
# threshold filters
for t in [1,2,3,4,5,6,8,10]:segments.append(summary(pairs[pairs.carry_gap<=t],f'carry_gap<={t}'))
for t in [4.5,5.5,6.5,7.5,8.5,9.5,10.5]:segments.append(summary(pairs[pairs.rb2_carry_line>=t],f'rb2_line>={t}'))
for t in [.30,.35,.40,.45]:segments.append(summary(pairs[pairs.rb2_share>=t],f'rb2_share>={t:.2f}'))
for t in [18,20,22,24,26,28]:segments.append(summary(pairs[pairs.combined_line>=t],f'combined_line>={t}'))
seg=pd.DataFrame(segments)
# BH FDR applied to Fisher p values as exploratory multiplicity control
if 'fisher_p' in seg:seg['fisher_p_BH_FDR']=bh_adjust(seg.fisher_p.values)
seg.to_csv(OUT/'angle_scan_prespecified.csv',index=False)

# Discovery (2023-24) / validation (2025) for exact pre-specified broad rules
rules=[]
def rule_row(name,mask):
    for split,smask in [('discovery_2023_2024',pairs.season.isin([2023,2024])),('validation_2025',pairs.season.eq(2025)),('all',pd.Series(True,index=pairs.index))]:
        r=summary(pairs[mask & smask],f'{name}|{split}'); r['rule']=name;r['split']=split;rules.append(r)
rule_row('all',pd.Series(True,index=pairs.index))
rule_row('true_committee_gap<=2',pairs.carry_gap<=2)
rule_row('committee_gap<=5',pairs.carry_gap<=5)
rule_row('rb2_line>=7.5',pairs.rb2_carry_line>=7.5)
rule_row('rb2_share>=40%',pairs.rb2_share>=.40)
rule_row('favorite',pairs.team_fav_margin>0)
rule_row('favorite_and_gap<=5',(pairs.team_fav_margin>0)&(pairs.carry_gap<=5))
rule_row('low_QB_rush_<15',pairs.qb_rush_yd_line<15)
rule_row('low_QB_and_gap<=5',(pairs.qb_rush_yd_line<15)&(pairs.carry_gap<=5))
pd.DataFrame(rules).to_csv(OUT/'discovery_validation_rules.csv',index=False)

# Team-season / New England descriptive only
teamrows=[]
for (season,team),g in pairs.groupby(['season','team']):
    r=summary(g,f'{season}_{team}');r['season']=season;r['team']=team;teamrows.append(r)
pd.DataFrame(teamrows).to_csv(OUT/'team_season_descriptive.csv',index=False)
if (pairs.team=='NE').any():
    nerows=[]
    for season,g in pairs[pairs.team=='NE'].groupby('season'):nerows.append(summary(g,f'NE_{season}'))
    nerows.append(summary(pairs[pairs.team=='NE'],'NE_all'))
    pd.DataFrame(nerows).to_csv(OUT/'new_england_focus.csv',index=False)

# Give sensitivity grid. Diagnostic only; price is NOT held fixed in real alternate markets.
give=[]
for strat in ['RB1_U_RB2_O','RB2_U_RB1_O']:
    for ug in range(0,5):
        for og in range(0,5):
            if strat=='RB1_U_RB2_O':hit=(pairs.rb1_actual_carries < pairs.rb1_carry_line+ug)&(pairs.rb2_actual_carries > pairs.rb2_carry_line-og)
            else:hit=(pairs.rb2_actual_carries < pairs.rb2_carry_line+ug)&(pairs.rb1_actual_carries > pairs.rb1_carry_line-og)
            rate=hit.mean(); give.append({'strategy':strat,'under_leg_easier_by':ug,'over_leg_easier_by':og,'hits':int(hit.sum()),'n':len(pairs),'rate':rate,'fair_american':fair_american(rate),'roi_if_+325_unchanged':rate*4.25-1})
pd.DataFrame(give).to_csv(OUT/'give_sensitivity.csv',index=False)

# ROI/fair-price grid from empirical rates; not actual archived SGP payouts.
roi=[]
for strat,rate in [('RB1_U_RB2_O',overall['rb1U_rb2O_rate']),('RB2_U_RB1_O',overall['rb2U_rb1O_rate'])]:
    for price in [150,175,200,225,250,275,300,325,350,400,450,500,600]:roi.append({'strategy':strat,'offered_american':price,'empirical_rate':rate,'hypothetical_roi':rate*(1+price/100)-1})
pd.DataFrame(roi).to_csv(OUT/'hypothetical_roi_grid.csv',index=False)

# Bayesian P(true rate > +325 breakeven) Jeffreys prior.
bayes=[]; be=1/4.25
for strat,hitcol in [('RB1_U_RB2_O','UO'),('RB2_U_RB1_O','OU')]:
    k=int((x.pair==hitcol).sum()); n=len(x); a=k+.5;b=n-k+.5
    bayes.append({'strategy':strat,'hits':k,'n':n,'observed_rate':k/n,'post_mean':a/(a+b),'post_95_low':beta.ppf(.025,a,b),'post_95_high':beta.ppf(.975,a,b),'prob_true_rate_gt_23.529pct':1-beta.cdf(be,a,b)})
pd.DataFrame(bayes).to_csv(OUT/'bayesian_325.csv',index=False)

# Leave-one-team-out
loo=[]
for team in sorted(pairs.team.dropna().unique()):
    r=summary(pairs[pairs.team!=team],f'exclude_{team}');r['excluded_team']=team;loo.append(r)
pd.DataFrame(loo).to_csv(OUT/'leave_one_team_out.csv',index=False)

# Cluster bootstrap by team-season, preserving intra-backfield dependence.
rng=np.random.default_rng(20260901); clusters=list(pairs.groupby(['season','team']).groups.keys()); boot=[]
for _ in range(3000):
    sampled=rng.choice(len(clusters),len(clusters),replace=True); chunks=[pairs.loc[pairs.groupby(['season','team']).groups[clusters[i]]] for i in sampled]
    bdf=pd.concat(chunks,ignore_index=True); s=summary(bdf)
    boot.append((s['rb1U_rb2O_rate'],s['rb2U_rb1O_rate'],s['rb1U_rb2O_uplift_vs_marginal'],s['rb2U_rb1O_uplift_vs_marginal']))
bd=pd.DataFrame(boot,columns=['rb1U_rb2O_rate','rb2U_rb1O_rate','rb1U_rb2O_uplift','rb2U_rb1O_uplift'])
bd.quantile([.025,.5,.975]).to_csv(OUT/'cluster_bootstrap_intervals.csv')

# Coverage/audit details
coverage=[]
for season in SEASONS:
    ps=pairs[pairs.season==season]
    coverage.append({'season':season,'qualifying_team_games':len(ps),'unique_events':ps.event_id.nunique(),'unique_teams':ps.team.nunique(),'weeks':','.join(map(str,sorted(ps.week.dropna().unique().astype(int)))),'push_pairs':int((ps.pair.str.contains('P')).sum())})
pd.DataFrame(coverage).to_csv(OUT/'coverage_audit.csv',index=False)

# Machine-readable report summary
result={'method':{'seasons':SEASONS,'rb_definition':'RB1/RB2 frozen by canonical pregame rushing-yard line; QBs excluded; tied top lines excluded','carry_line':'Action Network latest archived closing/main quote, consensus book 15 preferred, real-book median fallback','actuals':'nflverse weekly player stats','spread':'nflverse closing spread; positive means home favored'},'overall':overall,'conditional':cond.to_dict('records'),'coverage':coverage,'period_choice':period_choice}
(OUT/'result_summary.json').write_text(json.dumps(result,indent=2,default=lambda v:None if pd.isna(v) else float(v) if isinstance(v,(np.floating,)) else int(v) if isinstance(v,(np.integer,)) else str(v)))

# Plaintext summary for rapid inspection in workflow logs.
print('=== COVERAGE ===');print(pd.DataFrame(coverage).to_string(index=False))
print('\n=== OVERALL ===');print(pd.DataFrame([overall]).to_string(index=False))
print('\n=== CONDITIONAL ===');print(cond.to_string(index=False))
print('\n=== DISCOVERY / VALIDATION ===');print(pd.DataFrame(rules)[['rule','split','n','rb1U_rb2O_rate','rb1U_rb2O_marginal_independence','rb2U_rb1O_rate','rb2U_rb1O_marginal_independence']].to_string(index=False))
