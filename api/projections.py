# Vercel Python serverless function: live ML projections for the next gameweek.
# Pure stdlib + the trained ridge coefficients in model/model_export.json.
from http.server import BaseHTTPRequestHandler
import json, urllib.request, os, math

def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (fpl-model-hub)'})
    return json.loads(urllib.request.urlopen(req,timeout=20).read())

def load_model():
    # CHEAP+4 in-season model: validated 6/6 rolling-origin windows,
    # RMSE 2.633 / R2 +0.141 (perfect-deadline band is 2.7-2.9 / 0.12-0.17)
    path=os.path.join(os.path.dirname(__file__),'..','model','inseason_export.json')
    return json.load(open(path))

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            M=load_model()
            boot=fetch('https://fantasy.premierleague.com/api/bootstrap-static/')
            fixtures=fetch('https://fantasy.premierleague.com/api/fixtures/')
            ev=next((e['id'] for e in boot['events'] if e.get('is_next')),None) \
               or next((e['id'] for e in boot['events'] if not e['finished']),1)
            gws=ev-1
            if gws<2:
                self._json({'gw':ev,'players':[],'note':'cold start: <2 finished GWs, use baked model'})
                return
            # team rolling GF/GA last 6 from finished fixtures
            hist={}
            for f in fixtures:
                if not f.get('finished'):continue
                for tid,gf,ga in ((f['team_h'],f['team_h_score'],f['team_a_score']),
                                  (f['team_a'],f['team_a_score'],f['team_h_score'])):
                    hist.setdefault(tid,[]).append((f['event'],gf,ga))
            roll={}
            for tid,rows in hist.items():
                rows.sort(); last=rows[-6:]
                roll[tid]=(sum(r[1] for r in last)/len(last), sum(r[2] for r in last)/len(last))
            # next fixture per team
            nxt={}
            for f in fixtures:
                if f['event']==ev:
                    nxt[f['team_h']]=(f['team_a'],1); nxt[f['team_a']]=(f['team_h'],0)
            FE=M['feats']; C=dict(zip(FE,M['coef']))
            out=[]
            for e in boot['elements']:
                if e['status'] in ('i','u','n'): continue
                if e['team'] not in nxt: continue
                gp=max(1,gws)
                mins_avg=e['minutes']/gp
                if mins_avg<=0: continue
                opp,home=nxt[e['team']]
                tgf,tga=roll.get(e['team'],(1.4,1.4)); ogf,oga=roll.get(opp,(1.4,1.4))
                pos={1:'GK',2:'DEF',3:'MID',4:'FWD'}[e['element_type']]
                feats={'form4':float(e.get('form') or 0),'mins4':mins_avg,
                       'ppg':float(e.get('points_per_game') or 0),
                       'xgi4':float(e.get('expected_goal_involvements') or 0)/gp,
                       'price':e['now_cost']/10,'home':home,
                       'ict4':float(e.get('ict_index') or 0)/gp,
                       'sel_own':float(e.get('selected_by_percent') or 0)*10000,
                       'tbal':float((e.get('transfers_in_event') or 0)-(e.get('transfers_out_event') or 0)),
                       'start4':float(e.get('starts') or 0)/gp,
                       't_gf6':tgf,'t_ga6':tga,'o_gf6':ogf,'o_ga6':oga,
                       'pos_GK':1 if pos=='GK' else 0,'pos_DEF':1 if pos=='DEF' else 0,
                       'pos_MID':1 if pos=='MID' else 0,'pos_FWD':1 if pos=='FWD' else 0}
                xp=M['intercept']+sum(C[f]*feats[f] for f in FE)
                cop=e.get('chance_of_playing_next_round')
                if e['status']=='d': xp*= (cop/100 if cop is not None else 0.5)
                out.append({'id':e['id'],'xp':round(max(0,xp),2)})
            self._json({'gw':ev,'model':'ridge','holdout':M.get('holdout'),'players':out})
        except Exception as ex:
            self._json({'error':str(ex)[:200]},500)
    def _json(self,obj,code=200):
        body=json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','s-maxage=1800, stale-while-revalidate=3600')
        self.end_headers()
        self.wfile.write(body)
