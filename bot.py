import os, json, requests
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf

NY = ZoneInfo("America/New_York")
TOKEN, CHAT = os.environ["TG_TOKEN"], str(os.environ["TG_CHAT_ID"])
API = f"https://api.telegram.org/bot{TOKEN}"
COINS = {"gold": "GC=F", "xau": "GC=F", "xauusd": "GC=F", "silver": "SI=F", "oil": "CL=F",
         "btc": "BTC-USD", "eth": "ETH-USD", "sol": "SOL-USD", "bnb": "BNB-USD",
         "eurusd": "EURUSD=X", "gbpusd": "GBPUSD=X", "usdjpy": "JPY=X"}
TFS = ["30m", "1h", "2h", "3h", "4h"]
MIN_SCORE, MIN_RR = 6, 2.0
SRC = {"5m": ("5m", "5d"), "30m": ("30m", "55d"), "1h": ("60m", "180d"), "2h": ("60m", "180d"),
       "3h": ("60m", "180d"), "4h": ("60m", "180d"), "1d": ("1d", "3y")}

# ---------- Telegram ----------
def send(text):
    requests.post(f"{API}/sendMessage", data={"chat_id": CHAT, "text": text})

def send_photo(path):
    with open(path, "rb") as f:
        requests.post(f"{API}/sendPhoto", data={"chat_id": CHAT}, files={"photo": f})

MENU = [("🥇 Gold", "gold"), ("🥈 Silver", "silver"), ("🛢 Oil", "oil"),
        ("₿ BTC", "btc"), ("Ξ ETH", "eth"), ("SOL", "sol"), ("BNB", "bnb"),
        ("EUR/USD", "eurusd"), ("GBP/USD", "gbpusd"), ("USD/JPY", "usdjpy")]

HELP = ("🤖 Kon command ki kore:\n\n"
        "/list - coin/pair er button list\n"
        "/coin btc - sorasori btc analysis (je kono nam)\n"
        "/now - ekhoni analysis\n"
        "/status - ekhon kon symbol, khola trade, win/loss\n"
        "/stop - auto signal bondho\n"
        "/start - auto signal chalu\n"
        "/help - ei list\n\n"
        "🕗 Auto signal shudhu New York session e (Mon-Fri, 08:00-17:00 ET)")

def send_menu():
    rows = [MENU[i:i+3] for i in range(0, len(MENU), 3)]
    kb = {"inline_keyboard": [[{"text": t, "callback_data": "coin:" + k} for t, k in row] for row in rows]}
    requests.post(f"{API}/sendMessage", data={
        "chat_id": CHAT, "text": "📋 Kon symbol analysis korbo? Button e chap din (onno nam lagle: /coin xrp)",
        "reply_markup": json.dumps(kb)})

def set_commands():
    cmds = [("list", "Coin/pair er list"), ("now", "Ekhoni analysis"), ("status", "Status dekhun"),
            ("stop", "Auto signal bondho"), ("start", "Auto signal chalu"), ("help", "Command list")]
    requests.post(f"{API}/setMyCommands",
                  json={"commands": [{"command": c, "description": d} for c, d in cmds]})

# ---------- Data & indicators ----------
def load(sym, tf):
    iv, per = SRC[tf]
    df = yf.download(sym, interval=iv, period=per, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(NY)
    if tf in ("2h", "3h", "4h"):
        df = df.resample(tf).agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last", "Volume": "sum"}).dropna()
    return df

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l)

def atr(df, n=14):
    tr = pd.concat([df.High - df.Low, (df.High - df.Close.shift()).abs(),
                    (df.Low - df.Close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

# ---------- Price-action / SMC building blocks ----------
def swings(df, n=3):
    hi = df.High[df.High == df.High.rolling(2*n+1, center=True).max()]
    lo = df.Low[df.Low == df.Low.rolling(2*n+1, center=True).min()]
    return hi, lo

def structure(df):                       # +1 = HH/HL, -1 = LH/LL, 0 = unclear
    hi, lo = swings(df)
    h, l = hi.tail(2).values, lo.tail(2).values
    if len(h) < 2 or len(l) < 2: return 0
    if h[-1] > h[-2] and l[-1] > l[-2]: return 1
    if h[-1] < h[-2] and l[-1] < l[-2]: return -1
    return 0

def sr_levels(df, tol=0.0015):           # clustered swing points -> (level, touches)
    hi, lo = swings(df, 4)
    groups = []
    for p in sorted(list(hi.values) + list(lo.values)):
        if groups and abs(p - groups[-1][-1]) / p < tol: groups[-1].append(p)
        else: groups.append([p])
    return [(float(np.mean(g)), len(g)) for g in groups]

def fib_zone(df, bias):                  # 0.5-0.786 retracement of last leg
    d = df.tail(120)
    H, L = d.High.max(), d.Low.min()
    R = H - L
    if bias == 1 and d.Low.idxmin() < d.High.idxmax():
        return H - 0.786*R, H - 0.5*R
    if bias == -1 and d.High.idxmax() < d.Low.idxmin():
        return L + 0.5*R, L + 0.786*R
    return None

def order_block(df, bias, a):
    for i in range(len(df) - 4, 20, -1):
        c = df.iloc[i]
        move = df.Close.iloc[i + 3] - c.Close
        if bias == 1 and c.Close < c.Open and move > 1.5*a: return (c.Low, c.High)
        if bias == -1 and c.Close > c.Open and -move > 1.5*a: return (c.Low, c.High)
    return None

def fvgs(df, bias):
    out = []
    for i in range(2, len(df)):
        a, c = df.iloc[i-2], df.iloc[i]
        if bias == 1 and c.Low > a.High: out.append((a.High, c.Low))
        if bias == -1 and c.High < a.Low: out.append((c.High, a.Low))
    return out[-5:]

def swept(df, bias):                     # liquidity sweep of a prior swing, then close back
    hi, lo = swings(df, 3)
    r = df.tail(12)
    if bias == 1:
        return any((r.Low < p).any() and r.Close.iloc[-1] > p for p in lo[lo.index < r.index[0]].tail(3).values)
    return any((r.High > p).any() and r.Close.iloc[-1] < p for p in hi[hi.index < r.index[0]].tail(3).values)

def wyckoff(df, bias, a):                # spring / upthrust at the edge of a tight range
    r, t = df.tail(40).iloc[:-5], df.tail(5)
    if (r.High.max() - r.Low.min()) > 6*a: return False
    if bias == 1: return t.Low.min() < r.Low.min() and df.Close.iloc[-1] > r.Low.min()
    return t.High.max() > r.High.max() and df.Close.iloc[-1] < r.High.max()

# ---------- Plan builder ----------
def plan_for(ticker, tf, bias, h4):
    df = load(ticker, tf)
    if len(df) < 100: return None
    a, price = float(atr(df).iloc[-1]), float(df.Close.iloc[-1])
    r = float(rsi(df.Close).iloc[-1])
    pts, why = 0, []
    if structure(df) == bias:
        pts += 1; why.append(f"{tf} structure: " + ("HH + HL (bullish)" if bias == 1 else "LH + LL (bearish)"))
    fz = fib_zone(df, bias)
    if fz and fz[0] <= price <= fz[1]:
        pts += 1; why.append("Price Fibonacci 0.5-0.786 (OTE) zone-e")
    ob = order_block(df, bias, a)
    if ob and ob[0] - .3*a <= price <= ob[1] + .3*a:
        pts += 1; why.append(f"Order Block {ob[0]:.2f}-{ob[1]:.2f} er kache")
    if any(lo - .2*a <= price <= hi + .2*a for lo, hi in fvgs(df, bias)):
        pts += 1; why.append("Fair Value Gap er bhetore")
    if swept(df, bias):
        pts += 1; why.append("Liquidity sweep (stop hunt) hoye geche")
    if wyckoff(df, bias, a):
        pts += 1; why.append("Wyckoff " + ("Spring" if bias == 1 else "Upthrust"))
    lv = [(l, t, "Major") for l, t in sr_levels(h4) if t >= 3] + [(l, t, "Minor") for l, t in sr_levels(df) if t >= 2]
    near = [x for x in lv if (bias == 1 and x[0] <= price + .2*a and price - x[0] <= a)
                          or (bias == -1 and x[0] >= price - .2*a and x[0] - price <= a)]
    if near:
        l, t, kind = max(near, key=lambda x: x[1])
        pts += 1; why.append(f"{kind} {'Support' if bias == 1 else 'Resistance'} {l:.2f} ({t} bar touch)")
    if (bias == 1 and r < 60) or (bias == -1 and r > 40):
        pts += 1; why.append(f"RSI {r:.0f} (momentum bipokkhe na)")

    hi, lo = swings(df, 3)
    if bias == 1:
        c = [x for x in lo.tail(5).values if x < price]
        sl = (max(c) if c else price - 2*a) - .3*a
    else:
        c = [x for x in hi.tail(5).values if x > price]
        sl = (min(c) if c else price + 2*a) + .3*a
    risk = min(max(abs(price - sl), .8*a), 3*a)
    sl = price - bias*risk
    tps = [l for l, _, _ in lv if bias*(l - price) >= 2*risk]
    tp = (min(tps) if bias == 1 else max(tps)) if tps else price + bias*2*risk
    return {"tf": tf, "side": "BUY" if bias == 1 else "SELL", "entry": price, "sl": float(sl),
            "tp": float(tp), "rr": abs(tp - price)/risk, "score": pts, "why": why, "atr": a, "lv": lv}

def analyze(ticker):
    d1, h4 = load(ticker, "1d"), load(ticker, "4h")
    if d1.empty or h4.empty: return None, "data pawa jayni"
    sd, sh = structure(d1), structure(h4)
    if sd == 0 or sd != sh:
        return None, f"Daily/4H dik meleni (D={sd}, 4H={sh}) -> ekhon trade nei"
    plans = [p for p in (plan_for(ticker, tf, sd, h4) for tf in TFS) if p]
    best = max(plans, key=lambda p: (p["score"], TFS.index(p["tf"])))
    return best, f"Sera timeframe {best['tf']}, score {best['score']}/8, RR 1:{best['rr']:.1f}"

# ---------- Charts & message ----------
def make_chart(name, p, ticker):
    df = load(ticker, p["tf"]).tail(90)
    near = [l for l, _, _ in p["lv"] if abs(l - p["entry"]) < 6*p["atr"]][:4]
    mpf.plot(df, type="candle", style="charles", figsize=(10, 6),
             hlines=dict(hlines=[p["entry"], p["sl"], p["tp"]] + near,
                         colors=["blue", "red", "green"] + ["gray"]*len(near),
                         linestyle="--", linewidths=1),
             title=f"{name.upper()} {p['tf']} | {p['side']} | NY Session",
             savefig=dict(fname="chart.png", dpi=110))
    return "chart.png"

def make_overview(name, ticker, note):
    df = load(ticker, "1h").tail(90)
    h4 = load(ticker, "4h")
    price = float(df.Close.iloc[-1])
    a = float(atr(df).iloc[-1])
    lv = [l for l, t in sr_levels(h4) if t >= 3] + [l for l, t in sr_levels(df) if t >= 2]
    near = sorted([l for l in lv if abs(l - price) < 8*a], key=lambda l: abs(l - price))[:6]
    kw = {}
    if near:
        kw["hlines"] = dict(hlines=near, colors=["gray"]*len(near), linestyle="--", linewidths=1)
    mpf.plot(df, type="candle", style="charles", figsize=(10, 6),
             title=f"{name.upper()} 1H | Overview | NY Session",
             savefig=dict(fname="overview.png", dpi=110), **kw)
    return "overview.png"

def message(name, p, st, now):
    w, l = st["wins"], st["losses"]
    rate = f"{w/(w+l)*100:.0f}%" if w + l else "ekhono data nei"
    return (f"{'📈' if p['side']=='BUY' else '📉'} {name.upper()} | {p['side']} | Timeframe: {p['tf']}\n"
            f"🕗 New York Session | {now.strftime('%d %b %Y, %I:%M %p')} ET\n\n"
            f"Entry: {p['entry']:.2f}\nStop Loss: {p['sl']:.2f}\nTake Profit: {p['tp']:.2f}  (RR 1:{p['rr']:.1f})\n"
            f"Confluence score: {p['score']}/8\n\nKaron:\n" + "\n".join(f"• {x}" for x in p["why"]) +
            f"\n\n⚠️ Eta shombhabbo setup, nishchit noy. Sob shomoy correct hobe na. Risk 1-2% er beshi nebenna.\n"
            f"📊 Live record: {w} Win / {l} Loss (win rate: {rate})")

def check_trade(t):
    df = load(t["ticker"], "5m")
    if df.empty: return None
    df = df[df.index > pd.Timestamp(t["time"])]
    for ts, c in df.iterrows():
        if t["side"] == "BUY":
            if c.Low <= t["sl"]: return "SL", ts
            if c.High >= t["tp"]: return "TP", ts
        else:
            if c.High >= t["sl"]: return "SL", ts
            if c.Low <= t["tp"]: return "TP", ts
    return None

def resolve(arg):
    for tk in [COINS.get(arg), arg.upper() + "-USD", arg.upper() + "=X", arg.upper()]:
        if tk and not load(tk, "1d").empty: return tk
    return None

def in_session(now):
    return now.weekday() < 5 and 8 <= now.hour < 17

# ---------- Main ----------
def main():
    st = json.load(open("state.json"))
    now = datetime.now(NY)
    force = manual = False
    set_commands()

    r = requests.get(f"{API}/getUpdates", params={"offset": st["offset"], "timeout": 0}).json()
    for u in r.get("result", []):
        st["offset"] = u["update_id"] + 1
        cq = u.get("callback_query")
        if cq:                                   # button tap
            requests.post(f"{API}/answerCallbackQuery", data={"callback_query_id": cq["id"]})
            if str(cq["from"]["id"]) != CHAT: continue
            data = cq.get("data", "")
            if not data.startswith("coin:"): continue
            t = ["/coin", data[5:]]
        else:                                    # typed message
            m = u.get("message") or {}
            if str(m.get("chat", {}).get("id")) != CHAT: continue
            t = m.get("text", "").strip().lower().split()
            if not t: continue
        cmd = t[0].split("@")[0]

        if cmd in ("/coin", "/pair", "/symbol"):
            if len(t) < 2:
                send_menu()
            else:
                tk = resolve(t[1])
                if tk:
                    st["name"], st["ticker"] = t[1], tk
                    force = manual = True
                    send(f"✅ Symbol: {t[1].upper()}")
                else:
                    send(f"❌ '{t[1]}' pawa jayni. /list dekhun")
        elif cmd in ("/list", "/menu"): send_menu()
        elif cmd == "/help": send(HELP)
        elif cmd == "/now": force = manual = True
        elif cmd == "/stop": st["paused"] = True; send("⏸ Auto signal bondho")
        elif cmd == "/start": st["paused"] = False; send("▶️ Auto signal chalu (New York session)\n\n" + HELP)
        elif cmd == "/status":
            send(f"Symbol: {st['name'].upper()}\nKhola trade: {len(st['trades'])}\n"
                 f"Record: {st['wins']}W / {st['losses']}L\nPaused: {st['paused']}\n"
                 f"NY time: {now.strftime('%I:%M %p')} | Session: {'ON' if in_session(now) else 'OFF'}")

    keep = []
    for t in st["trades"]:
        res = check_trade(t)
        if not res: keep.append(t); continue
        kind, ts = res
        risk = abs(t["entry"] - t["sl"])
        if kind == "TP": st["wins"] += 1; txt = f"🟢 TAKE PROFIT HIT (+{abs(t['tp']-t['entry'])/risk:.1f}R)"
        else: st["losses"] += 1; txt = "🔴 STOP LOSS HIT (-1R)"
        send(f"{txt}\n{t['name'].upper()} {t['side']} | Entry {t['entry']:.2f} -> {t['tp'] if kind=='TP' else t['sl']:.2f}\n"
             f"Time: {ts.strftime('%d %b, %I:%M %p')} ET\nRecord: {st['wins']}W / {st['losses']}L\n\nNotun analysis shuru hocche...")
        if t["ticker"] == st["ticker"]: force = True
    st["trades"] = keep

    last = datetime.fromisoformat(st["last"]) if st["last"] else None
    auto = (not st["paused"]) and in_session(now) and (last is None or (now - last).total_seconds() >= 900)
    has_open = any(t["ticker"] == st["ticker"] for t in st["trades"])
    if ((force and (manual or in_session(now))) or auto) and not has_open:
        p, note = analyze(st["ticker"])
        st["last"] = now.isoformat()
        if p and p["score"] >= MIN_SCORE and p["rr"] >= MIN_RR:
            try:
                send_photo(make_chart(st["name"], p, st["ticker"]))
            except Exception as e:
                print("chart error:", e)
            send(message(st["name"], p, st, now))
            st["trades"].append({"name": st["name"], "ticker": st["ticker"], "side": p["side"],
                                 "entry": p["entry"], "sl": p["sl"], "tp": p["tp"], "time": now.isoformat()})
        elif manual:
            try:
                send_photo(make_overview(st["name"], st["ticker"], note))
            except Exception as e:
                print("chart error:", e)
            send(f"ℹ️ {st['name'].upper()}: ekhon A+ setup nei.\n{note}\n🕗 {now.strftime('%I:%M %p')} ET")
    elif force and not manual:
        send("🕗 New York session sesh. Porer session e notun setup pathabo.")

    json.dump(st, open("state.json", "w"), indent=1)

if __name__ == "__main__":
    main()
