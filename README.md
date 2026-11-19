# Mark-Solan
#!/usr/bin/env python3
"""
SOL/USDT AI Trading Analyst  (Binance, 4H main, 1H entry confirmation)

What it does
------------
* Fetches live SOL/USDT candles + price from Binance's public REST API
  (no API key needed), or reads your own OHLCV CSV files (offline mode).
* Uses CLOSED candles only for every indicator / signal.
* Calculates EMA 200, session VWAP (anchored at 00:00 UTC each day), ATR,
  swing pivots, market structure (HH/HL/LH/LL, BOS/CHoCH), liquidity sweeps,
  and Fair Value Gaps (3-candle method) with fresh / partial / full mitigation.
* Asks "What would you like to analyse?" (1 LONG / 2 SHORT / 3 FULL MARKET ANALYSIS),
  then prints market analysis, setup checks, trade plan and the 12-point FINAL REPORT
  ending in LONG / SHORT / WAIT.
* Says NO TRADE / WAITING when conditions conflict. It never forces a signal.

Install:   pip install pandas numpy requests
Run live:  python sol_analyst.py
           python sol_analyst.py --mode long --capital 1000
Run on your own candles (CSV columns: open_time,open,high,low,close,volume):
           python sol_analyst.py --csv4h sol_4h.csv --csv1h sol_1h.csv --mode short

Not financial advice. Never risk more than ~1% of capital per trade.
"""

import argparse
import sys
import datetime as dt

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
SYMBOL = "SOLUSDT"
BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",   # public market-data mirror
]
INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000}
EMA_PERIOD = 200
ATR_PERIOD = 14
PIVOT_N = 3            # candles on each side to confirm a swing high/low
SL_ATR_BUFFER = 0.5    # extra room beyond invalidation, in ATRs (SOL volatility)
MIN_RR_TP2 = 2.0       # minimum R:R to TP2 for a setup to be called VALID
RISK_PCT = 1.0         # default risk per trade, in % of capital


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _to_df(rows):
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "qav", "trades", "tbb", "tbq", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


def fetch_klines(interval, limit):
    last_err = None
    for base in BASES:
        try:
            r = requests.get(f"{base}/api/v3/klines",
                             params={"symbol": SYMBOL, "interval": interval, "limit": limit},
                             timeout=15)
            r.raise_for_status()
            return _to_df(r.json()), base
        except Exception as e:           # try next mirror
            last_err = e
    raise RuntimeError(f"Binance data unavailable on all endpoints: {last_err}")


def fetch_price(base):
    try:
        r = requests.get(f"{base}/api/v3/ticker/price", params={"symbol": SYMBOL}, timeout=10)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None


def load_csv(path, interval):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"open_time", "open", "high", "low", "close", "volume"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path}: CSV needs columns {sorted(need)}")
    t = df["open_time"]
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.iloc[0] > 1e11 else "s"
        df["open_time"] = pd.to_datetime(t, unit=unit, utc=True)
    else:
        df["open_time"] = pd.to_datetime(t, utc=True)
    df["close_time"] = df["open_time"] + pd.Timedelta(milliseconds=INTERVAL_MS[interval] - 1)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


def closed_only(df, offline):
    """Drop the still-forming candle (live mode). CSV candles are trusted as closed."""
    if offline:
        return df.reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC")
    return df[df["close_time"] < now].reset_index(drop=True)


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def session_vwap(df):
    """Session VWAP: typical price x volume, reset at 00:00 UTC each day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df["open_time"].dt.floor("D")
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum()
    return pv / vv


def atr(df, period=ATR_PERIOD):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def pivots(df, n=PIVOT_N):
    """Confirmed swing highs / lows as lists of (index, price)."""
    hi, lo = df["high"].values, df["low"].values
    ph, pl = [], []
    for i in range(n, len(df) - n):
        if hi[i] > hi[i - n:i].max() and hi[i] >= hi[i + 1:i + n + 1].max():
            ph.append((i, hi[i]))
        if lo[i] < lo[i - n:i].min() and lo[i] <= lo[i + 1:i + n + 1].min():
            pl.append((i, lo[i]))
    return ph, pl


def market_structure(df, ph, pl):
    out = {"label": "UNDEFINED", "detail": "not enough swings", "break": None}
    if len(ph) < 2 or len(pl) < 2:
        return out
    hh = ph[-1][1] > ph[-2][1]
    hl = pl[-1][1] > pl[-2][1]
    if hh and hl:
        out["label"], out["detail"] = "BULLISH", "higher highs + higher lows"
    elif (not hh) and (not hl):
        out["label"], out["detail"] = "BEARISH", "lower highs + lower lows"
    else:
        out["label"] = "RANGE / TRANSITION"
        out["detail"] = ("higher high, lower low" if hh else "lower high, higher low")
    c = df["close"].iloc[-1]
    if c > ph[-1][1]:
        out["break"] = "BOS up" if out["label"] == "BULLISH" else "CHoCH up (bullish)"
    elif c < pl[-1][1]:
        out["break"] = "BOS down" if out["label"] == "BEARISH" else "CHoCH down (bearish)"
    return out


def find_fvgs(df, max_lookback=200):
    """Three-candle imbalance. Mitigation is judged from wicks after candle 3."""
    h, l = df["high"].values, df["low"].values
    out = []
    for i in range(max(2, len(df) - max_lookback), len(df)):
        if l[i] > h[i - 2]:                                   # bullish
            lower, upper, kind = h[i - 2], l[i], "bullish"
            m = l[i + 1:].min() if i + 1 < len(df) else np.inf
            status = ("fully mitigated" if m <= lower else
                      "partially mitigated" if m < upper else "fresh")
        elif h[i] < l[i - 2]:                                 # bearish
            lower, upper, kind = h[i], l[i - 2], "bearish"
            m = h[i + 1:].max() if i + 1 < len(df) else -np.inf
            status = ("fully mitigated" if m >= upper else
                      "partially mitigated" if m > lower else "fresh")
        else:
            continue
        out.append({"kind": kind, "lower": lower, "upper": upper,
                    "status": status, "time": df["open_time"].iloc[i]})
    return out


def liquidity_sweep(df, direction, ph, pl, lookback=6):
    """Wick through a prior swing level in the last few candles, closing back inside."""
    n = len(df)
    recent = df.iloc[-lookback:]
    if direction == "long":
        levels = [(i, p) for i, p in pl if i < n - lookback and i > n - 80]
        for i, p in reversed(levels):
            hit = recent[(recent["low"] < p) & (recent["close"] > p)]
            if not hit.empty:
                return {"level": p, "extreme": hit["low"].min()}
    else:
        levels = [(i, p) for i, p in ph if i < n - lookback and i > n - 80]
        for i, p in reversed(levels):
            hit = recent[(recent["high"] > p) & (recent["close"] < p)]
            if not hit.empty:
                return {"level": p, "extreme": hit["high"].max()}
    return None


def candle_confirmation(df, direction, atr_val):
    c = df.iloc[-1]
    body, rng = abs(c["close"] - c["open"]), c["high"] - c["low"]
    dir_ok = c["close"] > c["open"] if direction == "long" else c["close"] < c["open"]
    strong = dir_ok and rng > 0 and body / rng >= 0.6 and body >= 0.8 * atr_val
    vol_avg = df["volume"].iloc[-21:-1].mean()
    vol_ratio = c["volume"] / vol_avg if vol_avg > 0 else 0.0
    return {"strong": bool(strong), "vol_ratio": vol_ratio,
            "passed": bool(strong and vol_ratio >= 1.0)}


# --------------------------------------------------------------------------
# Timeframe analysis
# --------------------------------------------------------------------------
def analyse_tf(df):
    df = df.copy()
    df["ema200"] = ema(df["close"], EMA_PERIOD)
    df["vwap"] = session_vwap(df)
    df["atr"] = atr(df)
    ph, pl = pivots(df)
    return {
        "df": df, "ph": ph, "pl": pl,
        "structure": market_structure(df, ph, pl),
        "fvgs": find_fvgs(df),
        "close": df["close"].iloc[-1],
        "ema200": df["ema200"].iloc[-1],
        "vwap": df["vwap"].iloc[-1],
        "atr": df["atr"].iloc[-1],
        "ema_reliable": len(df) >= EMA_PERIOD,
    }


def pick_fvg(tf, direction):
    """Nearest un-mitigated FVG on the correct side of price + current state."""
    want = "bullish" if direction == "long" else "bearish"
    c, a, df = tf["close"], tf["atr"], tf["df"]
    best = None
    for f in tf["fvgs"]:
        if f["kind"] != want or f["status"] == "fully mitigated":
            continue
        if direction == "long" and f["lower"] > c:
            continue
        if direction == "short" and f["upper"] < c:
            continue
        dist = (c - f["upper"]) if direction == "long" else (f["lower"] - c)
        dist = max(dist, 0.0)
        if best is None or dist < best[0]:
            best = (dist, f)
    if best is None:
        return {"zone": None, "passed": False, "state": "no relevant un-mitigated FVG"}
    dist, f = best
    last3 = df.iloc[-3:]
    if direction == "long":
        touched = (last3["low"] <= f["upper"]).any() and c >= f["lower"]
        inside = f["lower"] <= c <= f["upper"]
    else:
        touched = (last3["high"] >= f["lower"]).any() and c <= f["upper"]
        inside = f["lower"] <= c <= f["upper"]
    if inside:
        state = "price is INSIDE the zone"
    elif touched:
        state = "price touched the zone and is REJECTING it"
    elif dist <= 1.5 * a:
        state = "price is APPROACHING the zone"
    else:
        state = f"zone is far away ({dist:,.2f} from price)"
    return {"zone": f, "passed": bool(touched), "state": state}


# --------------------------------------------------------------------------
# Direction analysis -> trade plan
# --------------------------------------------------------------------------
def build_targets(tf, direction, entry, risk):
    ph, pl = tf["ph"], tf["pl"]
    levels = sorted({p for _, p in ph if p > entry}) if direction == "long" \
        else sorted({p for _, p in pl if p < entry}, reverse=True)
    picked = []
    for p in levels:
        r = abs(p - entry) / risk
        if r >= 1.0 and (not picked or abs(p - picked[-1]) > 0.3 * tf["atr"]):
            picked.append(p)
        if len(picked) == 3:
            break
    for m in (1.5, 2.5, 4.0):                      # fall back to R multiples
        if len(picked) >= 3:
            break
        p = entry + risk * m if direction == "long" else entry - risk * m
        beyond = (not picked) or (p > picked[-1] if direction == "long" else p < picked[-1])
        if beyond:
            picked.append(p)
    return picked[:3]


def analyse_direction(direction, t4, t1, capital=None):
    long = direction == "long"
    c, ema_v, vw, a = t4["close"], t4["ema200"], t4["vwap"], t4["atr"]
    s = t4["structure"]
    notes = []

    # --- five mandatory checks (4H) ---
    ema_pass = bool(c > ema_v if long else c < ema_v)
    vwap_pass = bool(c > vw if long else c < vw)
    fvg = pick_fvg(t4, direction)
    struct_pass = (s["label"] == ("BULLISH" if long else "BEARISH")) or \
                  (s["break"] or "").startswith("CHoCH " + ("up" if long else "down")) or \
                  (s["break"] or "").startswith("BOS " + ("up" if long else "down"))
    sweep = liquidity_sweep(t4["df"], direction, t4["ph"], t4["pl"])
    cc = candle_confirmation(t4["df"], direction, a)

    checks = {
        "EMA 200": ema_pass, "VWAP": vwap_pass, "FVG": fvg["passed"],
        "Market Structure": struct_pass, "Candle Confirmation": cc["passed"],
    }
    checks = {k: bool(v) for k, v in checks.items()}
    passes = sum(checks.values())

    # --- 1H confirmation (cannot silently override 4H) ---
    opp = "BEARISH" if long else "BULLISH"
    h_conflict = t1["structure"]["label"] == opp
    h_vwap_ok = (t1["close"] > t1["vwap"]) if long else (t1["close"] < t1["vwap"])
    h_ok = (not h_conflict) and h_vwap_ok

    # --- trade levels ---
    entry_zone = sl = None
    tps, rrs, entry_px = [], [], None
    z = fvg["zone"]
    last_piv = (t4["pl"][-1][1] if long else t4["ph"][-1][1]) if (t4["pl"] if long else t4["ph"]) else None
    if z is not None or sweep is not None:
        if cc["passed"] and checks["FVG"]:
            entry_px = c                                   # confirmed close
        elif z is not None:
            entry_px = (z["lower"] + z["upper"]) / 2       # potential future entry
        else:
            entry_px = c
        entry_zone = (z["lower"], z["upper"]) if z is not None else (entry_px, entry_px)
        bases = []
        if sweep:
            bases.append(sweep["extreme"])
        if z is not None:
            bases.append(z["lower"] if long else z["upper"])
        if not bases and last_piv is not None:
            bases.append(last_piv)
        base = min(bases) if long else max(bases)
        sl = base - SL_ATR_BUFFER * a if long else base + SL_ATR_BUFFER * a
        risk = (entry_px - sl) if long else (sl - entry_px)
        if risk <= 0:
            entry_zone = sl = entry_px = None
        else:
            tps = build_targets(t4, direction, entry_px, risk)
            rrs = [abs(tp - entry_px) / risk for tp in tps]

    rr_ok = len(rrs) >= 2 and rrs[1] >= MIN_RR_TP2

    # --- status ---
    if passes == 5 and h_ok and rr_ok and entry_px is not None:
        status = "ACTIVE SETUP"
    elif passes >= 4 and entry_px is not None:
        status = "WAITING"
    else:
        status = "NO TRADE"
    if status != "NO TRADE" and not (ema_pass or struct_pass):
        status = "NO TRADE"

    if not t4["ema_reliable"]:
        notes.append(f"Only {len(t4['df'])} closed 4H candles loaded; EMA 200 is not fully warmed up.")
    if passes == 5 and not h_ok:
        notes.append("4H passes but 1H conflicts (structure or VWAP). Wait for 1H to align.")
    if passes == 5 and h_ok and not rr_ok:
        notes.append(f"R:R to TP2 is below {MIN_RR_TP2}:1, so it is not worth taking.")
    if sweep:
        notes.append(f"Liquidity sweep detected: wick through {sweep['level']:,.2f} "
                     f"(extreme {sweep['extreme']:,.2f}), then closed back inside.")
    if not cc["strong"]:
        notes.append("Last closed 4H candle is not a strong candle in this direction.")
    elif cc["vol_ratio"] < 1.0:
        notes.append(f"Candle is strong but volume is only {cc['vol_ratio']:.2f}x its 20-candle average.")

    size = None
    if capital and entry_px is not None and sl is not None:
        risk_amt = capital * RISK_PCT / 100
        units = risk_amt / abs(entry_px - sl)
        size = {"risk_amt": risk_amt, "units": units, "notional": units * entry_px}

    return {
        "direction": direction.upper(), "status": status, "checks": checks, "passes": passes,
        "fvg": fvg, "sweep": sweep, "candle": cc, "h": {"ok": h_ok, "conflict": h_conflict,
                                                      "vwap_ok": h_vwap_ok},
        "entry_zone": entry_zone, "entry_px": entry_px, "sl": sl, "tps": tps, "rrs": rrs,
        "notes": notes, "size": size,
        "invalidation": sl,
    }


# --------------------------------------------------------------------------
# Market context
# --------------------------------------------------------------------------
def market_context(t4):
    df, c = t4["df"], t4["close"]
    s = t4["structure"]["label"]
    bull = int(s == "BULLISH") + int(c > t4["ema200"]) + int(c > t4["vwap"])
    bear = int(s == "BEARISH") + int(c < t4["ema200"]) + int(c < t4["vwap"])
    direction = "BULLISH" if bull >= 2 and bull > bear else \
                "BEARISH" if bear >= 2 and bear > bull else "SIDEWAYS"
    w = df["close"].iloc[-21:]
    er = abs(w.iloc[-1] - w.iloc[0]) / max(w.diff().abs().sum(), 1e-9)
    condition = "TRENDING" if er > 0.30 else "CONSOLIDATING / CHOPPY"
    last10 = df.iloc[-10:]
    up_vol = last10.loc[last10["close"] >= last10["open"], "volume"].sum()
    dn_vol = last10.loc[last10["close"] < last10["open"], "volume"].sum()
    control = "BUYERS" if up_vol > dn_vol * 1.15 else "SELLERS" if dn_vol > up_vol * 1.15 else "BALANCED"
    body = (df["close"] - df["open"]).abs()
    mom = "INCREASING" if body.iloc[-5:].mean() > body.iloc[-10:-5].mean() * 1.1 else \
          "DECREASING" if body.iloc[-5:].mean() < body.iloc[-10:-5].mean() * 0.9 else "STEADY"
    vol_trend = df["volume"].iloc[-5:].mean() / max(df["volume"].iloc[-25:-5].mean(), 1e-9)
    return {"direction": direction, "condition": condition, "control": control,
            "momentum": mom, "vol_trend": vol_trend, "bull": bull, "bear": bear}


def sr_levels(t4):
    c = t4["close"]
    sup = sorted({p for _, p in t4["pl"] if p < c}, reverse=True)[:3]
    res = sorted({p for _, p in t4["ph"] if p > c})[:3]
    return sup, res


# --------------------------------------------------------------------------
# Live stats, liquidity zones
# --------------------------------------------------------------------------
def fetch_24h(base):
    try:
        r = requests.get(f"{base}/api/v3/ticker/24hr", params={"symbol": SYMBOL}, timeout=10)
        r.raise_for_status()
        j = r.json()
        return {"price": float(j["lastPrice"]), "change_pct": float(j["priceChangePercent"]),
                "high": float(j["highPrice"]), "low": float(j["lowPrice"]),
                "vol_sol": float(j["volume"]), "vol_usdt": float(j["quoteVolume"])}
    except Exception:
        return None


def _cluster(prices, tol):
    """Group nearby swing levels; 2+ swings at ~the same price = equal highs/lows (liquidity pool)."""
    prices, groups, cur = sorted(prices), [], []
    for p in prices:
        if cur and p - cur[-1] > tol:
            groups.append(cur)
            cur = []
        cur.append(p)
    if cur:
        groups.append(cur)
    return [(sum(g) / len(g), len(g)) for g in groups if len(g) >= 2]


def liquidity_zones(t4):
    df, n, c = t4["df"], len(t4["df"]), t4["close"]
    tol = 0.25 * t4["atr"]
    hi = [p for i, p in t4["ph"] if i > n - 150]
    lo = [p for i, p in t4["pl"] if i > n - 150]
    eqh = sorted([(p, k) for p, k in _cluster(hi, tol) if p > c])[:3]
    eql = sorted([(p, k) for p, k in _cluster(lo, tol) if p < c], reverse=True)[:3]
    return {"eqh": eqh, "eql": eql,
            "range_high": df["high"].iloc[-50:].max(), "range_low": df["low"].iloc[-50:].min()}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def f(x):
    return "n/a" if x is None else f"{x:,.3f}"


def fvg_lines(t4, kind):
    items = [x for x in t4["fvgs"] if x["kind"] == kind and x["status"] != "fully mitigated"]
    items = sorted(items, key=lambda x: abs(t4["close"] - (x["lower"] + x["upper"]) / 2))[:3]
    if not items:
        return ["none un-mitigated nearby"]
    return [f"{x['lower']:,.3f} - {x['upper']:,.3f} ({x['status']}, formed {x['time']:%Y-%m-%d %H:%M} UTC)"
            for x in items]


def scenarios(t4, sup, res):
    s1 = f(sup[0]) if sup else "the nearest swing low"
    r1 = f(res[0]) if res else "the nearest swing high"
    bull = (f"Price holds {s1}, then a 4H candle CLOSES above {r1} (and above EMA 200 {f(t4['ema200'])} "
            f"/ VWAP {f(t4['vwap'])}) on rising volume. Invalidated by a 4H close below {s1}.")
    bear = (f"Price breaks {s1} on a 4H close, or sellers reject {r1} / EMA 200 / VWAP and print a lower "
            f"high. Invalidated by a 4H close above {r1} that holds.")
    return bull, bear


def sr_levels(t4):
    c = t4["close"]
    sup = sorted({p for _, p in t4["pl"] if p < c}, reverse=True)[:3]
    res = sorted({p for _, p in t4["ph"] if p > c})[:3]
    return sup, res


def print_setup(a):
    print(f"\n{a['direction']} SETUP CHECK  ->  status: {a['status']}  ({a['passes']}/5 core checks)")
    for k in ["EMA 200", "VWAP", "FVG", "Market Structure"]:
        print(f"  {k:<20}: {'PASS' if a['checks'][k] else 'FAIL'}")
    sw = a["sweep"]
    print(f"  {'Liquidity sweep':<20}: " +
          (f"YES (wick through {f(sw['level'])}, extreme {f(sw['extreme'])}, closed back inside)" if sw else "NO"))
    vr = a["candle"]["vol_ratio"]
    print(f"  {'Volume':<20}: {'PASS' if vr >= 1.0 else 'FAIL'} ({vr:.2f}x the 20-candle average, last closed 4H)")
    cc = "PASS" if a["checks"]["Candle Confirmation"] else "FAIL"
    print(f"  {'Candle confirmation':<20}: {cc}" + ("" if a["candle"]["strong"] else " (no strong candle in this direction)"))
    h = a["h"]
    print(f"  {'1H confirmation':<20}: {'PASS' if h['ok'] else 'FAIL'} "
          f"(1H structure {'CONFLICTS' if h['conflict'] else 'ok'}, 1H VWAP {'ok' if h['vwap_ok'] else 'against'})")
    print(f"  {'FVG state':<20}: {a['fvg']['state']}")
    for n in a["notes"]:
        print(f"  note: {n}")


def print_plan(a):
    print(f"\n{a['direction']} TRADE PLAN")
    ez = a["entry_zone"]
    if ez is None:
        print("  No entry zone can be defined (no un-mitigated FVG or sweep to anchor it). WAIT.")
        return
    if a["status"] == "ACTIVE SETUP":
        tag = "ACTIVE: enter on the confirmed close"
    elif a["status"] == "WAITING":
        tag = "CONDITIONAL: do NOT enter until a strong confirmed 4H candle + 1H alignment"
    else:
        tag = "REFERENCE ONLY, conditions conflict: NOT a signal"
    print(f"  [{tag}]")
    print(f"  Entry price/zone : {f(ez[0])} - {f(ez[1])}  (reference {f(a['entry_px'])})")
    print(f"  Stop-loss        : {f(a['sl'])}")
    for i, tp in enumerate(a["tps"], 1):
        print(f"  TP{i}              : {f(tp)}  (R:R {a['rrs'][i - 1]:.2f}:1)")
    print(f"  Risk-to-reward   : {a['rrs'][0]:.2f} : {a['rrs'][1]:.2f} : {a['rrs'][2]:.2f}  (to TP1 : TP2 : TP3)")
    print(f"  Invalidation     : 4H close beyond {f(a['invalidation'])} cancels this idea")
    if a["size"]:
        sz = a["size"]
        print(f"  Position size    : risk {sz['risk_amt']:,.2f} USDT ({RISK_PCT}%) -> {sz['units']:,.3f} SOL "
              f"(~{sz['notional']:,.2f} USDT notional)")


def decide(analyses):
    """LONG/SHORT only when exactly one direction is an ACTIVE SETUP. Otherwise WAIT."""
    active = [a for a in analyses if a["status"] == "ACTIVE SETUP"]
    if len(active) == 1:
        return active[0]["direction"], active[0]
    live = [a for a in analyses if a["status"] != "NO TRADE"]
    best = max(live, key=lambda a: a["passes"]) if live else None
    return "WAIT", best


def report(mode, t4, t1, ctx, analyses, meta):
    sup, res = sr_levels(t4)
    liq = liquidity_zones(t4)
    bull, bear = scenarios(t4, sup, res)
    line = "=" * 76
    st = meta["stats"]
    c4, c1 = t4["df"].iloc[-1], t1["df"].iloc[-1]
    price = meta["price"]

    print(line)
    print("SOL/USDT AI TRADING ANALYST  |  Binance  |  4H main, 1H confirmation")
    print(line)
    print(f"Data source : {meta['source']} | {meta['live']}")
    print(f"Fetched     : {meta['fetched']:%Y-%m-%d %H:%M:%S} UTC")
    print(f"Last CLOSED : 4H candle {c4['open_time']:%Y-%m-%d %H:%M} UTC | 1H candle {c1['open_time']:%Y-%m-%d %H:%M} UTC")
    print(f"Candles used: {len(t4['df'])} x 4H, {len(t1['df'])} x 1H (closed candles only)")

    print("\nLIVE MARKET DATA")
    print(f"  Live price       : {f(price) if price else 'n/a (offline CSV), last closed 4H close ' + f(t4['close'])}")
    if st:
        print(f"  24h change       : {st['change_pct']:+.2f}%   24h high/low {f(st['high'])} / {f(st['low'])}")
        print(f"  24h volume       : {st['vol_sol']:,.0f} SOL  (~{st['vol_usdt']:,.0f} USDT)")
    print(f"  Last 4H candle   : O {f(c4['open'])} H {f(c4['high'])} L {f(c4['low'])} C {f(c4['close'])} "
          f"Vol {c4['volume']:,.0f}")
    print(f"  Last 1H candle   : O {f(c1['open'])} H {f(c1['high'])} L {f(c1['low'])} C {f(c1['close'])} "
          f"Vol {c1['volume']:,.0f}")

    print("\nMARKET ANALYSIS")
    s, s1 = t4["structure"], t1["structure"]
    print(f"  Bullish/bearish/sideways : {ctx['direction']}  (market is {ctx['condition']})")
    print(f"  Buyers or sellers        : {ctx['control']} in control (last 10 candles, volume-weighted)")
    print(f"  Momentum                 : {ctx['momentum']}  (volume {ctx['vol_trend']:.2f}x vs prior 20 candles)")
    print(f"  4H structure             : {s['label']} - {s['detail']}" + (f" | {s['break']}" if s["break"] else ""))
    print(f"  1H structure             : {s1['label']} - {s1['detail']}" + (f" | {s1['break']}" if s1["break"] else ""))
    dist = (t4["close"] / t4["ema200"] - 1) * 100
    pos = "ABOVE" if dist > 0 else "BELOW"
    near = abs(t4["close"] - t4["ema200"]) <= t4["atr"]
    role = ("dynamic support" if dist > 0 else "dynamic resistance") if near else "not in play (far from price)"
    warm = "" if t4["ema_reliable"] else "  [WARNING: <200 candles, EMA not fully warmed up]"
    print(f"  EMA 200 (4H)             : {f(t4['ema200'])} | price {pos} by {abs(dist):.2f}% | {role}{warm}")
    print(f"  VWAP (4H, daily reset)   : {f(t4['vwap'])} | price {'ABOVE' if t4['close'] > t4['vwap'] else 'BELOW'}"
          f"   (1H VWAP {f(t1['vwap'])})")
    print("  Bullish FVG (4H)         : " + "; ".join(fvg_lines(t4, "bullish")))
    print("  Bearish FVG (4H)         : " + "; ".join(fvg_lines(t4, "bearish")))
    print("  Support                  : " + (", ".join(f(x) for x in sup) or "none found (price at range lows)"))
    print("  Resistance               : " + (", ".join(f(x) for x in res) or "none found (price at range highs)"))
    eqh = ", ".join(f"{f(p)} ({k} swings)" for p, k in liq["eqh"]) or "none"
    eql = ", ".join(f"{f(p)} ({k} swings)" for p, k in liq["eql"]) or "none"
    print(f"  Liquidity above          : equal highs {eqh}; 50-candle high {f(liq['range_high'])}")
    print(f"  Liquidity below          : equal lows {eql}; 50-candle low {f(liq['range_low'])}")
    print(f"  Bullish scenario         : {bull}")
    print(f"  Bearish scenario         : {bear}")
    likely = {"BULLISH": "bullish continuation; best entries come on pullbacks into a bullish FVG / VWAP.",
              "BEARISH": "bearish continuation; rallies into resistance / bearish FVGs are likely to be sold.",
              "SIDEWAYS": "range-bound and choppy; avoid trading the middle of the range."}[ctx["direction"]]
    print(f"  Most likely              : {likely}")
    print(f"  Watch next               : 4H close vs {f(res[0]) if res else 'resistance'} / "
          f"{f(sup[0]) if sup else 'support'}, EMA 200 {f(t4['ema200'])}, VWAP {f(t4['vwap'])}, "
          f"and whether volume expands on the break.")

    for a in analyses:
        print_setup(a)
    for a in analyses:
        print_plan(a)

    decision, best = decide(analyses)
    print("\n" + line)
    print("FINAL REPORT")
    print(line)
    print(f" 1. Current market price : {f(price) if price else f(t4['close']) + ' (last closed 4H close, offline)'}")
    print(f" 2. Market direction     : {ctx['direction']}")
    print(f" 3. Market condition     : {ctx['condition']}; {ctx['control']} in control; momentum {ctx['momentum']}")
    print(f" 4. EMA 200              : {f(t4['ema200'])} (price {pos} by {abs(dist):.2f}%)")
    print(f" 5. VWAP                 : {f(t4['vwap'])} (price {'ABOVE' if t4['close'] > t4['vwap'] else 'BELOW'})")
    bf = fvg_lines(t4, "bullish")[0]
    brf = fvg_lines(t4, "bearish")[0]
    print(f" 6. Bullish / bearish FVG: {bf}  ||  {brf}")
    print(f" 7. Support / resistance : {', '.join(f(x) for x in sup) or 'n/a'}  /  {', '.join(f(x) for x in res) or 'n/a'}")
    print(f" 8. Bullish scenario     : {bull}")
    print(f" 9. Bearish scenario     : {bear}")
    if best is None:
        print("10. Best opportunity     : none. No direction has enough aligned conditions right now.")
        print("11. Entry, SL, targets   : none issued")
    else:
        print(f"10. Best opportunity     : {best['direction']} ({best['status']}, {best['passes']}/5 checks)")
        if best["entry_zone"]:
            ez = best["entry_zone"]
            tps = " / ".join(f(t) for t in best["tps"])
            print(f"11. Entry, SL, targets   : entry {f(ez[0])}-{f(ez[1])} | SL {f(best['sl'])} | TP {tps} | "
                  f"R:R {best['rrs'][0]:.1f}/{best['rrs'][1]:.1f}/{best['rrs'][2]:.1f}"
                  + ("" if best["status"] == "ACTIVE SETUP" else "  (CONDITIONAL, not active)"))
        else:
            print("11. Entry, SL, targets   : no defined zone yet")
    print(f"12. FINAL DECISION       : {decision}")
    if decision == "WAIT":
        reasons = []
        for a in analyses:
            reasons.append(f"{a['direction']} is {a['status']} ({a['passes']}/5)")
        print("    Reason: " + "; ".join(reasons) + ". Conditions are not aligned enough. Do not force a trade.")
        if best is not None:
            print(f"    Trigger to watch: strong confirmed 4H {best['direction']} candle at the zone with volume "
                  f"and 1H alignment. Cancel if 4H closes beyond {f(best['invalidation'])}.")
    else:
        print(f"    {decision} is an ACTIVE SETUP. Risk at most {RISK_PCT}% of capital, stop at {f(best['sl'])}.")
    print("    No target is guaranteed. Not financial advice. Capital preservation comes first.")
    print(line)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def ask_mode():
    print("What would you like to analyse?\n")
    print("  1. LONG")
    print("  2. SHORT")
    print("  3. FULL MARKET ANALYSIS   (press Enter for this)")
    while True:
        a = input("\nChoose 1 / 2 / 3: ").strip().lower()
        if a in ("1", "long", "l"):
            return "long"
        if a in ("2", "short", "s"):
            return "short"
        if a in ("3", "full", "f", ""):
            return "full"
        print("Please enter 1, 2 or 3.")


def main():
    ap = argparse.ArgumentParser(description="SOL/USDT analyst: 4H main, 1H confirmation")
    ap.add_argument("--mode", choices=["long", "short", "full"],
                    help="skip the question (otherwise you are asked every run)")
    ap.add_argument("--csv4h", help="4H OHLCV CSV (offline mode)")
    ap.add_argument("--csv1h", help="1H OHLCV CSV (offline mode)")
    ap.add_argument("--capital", type=float, help="account size in USDT for 1%% risk position sizing")
    args = ap.parse_args()

    mode = args.mode or ask_mode()
    if bool(args.csv4h) != bool(args.csv1h):
        sys.exit("Provide BOTH --csv4h and --csv1h, or neither for live Binance data.")
    offline = bool(args.csv4h)

    try:
        if offline:
            d4, d1 = load_csv(args.csv4h, "4h"), load_csv(args.csv1h, "1h")
            stats, price = None, None
            source, live = "user-supplied CSV", "OFFLINE data (NOT live)"
        else:
            d4, base = fetch_klines("4h", 1000)
            d1, _ = fetch_klines("1h", 500)
            stats = fetch_24h(base)
            price = stats["price"] if stats else fetch_price(base)
            source, live = f"Binance REST ({base})", "LIVE data (fetched just now)"
    except Exception as e:
        sys.exit(f"DATA UNAVAILABLE: {e}\nI will not invent market data. "
                 f"Check your connection/region, or use --csv4h / --csv1h.")

    d4, d1 = closed_only(d4, offline), closed_only(d1, offline)
    if len(d4) < 60 or len(d1) < 60:
        sys.exit("Not enough closed candles (need at least 60 per timeframe; 250+ 4H candles is ideal).")

    t4, t1 = analyse_tf(d4), analyse_tf(d1)
    ctx = market_context(t4)
    dirs = ["long", "short"] if mode == "full" else [mode]
    analyses = [analyse_direction(d, t4, t1, args.capital) for d in dirs]
    meta = {"source": source, "live": live, "price": price, "stats": stats,
            "fetched": dt.datetime.now(dt.timezone.utc)}
    report(mode, t4, t1, ctx, analyses, meta)


if __name__ == "__main__":
    main()
