import pandas as pd
import numpy as np

# ---- Indicators ----

def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s, n):
    """Hull Moving Average."""
    n2 = int(max(2, n // 2))
    sqrt_n = int(max(2, int(np.sqrt(n))))
    wma1 = wma(s, n2)
    wma2 = wma(s, n)
    return wma(2 * wma1 - wma2, sqrt_n)


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = ema(up, n) / ema(down, n)
    return 100 - (100 / (1 + rs))


def stoch_k(high, low, close, n=14):
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    return 100 * (close - ll) / (hh - ll)


def stoch_rsi(close, n_rsi=14, n_stoch=14):
    r = rsi(close, n_rsi)
    r_min = r.rolling(n_stoch).min()
    r_max = r.rolling(n_stoch).max()
    return 100 * (r - r_min) / (r_max - r_min)


def cci(high, low, close, n=20):
    tp = (high + low + close) / 3
    sma_tp = sma(tp, n)
    md = (tp - sma_tp).abs().rolling(n).mean()
    return (tp - sma_tp) / (0.015 * md)


def williams_r(high, low, close, n=14):
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return -100 * (hh - close) / (hh - ll)


def momentum(close, n=10):
    return close - close.shift(n)


def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def ao(high, low):
    median = (high + low) / 2
    return sma(median, 5) - sma(median, 34)


def bulls_bears_power(high, low, close, n=13):
    e = ema(close, n)
    bulls = high - e
    bears = low - e
    return bulls, bears


def ultimate_osc(high, low, close, s1=7, s2=14, s3=28):
    bp = close - np.minimum(low, close.shift(1))
    tr = np.maximum(high, close.shift(1)) - np.minimum(low, close.shift(1))
    avg1 = bp.rolling(s1).sum() / tr.rolling(s1).sum()
    avg2 = bp.rolling(s2).sum() / tr.rolling(s2).sum()
    avg3 = bp.rolling(s3).sum() / tr.rolling(s3).sum()
    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


def adx(high, low, close, n=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(n).sum() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(n).sum() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(n).mean(), plus_di, minus_di


def vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    cum_vol = volume.cumsum()
    cum_pv = (tp * volume).cumsum()
    return cum_pv / cum_vol


def ichimoku_kijun(high, low, n=26):
    return (high.rolling(n).max() + low.rolling(n).min()) / 2


# ---- Feature builder ----
def build_features(df):
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
    feats = pd.DataFrame(index=df.index)
    feats['rsi'] = rsi(c, 14)
    feats['stoch_k'] = stoch_k(h, l, c, 14)
    feats['stoch_rsi'] = stoch_rsi(c, 14, 14)
    feats['cci'] = cci(h, l, c, 20)
    feats['willr'] = williams_r(h, l, c, 14)
    feats['mom10'] = momentum(c, 10)
    macd_line, macd_sig, macd_hist = macd(c)
    feats['macd'] = macd_line
    feats['macd_sig'] = macd_sig
    feats['macd_hist'] = macd_hist
    feats['ao'] = ao(h, l)
    bulls, bears = bulls_bears_power(h, l, c)
    feats['bulls'] = bulls
    feats['bears'] = bears
    feats['uo'] = ultimate_osc(h, l, c)
    adx_val, plus_di, minus_di = adx(h, l, c)
    feats['adx'] = adx_val
    feats['plus_di'] = plus_di
    feats['minus_di'] = minus_di
    feats['ema50'] = ema(c, 50)
    feats['ema200'] = ema(c, 200)
    feats['sma20'] = sma(c, 20)
    feats['hma55'] = hma(c, 55)
    feats['vwap'] = vwap(h, l, c, v)
    feats['kijun'] = ichimoku_kijun(h, l, 26)
    feats['atr14'] = (h - l).rolling(14).mean()  # simplified ATR
    return feats


# ---- Regime classifier ----
def regime(feats):
    trend = (feats['adx'] > 25) & (feats['ema50'] > feats['ema200'])
    return np.where(trend, 'trend', 'range')


# ---- Signal logic ----
def signals(df):
    feats = build_features(df)
    c = df['close']
    reg = pd.Series(regime(feats), index=df.index)
    sig = pd.Series(0, index=df.index)  # +1 long, -1 short, 0 flat

    cond_long = (
        (reg == 'trend') &
        (c > feats['ema50']) &
        (feats['macd_hist'] > 0) &
        (feats['stoch_rsi'] > 60) &
        (c > c.shift(1))
    )

    cond_short = (
        (reg == 'range') &
        (feats['willr'] > -20) &
        (feats['stoch_k'] > 80) &
        (feats['rsi'] > 70)
    )

    sig[cond_long] = 1
    sig[cond_short] = -1
    return sig, feats


# ---- Triple-barrier SL/TP ----
def barriers(df, feats, risk_R=1.0):
    atr = feats['atr14'].fillna(method='bfill')
    sl = df['close'] - risk_R * atr
    tp = df['close'] + 1.5 * risk_R * atr
    return sl, tp

