# src/evolution/primitives.py
# Market Microstructure Feature Library untuk DEAP Primitive Set
# DILARANG: RSI, MACD, MA, Bollinger, Stochastic, ATR, CCI

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

# ── Type alias ───────────────────────────────────────────────────────────
Arr = NDArray[np.float64]


# ══════════════════════════════════════════════════════════════════════════
# KELOMPOK 1: OPERASI MATEMATIKA MURNI
# Digunakan sebagai node internal pohon GP untuk komposisi
# ══════════════════════════════════════════════════════════════════════════

def safe_div(a: float, b: float) -> float:
    """Protected division — return 1.0 jika denominator ~ 0."""
    return a / b if abs(b) > 1e-10 else 1.0

def safe_log(x: float) -> float:
    """Protected log — return 0.0 jika x <= 0."""
    return np.log(abs(x)) if abs(x) > 1e-10 else 0.0

def safe_sqrt(x: float) -> float:
    """Protected sqrt — return 0.0 jika x < 0."""
    return np.sqrt(abs(x))

def neg(x: float) -> float:
    return -x

def square(x: float) -> float:
    return x * x

def cube(x: float) -> float:
    return x * x * x

def sigmoid(x: float) -> float:
    """Squash ke (0, 1) — berguna untuk normalisasi signal."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sign_fn(x: float) -> float:
    """Return 1.0, -1.0, atau 0.0."""
    return float(np.sign(x))

def max2(a: float, b: float) -> float:
    return max(a, b)

def min2(a: float, b: float) -> float:
    return min(a, b)


# ══════════════════════════════════════════════════════════════════════════
# KELOMPOK 2: MARKET MICROSTRUCTURE FEATURES
# Dihitung dari tick-level data: bid, ask, bid_size, ask_size
# ══════════════════════════════════════════════════════════════════════════

def order_book_imbalance(bid_size: Arr, ask_size: Arr, window: int = 20) -> float:
    """
    Order Book Imbalance (OBI) — mengukur tekanan beli vs jual.

    Formula: OBI = (BidVol - AskVol) / (BidVol + AskVol)
    Range: [-1, +1]. Positif = tekanan beli dominan.

    Args:
        bid_size, ask_size: array tick-level volume
        window: jumlah tick terakhir yang digunakan

    Returns:
        float scalar OBI saat ini
    """
    w = min(window, len(bid_size))
    # Cast before summing because ClickHouse volumes arrive as UInt32.  Keeping
    # the unsigned dtype can overflow on total_bid - total_ask.
    total_bid = float(np.sum(bid_size[-w:].astype(np.float64)))
    total_ask = float(np.sum(ask_size[-w:].astype(np.float64)))
    total = total_bid + total_ask
    return float((total_bid - total_ask) / total) if total > 0 else 0.0


def tick_velocity(bid: Arr, ask: Arr, window: int = 10) -> float:
    """
    Tick Velocity — rata-rata perubahan mid-price per tick.

    Formula: mean(|delta(mid_price)|) untuk window tick terakhir.
    Mengukur kecepatan pergerakan harga, bukan arahnya.

    Returns:
        float >= 0, dalam satuan pips (dikalikan 10000 untuk FX)
    """
    w = min(window, len(bid))
    mid = (bid[-w:] + ask[-w:]) / 2.0
    if len(mid) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(mid))) * 10000)  # konversi ke pips


def spread_dynamics(bid: Arr, ask: Arr, window: int = 20) -> float:
    """
    Spread Dynamics — volatilitas spread dalam window terakhir.

    Formula: std(ask - bid) / mean(ask - bid)
    Nilai tinggi = spread tidak stabil (likuiditas buruk).
    Nilai rendah = spread stabil (likuiditas baik).

    Returns:
        float CoV (Coefficient of Variation) dari spread
    """
    w = min(window, len(bid))
    spread = ask[-w:] - bid[-w:]
    mean_s = np.mean(spread)
    std_s = np.std(spread)
    return float(std_s / mean_s) if mean_s > 1e-10 else 0.0


def tick_density(bid: Arr, ask: Arr, timestamps: Arr, window_sec: float = 60.0) -> float:
    """
    Tick Density — jumlah tick per detik dalam window terakhir.

    Mengukur market activity level. Tinggi = banyak quote update,
    menandakan price discovery aktif atau news event.

    Args:
        timestamps: UNIX timestamps dalam detik
        window_sec: durasi window dalam detik

    Returns:
        float ticks/second
    """
    if len(timestamps) < 2:
        return 0.0
    t_now = timestamps[-1]
    mask = timestamps >= (t_now - window_sec)
    count = int(np.sum(mask))
    return float(count / window_sec)


def volume_clock_skew(bid_size: Arr, ask_size: Arr, window: int = 30) -> float:
    """
    Volume Clock Skew — asimetri distribusi volume bid vs ask.

    Formula: (BidVol_std - AskVol_std) / (BidVol_std + AskVol_std)
    Mengukur seberapa 'tidak seimbang' volatilitas volume di kedua sisi.

    Returns:
        float [-1, +1]
    """
    w = min(window, len(bid_size))
    bid_std = np.std(bid_size[-w:])
    ask_std = np.std(ask_size[-w:])
    total = bid_std + ask_std
    return float((bid_std - ask_std) / total) if total > 1e-10 else 0.0


def mid_price_momentum(bid: Arr, ask: Arr, short_w: int = 5, long_w: int = 20) -> float:
    """
    Mid-Price Momentum — selisih rata-rata mid-price dua window.

    Formula: mean(mid[-short_w:]) - mean(mid[-long_w:])
    Murni dari price tick, bukan dari OHLCV candle.

    Returns:
        float dalam satuan pips
    """
    mid = (bid + ask) / 2.0
    sw = min(short_w, len(mid))
    lw = min(long_w, len(mid))
    return float((np.mean(mid[-sw:]) - np.mean(mid[-lw:])) * 10000)


# ══════════════════════════════════════════════════════════════════════════
# KELOMPOK 3: STATISTICAL MOMENTS
# Higher-order statistics dari return distribution
# ══════════════════════════════════════════════════════════════════════════

def rolling_skewness(bid: Arr, ask: Arr, window: int = 30) -> float:
    """
    Skewness dari distribusi mid-price returns.
    Positif = ekor kanan tebal (outlier positif lebih besar).
    """
    mid = (bid + ask) / 2.0
    w = min(window, len(mid))
    returns = np.diff(mid[-w:])
    if len(returns) < 3:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    if std_r < 1e-10:
        return 0.0
    return float(np.mean(((returns - mean_r) / std_r) ** 3))


def rolling_kurtosis(bid: Arr, ask: Arr, window: int = 30) -> float:
    """
    Excess Kurtosis dari distribusi mid-price returns.
    Positif = heavy-tailed (lebih banyak outlier dari normal).
    """
    mid = (bid + ask) / 2.0
    w = min(window, len(mid))
    returns = np.diff(mid[-w:])
    if len(returns) < 4:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns)
    if std_r < 1e-10:
        return 0.0
    return float(np.mean(((returns - mean_r) / std_r) ** 4) - 3.0)  # excess


def volume_weighted_spread(bid: Arr, ask: Arr, bid_size: Arr, ask_size: Arr,
                           window: int = 20) -> float:
    """
    Volume-Weighted Effective Spread.
    Spread yang lebih besar dari sisi volume yang lebih kecil = likuiditas timpang.
    """
    w = min(window, len(bid))
    spread = (ask[-w:] - bid[-w:]) * 10000  # dalam pips
    total_vol = bid_size[-w:] + ask_size[-w:]
    total_vol = np.where(total_vol < 1e-10, 1.0, total_vol)
    vw_spread = np.sum(spread * total_vol) / np.sum(total_vol)
    return float(vw_spread)


# ── Registry untuk DEAP ──────────────────────────────────────────────────
# Gunakan dict ini di DEAPGenerator untuk mendaftarkan semua primitives

BINARY_OPS = [
    ('add',     lambda a,b: a+b,  2),
    ('sub',     lambda a,b: a-b,  2),
    ('mul',     lambda a,b: a*b,  2),
    ('div',     safe_div,         2),
    ('max2',    max2,             2),
    ('min2',    min2,             2),
]

UNARY_OPS = [
    ('neg',     neg,      1),
    ('square',  square,   1),
    ('cube',    cube,     1),
    ('log',     safe_log, 1),
    ('sqrt',    safe_sqrt,1),
    ('sigmoid', sigmoid,  1),
    ('sign',    sign_fn,  1),
]

MICROSTRUCTURE_FEATURES = [
    'obi',          # order_book_imbalance
    'tick_vel',     # tick_velocity
    'spread_dyn',   # spread_dynamics
    'tick_den',     # tick_density
    'vol_skew',     # volume_clock_skew
    'mid_mom',      # mid_price_momentum
    'skewness',     # rolling_skewness
    'kurtosis',     # rolling_kurtosis
    'vw_spread',    # volume_weighted_spread
]
