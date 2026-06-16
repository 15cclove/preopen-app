"""
analysis.py — 法科盤前方向引擎（交易研究版 v0.2）
==================================================================
相對 v0.1 的調整（依三輪 review 共識）：
  1. 資料層抽成 DataSource 介面 → 之後接 TXF 只需實作一個類別，其餘不動
  2. 加入指數期貨隔夜特徵 (NQ/ES) 與匯率，並標記「夜盤是否已 price-in」
     → TXF 模式可一鍵切成「殘差特徵集」，避免重複計入隔夜美股
  3. 績效從「方向準確率」升級為交易指標：
     期望值 / Profit Factor / 最大回撤 / Sharpe / 賺賠比 / 交易次數，且全部扣成本
  4. 信心分位數報酬表：驗證「機率越高，報酬是否越好」
  5. L2 正則化強度 C 用 TimeSeriesSplit CV 自動選（取代固定 C）
  6. Benjamini-Hochberg FDR：在「選因子」階段控制假顯著
  7. 移除共整合（對方向模型無價值；保留為獨立 stat-arb 診斷的插孔）
  8. 內建 kill criterion：高信心區間扣成本後若無正期望值 → 明確判定不該交易

★ 代理版仍用 ^TWII。它只驗「架構與統計價值」，不等於 TXF 能不能賺錢。
  換 TXF：實作 TXFDataSource 並把 ACTIVE_SOURCE 指過去即可。
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ==================================================================
# 設定
# ==================================================================
START = "2018-01-01"
TARGET = "^TWII"            # 代理；換 TXF 時改這裡與 ACTIVE_SOURCE

# 交易假設（log-return 比例；TXF 請把點數換算成佔指數的比例）
COST_RT = 0.0006           # 來回成本（手續費+稅+價差），約 6bps
SLIP_OPEN = 0.0004         # 開盤額外滑價（盤前模型在 08:45 附近成交較差）
THR = 0.55                 # 進場門檻：p_up>=THR 做多、<=1-THR 做空、其餘空手

# TXF 模式：夜盤已消化隔夜美股 → 設 True 自動丟掉 priced_in 特徵，只留殘差
RESIDUAL_ONLY = False

# 預測變數：ticker, 顯示名, 群組, 報酬型(cc=收盤對收盤 / gap=開盤跳空), 夜盤是否已price-in
PREDICTORS = [
    ("TSM",       "台積電 ADR",     "美股現貨", "cc",  True),
    ("^SOX",      "費城半導體",     "美股現貨", "cc",  True),
    ("^NDX",      "那斯達克100",    "美股現貨", "cc",  True),
    ("^GSPC",     "標普500",        "美股現貨", "cc",  True),
    ("NVDA",      "輝達",           "美股現貨", "cc",  True),
    ("NQ=F",      "NQ 期貨隔夜",    "指數期貨", "cc",  False),
    ("ES=F",      "ES 期貨隔夜",    "指數期貨", "cc",  False),
    ("^VIX",      "VIX 波動率",     "風險",     "cc",  True),
    ("DX-Y.NYB",  "美元指數",       "風險",     "cc",  True),
    ("TWD=X",     "美元兌台幣",     "風險",     "cc",  False),
    ("^KS11",     "KOSPI 開盤跳空", "亞洲早盤", "gap", False),
    ("^N225",     "日經 開盤跳空",  "亞洲早盤", "gap", False),
]
NAME = {p[0]: p[1] for p in PREDICTORS}
GROUP = {p[0]: p[2] for p in PREDICTORS}
RET = {p[0]: p[3] for p in PREDICTORS}
PRICED_IN = {p[0]: p[4] for p in PREDICTORS}


# ==================================================================
# 資料層介面（換 TXF 的唯一接點）
# ==================================================================
class DataSource:
    """所有資料來源需實作 daily(ticker) → 含 Open/High/Low/Close 的日線 DataFrame。"""
    def daily(self, ticker, force=False):
        raise NotImplementedError


class YahooDataSource(DataSource):
    def __init__(self, ttl=1800):
        import yfinance as yf
        self.yf = yf
        self.ttl = ttl
        self.dir = os.path.join(os.path.dirname(__file__), ".cache")
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, t):
        safe = t.replace("^", "_").replace("=", "_").replace(".", "_")
        return os.path.join(self.dir, f"{safe}.pkl")

    def daily(self, ticker, force=False):
        p = self._path(ticker)
        if (not force) and os.path.exists(p) and time.time() - os.path.getmtime(p) < self.ttl:
            return pickle.load(open(p, "rb"))
        df = self.yf.download(ticker, start=START, progress=False, auto_adjust=False)
        if df is None or len(df) == 0:
            raise ValueError(f"抓不到 {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        pickle.dump(df, open(p, "wb"))
        return df

    def clear(self):
        for f in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, f))


class TXFDataSource(DataSource):
    """台指期資料源（FinMind 版）。
    daily("TX") 回傳近月合約日盤 OHLC（Open=08:45、Close=13:45），格式與 YahooDataSource 相同。
    其他 ticker 仍 fallback 到 YahooDataSource，讓預測特徵（美股/VIX 等）照常運作。
    """
    def __init__(self, ttl=3600):
        self.ttl = ttl
        self.dir = os.path.join(os.path.dirname(__file__), ".cache")
        os.makedirs(self.dir, exist_ok=True)
        self._yahoo = YahooDataSource(ttl=ttl)

    def _txf_path(self):
        return os.path.join(self.dir, "TXF_finmind.pkl")

    def _load_txf(self, force=False):
        p = self._txf_path()
        if (not force) and os.path.exists(p) and time.time() - os.path.getmtime(p) < self.ttl:
            return pickle.load(open(p, "rb"))
        try:
            from FinMind.data import DataLoader
        except ImportError:
            raise ImportError("請先安裝 FinMind：pip3 install FinMind")
        dl = DataLoader()
        raw = dl.taiwan_futures_daily(futures_id="TX", start_date=START)
        # 只留日盤、只留近月（contract_date 最小 = 近月）
        raw = raw[raw["trading_session"] == "position"]
        raw = raw[raw["settlement_price"] > 0]
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw[raw["contract_date"] == raw.groupby("date")["contract_date"].transform("min")]
        raw = raw.set_index("date").sort_index()
        df = raw[["open", "max", "min", "close"]].copy()
        df.columns = ["Open", "High", "Low", "Close"]
        df = df.dropna()
        pickle.dump(df, open(p, "wb"))
        return df

    def daily(self, ticker, force=False):
        if ticker in ("TX", "TXF", TARGET) and TARGET != "^TWII":
            return self._load_txf(force=force)
        # 其餘特徵（美股 / VIX 等）繼續用 Yahoo
        return self._yahoo.daily(ticker, force=force)

    def clear(self):
        p = self._txf_path()
        if os.path.exists(p):
            os.remove(p)
        self._yahoo.clear()


# ==================================================================
# 資料源切換：True = 真實台指期（FinMind），False = 代理版（^TWII / Yahoo）
# ==================================================================
USE_TXF = True

if USE_TXF:
    ACTIVE_SOURCE = TXFDataSource()
    TARGET = "TX"
    RESIDUAL_ONLY = True
else:
    ACTIVE_SOURCE = YahooDataSource()


def fetch(ticker, force=False):
    return ACTIVE_SOURCE.daily(ticker, force=force)


def clear_cache():
    if hasattr(ACTIVE_SOURCE, "clear"):
        ACTIVE_SOURCE.clear()


# ==================================================================
# 報酬與目標
# ==================================================================
def _ret(df, kind):
    if kind == "gap":
        return np.log(df["Open"] / df["Close"].shift(1)).rename("r")
    return np.log(df["Close"] / df["Close"].shift(1)).rename("r")


def build_targets(df):
    out = pd.DataFrame(index=df.index)
    out["gap"] = np.log(df["Open"] / df["Close"].shift(1))
    out["day"] = np.log(df["Close"] / df["Close"].shift(1))
    out["intraday"] = np.log(df["Close"] / df["Open"])     # 主目標：開盤後可交易方向
    return out.dropna()


def active_predictors():
    """套用 RESIDUAL_ONLY 過濾。"""
    return [p[0] for p in PREDICTORS if (not RESIDUAL_ONLY) or (not PRICED_IN[p[0]])]


# ==================================================================
# 對齊（無 look-ahead）：cc→shift+1 台股日；gap→同日
# ==================================================================
def assemble(force=False):
    tdf = fetch(TARGET, force)
    targets = build_targets(tdf).reset_index()
    targets.columns = ["date"] + list(targets.columns[1:])
    feats = targets[["date"]].copy()
    available = []
    for t in active_predictors():
        try:
            r = _ret(fetch(t, force), RET[t]).dropna().reset_index()
        except Exception as e:
            print(f"[skip] {t}: {e}")
            continue
        r.columns = ["date", t]
        if RET[t] == "cc":
            r["date"] = r["date"] + pd.Timedelta(days=1)   # 隔日台股早上才可用
        feats = pd.merge_asof(feats.sort_values("date"), r.sort_values("date"),
                              on="date", direction="backward", tolerance=pd.Timedelta(days=4))
        available.append(t)
    data = targets.merge(feats, on="date").set_index("date").dropna()
    return data, available


# ==================================================================
# 樣本外預測（walk-forward expanding，每折用 CV 選 C）
# ==================================================================
def _fit_clf(Xtr, ytr):
    cv = TimeSeriesSplit(n_splits=3)
    return LogisticRegressionCV(Cs=[0.03, 0.1, 0.3, 1.0, 3.0], cv=cv,
                                penalty="l2", scoring="accuracy",
                                max_iter=2000).fit(Xtr, ytr)


def walk_forward_oos(data, feats, tgt, n_splits=5):
    """回傳整段樣本外的 DataFrame：p_up, y_ret（目標原始 log 報酬）。"""
    X = data[feats].values
    y_ret = data[tgt].values
    y = (y_ret > 0).astype(int)
    n = len(y)
    fold = n // (n_splits + 1)
    rows = []
    for i in range(1, n_splits + 1):
        tr, te = slice(0, fold * i), slice(fold * i, fold * (i + 1))
        if len(np.unique(y[tr])) < 2 or (te.stop - te.start) == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = _fit_clf(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        for pu, yr in zip(p, y_ret[te]):
            rows.append((float(pu), float(yr)))
    return pd.DataFrame(rows, columns=["p_up", "y_ret"])


# ==================================================================
# 交易績效（全部扣成本）
# ==================================================================
def _positions(oos, thr=THR):
    p = oos["p_up"].values
    pos = np.where(p >= thr, 1, np.where(p <= 1 - thr, -1, 0))
    gross = pos * oos["y_ret"].values
    cost = (pos != 0) * (COST_RT + SLIP_OPEN)
    net = gross - cost
    return pos, net


def _max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def trading_metrics(oos, thr=THR):
    pos, net = _positions(oos, thr)
    active = net[pos != 0]
    days = len(net)
    if len(active) == 0:
        return {"n_trades": 0, "note": "門檻下無進場"}
    wins, losses = active[active > 0], active[active < 0]
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None
    equity = np.cumsum(net)
    sharpe = float(np.mean(net) / np.std(net) * np.sqrt(252)) if np.std(net) > 0 else None
    return {
        "n_trades": int(len(active)),
        "trade_rate": round(len(active) / days, 3),
        "expectancy_bps": round(float(active.mean()) * 1e4, 2),
        "win_rate": round(float((active > 0).mean()), 3),
        "avg_win_bps": round(float(wins.mean()) * 1e4, 2) if len(wins) else 0,
        "avg_loss_bps": round(float(losses.mean()) * 1e4, 2) if len(losses) else 0,
        "payoff": round(float(wins.mean() / -losses.mean()), 2) if len(wins) and len(losses) else None,
        "profit_factor": round(pf, 2) if pf is not None else None,
        "max_dd_bps": round(_max_dd(equity) * 1e4, 1),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "total_ret_bps": round(float(net.sum()) * 1e4, 1),
    }


# ==================================================================
# 信心分位數報酬表：機率越高，報酬是否越好？（毛報酬，純看訊號品質）
# ==================================================================
def confidence_table(oos):
    edges = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.01]
    labels = ["<0.40", "0.40-0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60", "0.60-0.65", ">=0.65"]
    rows = []
    p = oos["p_up"].values
    yr = oos["y_ret"].values
    signed = np.sign(p - 0.5) * yr  # 照模型方向取的毛報酬
    for i, lab in enumerate(labels):
        mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.sum() == 0:
            rows.append({"bucket": lab, "n": 0})
            continue
        rows.append({"bucket": lab, "n": int(mask.sum()),
                     "hit_rate": round(float((signed[mask] > 0).mean()), 3),
                     "mean_bps": round(float(signed[mask].mean()) * 1e4, 2)})
    return rows


# ==================================================================
# Benjamini-Hochberg FDR
# ==================================================================
def bh_fdr(pvals, alpha=0.05):
    p = np.array([x if x is not None else 1.0 for x in pvals], float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    sig = np.zeros(n, bool)
    if passed.any():
        kmax = int(np.max(np.where(passed)[0]))
        sig[order[:kmax + 1]] = True
    q_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_ranked, 0, 1)
    return sig.tolist(), [round(float(x), 4) for x in q]


# ==================================================================
# 相關性（含 FDR 標記）
# ==================================================================
def correlations(data, feats, tgt):
    y = data[tgt]
    rows, pvals = [], []
    for c in feats:
        pear, pp = stats.pearsonr(data[c], y)
        spear, _ = stats.spearmanr(data[c], y)
        rows.append({"ticker": c, "name": NAME[c], "group": GROUP[c],
                     "pearson": round(float(pear), 3), "p": round(float(pp), 4),
                     "spearman": round(float(spear), 3)})
        pvals.append(float(pp))
    sig, q = bh_fdr(pvals)
    for r, s, qq in zip(rows, sig, q):
        r["fdr_sig"], r["q"] = bool(s), qq
    rows.sort(key=lambda r: abs(r["pearson"]), reverse=True)
    return rows


# ==================================================================
# CCF / Granger 診斷（原始同曆日；lag>0 = 該標的領先台股，與計算邏輯一致）
# ==================================================================
def _raw_matrix(feats):
    cols = {TARGET: _ret(fetch(TARGET), "cc")}
    for t in feats:
        cols[t] = _ret(fetch(t), RET[t])
    m = pd.concat(cols, axis=1).dropna()
    m.columns = [TARGET] + feats
    return m


def ccf_diag(feats, max_lag=3):
    m = _raw_matrix(feats)
    y = m[TARGET]
    out = []
    for t in feats:
        s = {}
        for k in range(-max_lag, max_lag + 1):
            j = pd.concat([m[t].shift(k), y], axis=1).dropna()  # shift(k>0)=該標的領先
            s[k] = j.iloc[:, 0].corr(j.iloc[:, 1]) if len(j) > 20 else np.nan
        best = max(s, key=lambda k: abs(s[k]) if not np.isnan(s[k]) else -1)
        out.append({"ticker": t, "name": NAME[t], "best_lag": int(best),
                    "best_r": round(float(s[best]), 3),
                    "lag0": round(float(s[0]), 3), "lag1": round(float(s[1]), 3),
                    "leads": bool(best > 0)})
    return out


def granger_diag(feats, maxlag=2):
    m = _raw_matrix(feats)
    y = m[TARGET]
    out = []
    for t in feats:
        df2 = pd.concat([y, m[t]], axis=1).dropna()
        try:
            res = grangercausalitytests(df2.values, maxlag=maxlag, verbose=False)
            pv = [round(float(res[l][0]["ssr_ftest"][1]), 4) for l in range(1, maxlag + 1)]
        except Exception:
            pv = [None] * maxlag
        out.append({"ticker": t, "name": NAME[t], "pvals": pv,
                    "leads": any(p is not None and p < 0.05 for p in pv)})
    return out


# ==================================================================
# kill criterion：高信心區間扣成本後是否值得交易
# ==================================================================
def kill_criterion(oos, min_trades=40):
    hi = oos[(oos["p_up"] >= 0.60) | (oos["p_up"] <= 0.40)].copy()
    if len(hi) < min_trades:
        return {"verdict": "資料不足", "pass": False,
                "reason": f"高信心樣本僅 {len(hi)} 筆（<{min_trades}）", "n": int(len(hi))}
    m = trading_metrics(hi, thr=0.60)
    ok = (m.get("expectancy_bps", -1) > 0) and ((m.get("profit_factor") or 0) > 1.0)
    return {"verdict": "通過初步檢核（值得繼續開發）" if ok else "未通過（高信心扣成本後無正期望值）",
            "pass": bool(ok), "n": int(len(hi)),
            "expectancy_bps": m.get("expectancy_bps"),
            "profit_factor": m.get("profit_factor")}


# ==================================================================
# 對外：完整研究報告
# ==================================================================
def run_research(force=False):
    data, feats = assemble(force)
    targets = {}
    for tgt in ["gap", "day", "intraday"]:
        oos = walk_forward_oos(data, feats, tgt)
        targets[tgt] = {
            "correlations": correlations(data, feats, tgt),
            "metrics": trading_metrics(oos),
            "confidence": confidence_table(oos),
            "kill": kill_criterion(oos),
        }
    return {
        "sample": {"n": int(len(data)), "features": feats,
                   "residual_only": RESIDUAL_ONLY,
                   "start": str(data.index.min().date()),
                   "end": str(data.index.max().date()),
                   "cost_rt_bps": round((COST_RT + SLIP_OPEN) * 1e4, 1),
                   "primary_target": "intraday"},
        "targets": targets,
        "ccf": ccf_diag(feats),
        "granger": granger_diag(feats),
        "note": "真實台指期（FinMind TX 近月）：樣本外 walk-forward 結果可作為交易參考。" if USE_TXF else "代理版（^TWII）：僅驗架構與統計價值，非台指期可交易結論。",
    }


# ==================================================================
# 對外：今日盤前判讀（含交易動作建議）
# ==================================================================
def _latest_vector(feats):
    feat, detail = {}, []
    for t in feats:
        r = _ret(fetch(t), RET[t]).dropna()
        v = float(r.iloc[-1])
        feat[t] = v
        detail.append({"ticker": t, "name": NAME[t], "group": GROUP[t],
                       "value": round(v * 100, 2), "asof": str(r.index[-1].date())})
    return np.array([[feat[c] for c in feats]]), detail


def _action(p):
    if p >= 0.58:
        return "只找多方進場"
    if p <= 0.42:
        return "只找空方進場"
    return "中性 · 不交易或等盤中確認"


def predict_today(force=False):
    data, feats = assemble(force)
    X = data[feats].values
    sc = StandardScaler().fit(X)
    vec, detail = _latest_vector(feats)
    preds = {}
    for tgt in ["gap", "day", "intraday"]:
        y = (data[tgt] > 0).astype(int).values
        clf = _fit_clf(sc.transform(X), y)
        p = float(clf.predict_proba(sc.transform(vec))[0, 1])
        d = "偏多" if p > 0.55 else "偏空" if p < 0.45 else "中性"
        preds[tgt] = {"prob_up": round(p, 3), "direction": d,
                      "confidence": round(abs(p - 0.5) * 200), "action": _action(p)}
    return {"date": time.strftime("%Y-%m-%d"), "primary_target": "intraday",
            "features": detail, "predictions": preds}
