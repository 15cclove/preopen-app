"""
app.py — 法科盤前方向引擎後端
  GET  /              → 儀表板
  GET  /api/today     → 今日盤前方向 + 特徵 + 動作建議
  GET  /api/research  → 相關性 / 交易績效 / 信心分位 / kill / CCF / Granger
  POST /api/refresh   → 清快取重抓
"""

import os
import math
import traceback
import datetime
import pytz
from flask import Flask, jsonify, render_template
from apscheduler.schedulers.background import BackgroundScheduler

import analysis

app = Flask(__name__)
_cache = {}

TAIPEI = pytz.timezone("Asia/Taipei")


def _taipei_now():
    return datetime.datetime.now(TAIPEI)


def clean(o):
    if isinstance(o, float):
        return None if (math.isinf(o) or math.isnan(o)) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def _populate_cache(force=False):
    """抓資料並記錄台北時間更新時間。"""
    now = _taipei_now()
    _cache["today"] = analysis.predict_today(force=force)
    if "research" not in _cache or force:
        _cache["research"] = analysis.run_research(force=False)
    _cache["updated_at"] = now.strftime("%Y年%m月%d日 %H:%M")
    _cache["target_date"] = now.strftime("%Y年%m月%d日")


def _scheduled_refresh():
    """每天 08:30 台北時間自動觸發。"""
    try:
        analysis.clear_cache()
        _cache.clear()
        _populate_cache(force=True)
        print(f"[排程] {_cache['updated_at']} 自動更新完成")
    except Exception as e:
        print(f"[排程] 自動更新失敗：{e}")
        traceback.print_exc()


# ── 排程：每天台北時間 08:30 自動更新 ──
_scheduler = BackgroundScheduler(timezone=TAIPEI)
_scheduler.add_job(_scheduled_refresh, "cron", hour=8, minute=30)
_scheduler.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/today")
def api_today():
    try:
        if "today" not in _cache:
            _populate_cache()
        result = dict(_cache["today"])
        result["updated_at"] = _cache.get("updated_at", "—")
        result["target_date"] = _cache.get("target_date", "—")
        return jsonify(clean(result))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/research")
def api_research():
    try:
        if "research" not in _cache:
            _populate_cache()
        return jsonify(clean(_cache["research"]))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    try:
        analysis.clear_cache()
        _cache.clear()
        _populate_cache(force=True)
        return jsonify({"ok": True, "updated_at": _cache.get("updated_at")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
