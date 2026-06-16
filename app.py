"""
app.py — 法科盤前戰情室 後端
================================
  GET  /              → 儀表板
  GET  /api/today     → 今日盤前方向 + 特徵 + 動作建議
  GET  /api/research  → 相關性(含FDR) / 交易績效 / 信心分位 / kill / CCF / Granger
  POST /api/refresh   → 清快取重抓

本機：python app.py → http://127.0.0.1:5000
部署：gunicorn 讀 PORT（見 Procfile）
"""

import os
import math
import traceback
from flask import Flask, jsonify, render_template

import analysis

app = Flask(__name__)
_cache = {}


def clean(o):
    """把 inf / nan 轉成 None，確保 JSON 合法。"""
    if isinstance(o, float):
        return None if (math.isinf(o) or math.isnan(o)) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/today")
def api_today():
    try:
        if "today" not in _cache:
            _cache["today"] = analysis.predict_today()
        return jsonify(clean(_cache["today"]))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/research")
def api_research():
    try:
        if "research" not in _cache:
            _cache["research"] = analysis.run_research()
        return jsonify(clean(_cache["research"]))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    try:
        analysis.clear_cache()
        _cache.clear()
        _cache["today"] = analysis.predict_today(force=True)
        _cache["research"] = analysis.run_research(force=False)
        return jsonify({"ok": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
