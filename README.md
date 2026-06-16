# 法科 · 盤前戰情室 (TAIEX Pre-Open Engine) — v0.2

用「台北時間 08:45（台指期日盤開盤）前已可確定」的全球市場資訊，量化哪些指數/個股
與台指期相關且具領先性，並輸出今日盤前方向判讀與交易動作建議。

## 本機執行

```bash
pip install -r requirements.txt
python app.py            # 開 http://127.0.0.1:5000
```

首次載入會抓 Yahoo 行情、訓練模型並跑樣本外回測（約一分鐘），之後 30 分鐘內走快取。
右上「重抓資料」可強制更新。

## v0.2 重點（依三輪 review 共識）

- **資料層介面化**：`DataSource` 抽象類別。換台指期只需實作 `TXFDataSource.daily()`
  並把 `analysis.ACTIVE_SOURCE` 指過去，其餘程式不動。
- **夜盤殘差特徵**：新增 NQ/ES 期貨隔夜與匯率，並對每個特徵標記 `priced_in`
  （夜盤是否已消化）。設定 `RESIDUAL_ONLY=True` 即自動丟掉已被夜盤反映的特徵，
  只留 05:00→08:45 的殘差資訊，避免重複計入隔夜美股。
- **交易績效**：取代單純準確率，輸出期望值 / Profit Factor / 最大回撤 / Sharpe /
  賺賠比 / 交易次數，且全部扣交易成本（`COST_RT` + `SLIP_OPEN`）。
- **信心分位數表**：驗證「模型機率越高，報酬是否越好」——這是分數模型能否實戰的關鍵。
- **L2 強度自動選**：以 `TimeSeriesSplit` CV 選 `C`（取代固定值），緩解共線性下係數不穩。
- **BH-FDR**：相關性檢定加 Benjamini-Hochberg 校正，控制多重檢定的假顯著（僅用於篩因子）。
- **kill criterion**：高信心區間（p≥0.60 / ≤0.40）扣成本後若無正期望值，明確判定「不該交易」。
- **移除共整合**：對方向模型無價值，已自報告移除（`TXFDataSource` 留有 stat-arb 插孔可另接）。

## 換成真正的台指期（最關鍵的一步）

1. 實作 `analysis.TXFDataSource.daily(ticker)`，回傳含 `Open`(=08:45)、`Close`(=13:45)
   的日盤 DataFrame；建議另存夜盤 05:00 收盤以建殘差特徵。
2. `analysis.ACTIVE_SOURCE = TXFDataSource(...)`、`analysis.TARGET = "你的TXF代碼"`。
3. 設 `RESIDUAL_ONLY = True`（夜盤已 price-in 隔夜美股）。
4. 重跑。此時的績效與 kill criterion 才是「台指期可不可交易」的結論。

## 重要提醒

- 代理版（`^TWII`）只驗架構與統計價值；`^TWII` 09:00 才開盤，無法定義 08:45 開盤目標。
- 已用合成隨機資料做過 smoke test：在無訊號資料上，引擎正確回報期望值為負、
  FDR 無顯著、kill criterion 全數未通過——它不會在雜訊上給你假 edge。
- 所有結果為歷史樣本外統計，非投資建議。

## 結構

```
analysis.py          分析核心（DataSource 介面、特徵、績效、FDR、kill）
app.py               Flask 後端（/api/today, /api/research, /api/refresh）
templates/index.html 暗色戰情室儀表板（紅漲綠跌、印章判讀、績效、信心分位）
requirements.txt / Procfile  本機與 Railway 部署
```
