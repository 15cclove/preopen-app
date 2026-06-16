# CLAUDE.md — 法科盤前方向引擎

> 這份檔案會在每次 Claude Code session 啟動時自動載入。請先讀完再動手。

## 專案目的

用「台北時間 08:45（台指期日盤開盤）前已可確定」的全球市場資訊，預測台指期當日
**可交易方向**，並輸出盤前方向分數，最終整合進使用者既有的「法科」日內交易系統。

## 現況

- 版本 v0.2，代理研究版，可本機跑、可部署 Railway。
- 本機啟動：`pip install -r requirements.txt` 然後 `python app.py` → http://127.0.0.1:5000
- 已用合成隨機資料做過 smoke test：無訊號資料上引擎正確回報期望值為負、FDR 零顯著、
  kill criterion 全數未通過（不會在雜訊上產生假 edge）。

## 檔案

- `analysis.py` — 分析核心。`DataSource` 介面、特徵、對齊、walk-forward、交易績效、
  信心分位、BH-FDR、CCF/Granger、kill criterion、`predict_today()`。
- `app.py` — Flask 後端，`/api/today`、`/api/research`、`/api/refresh`，含 JSON 清洗。
- `templates/index.html` — 暗色戰情室儀表板（紅漲綠跌、印章判讀、績效、信心分位）。

## 已拍板的設計決策（請勿推翻，除非有強理由）

1. **用對數報酬，不用價格水準**（避免假性相關）。
2. **無 look-ahead 紀律**：美股類 `cc` 報酬 shift +1 台股日；亞洲類**只能用開盤跳空**
   (`gap`)，絕不可用當日收盤（會看到未來）。
3. **主目標是 `intraday`（Close/Open）**＝開盤後可交易方向。`gap` 多已反映在開盤價、
   `day` 易被跳空主導，兩者只作輔助。
4. **績效看期望值/PF/最大回撤/Sharpe，全部扣成本**；勝率只是輔助。
5. **信心分位數表是核心驗證**：機率越高、報酬要越好，否則分數模型沒用。
6. **FDR 只用於「篩因子」階段**，不可拿來當「能否交易」的判準；判交易看樣本外期望值。
7. **L2 強度用 TimeSeriesSplit CV 自動選**，緩解美股科技股共線性下的係數不穩；
   因此**不要把個別共線係數大小當結論**。
8. **kill criterion**：高信心區間（p≥0.60/≤0.40）扣成本後若無正期望值，判定不該交易。
   亮 HOLD 不一定是 bug，可能就是這題沒 edge——那是有價值的結論。
9. **共整合已移除**（對方向模型無價值）。要做請另開 stat-arb 模組
   （TSM ADR + 匯率 vs 台積電現股），別塞回方向模型。

## 下一步（依優先序）

1. **接真實台指期資料（最高優先、卡住一切）**：實作 `analysis.TXFDataSource.daily()`，
   回傳含 `Open`(=08:45)、`Close`(=13:45) 的日盤 DataFrame；另存夜盤 05:00 收盤。
   接好後 `ACTIVE_SOURCE = TXFDataSource(...)`、`TARGET = "<TXF代碼>"`、`RESIDUAL_ONLY = True`。
   ★ 動工前先問使用者資料來源與格式（CSV？哪個資料商？有沒有 08:45 開盤、夜盤收盤）。
2. **夜盤 price-in 殘差特徵**：TXF 夜盤已消化隔夜美股，特徵不可同時放「美股收盤」與
   「夜盤報酬」（重複計入）。用 `RESIDUAL_ONLY=True` 丟掉 `priced_in` 特徵，
   只留 05:00→08:45 的殘差 + 韓日開盤。
3. **加入交易績效後的門檻最佳化**：掃 THR 與成本敏感度。
4. **regime model（謹慎）**：一次只加**一個** regime 軸（先試 VIX 高/低），確認樣本外
   有加值再加第二個。樣本只有約 2000 日，五維 regime 會過擬合，**不要一步到位**。

## 不要做的事

- 不要用亞洲市場的當日收盤當特徵（look-ahead）。
- 不要把代理版（^TWII）的漂亮數字當成台指期可交易的結論。
- 不要相信 in-sample 結果；一切以 walk-forward 樣本外為準。
- 不要一次堆五個 regime 維度。
- 不要把共整合塞回方向模型。

## 慣例

- 程式註解與 commit 訊息用**繁體中文**。
- 改動 `analysis.py` 後跑一次 smoke（用合成 `DataSource` 覆蓋 `ACTIVE_SOURCE`，
  確認 `run_research()` / `predict_today()` 能跑且 JSON 可序列化）再 commit。
- 新增相依套件要同步更新 `requirements.txt`。
- 環境裝 pip 套件用 `pip install --break-system-packages`（若在受限環境）。

## 開發環境

- Python 3，需 `pandas numpy scipy statsmodels scikit-learn flask yfinance`（見 requirements.txt）。
- 代理版資料來自 Yahoo Finance（需可對外連網）。
