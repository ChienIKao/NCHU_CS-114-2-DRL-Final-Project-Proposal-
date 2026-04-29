# 深度強化學習與情緒分析在自動化股票交易之研究
## 相關技術與論文綜述報告

## 摘要
自動化交易系統已從規則式專家系統演進至資料驅動模型。隨著深度學習發展，深度強化學習（Deep Reinforcement Learning, DRL）成為金融決策的重要方法。相較於僅預測價格的監督式學習，DRL 將交易建模為序列決策問題，透過持續互動優化長期風險調整後報酬。

本報告整理三個主軸：
- FinRL 與 FinRL-X 架構
- FinGPT 情緒訊號整合
- Alpaca API 即時模擬交易部署

---

## 1. 金融交易的 MDP 建模

### 1.1 基本定義
股票交易可表為馬可夫決策過程（MDP）：
- 狀態：$s_t$
- 動作：$a_t$
- 獎勵：$r_t$
- 轉移：$s_t \rightarrow s_{t+1}$
- 策略：$\pi(a_t|s_t)$

### 1.2 狀態空間（State Space）
提案設計：
- Baseline：301 維
- Enhanced：331 維

現代 DRL 多整合 OHLCV、技術指標、風險指標與情緒特徵，以提升在非平穩市場中的適應性。

| 特徵類別 | 內容 | 功能 |
|---|---|---|
| 帳戶特徵 | 餘額、持倉量、現金比例 | 風險控管與資金配置 |
| 價格特徵 | 收盤價、最高價、最低價 | 反映市場走勢 |
| 技術指標 | MACD, RSI, CCI, ADX, SMA | 捕捉趨勢、動量、超買超賣 |
| 風險指標 | VIX, Turbulence Index | 量化恐慌與異常波動 |
| 情緒特徵 | FinGPT 分數（-1 到 +1） | 捕捉新聞衝擊 |

### 1.3 動作空間（Action Space）
- 單一資產：常用離散動作 {買入、賣出、持有}
- 多資產組合（如 DOW30）：常用連續動作空間

常見連續表示：$a \in [-1, 1]$，用於持倉比例調整與再平衡。

### 1.4 獎勵函數（Reward Function）
基礎獎勵：
- $r_t = v_t - v_{t-1}$

改進方向：
- 風險調整（Sharpe 導向）
- 最大回撤懲罰
- LLM 風險評級動態調節

---

## 2. FinRL 與 FinRL-X

### 2.1 FinRL 三層架構
- 市場環境層（Market Environments）
- DRL 演算法層（DRL Agents）
- 金融應用層（Financial Applications）

可彈性切換資料源（Yahoo Finance、Alpaca、WRDS）與演算法庫（Stable Baselines3、ElegantRL）。

### 2.2 FinRL-X（Stage 3.0）
FinRL-X 導入權重中心化（Weight-Centric）介面：策略直接輸出目標權重 $w_t$，再由執行層轉換為實際下單量，降低 Sim-to-Real 落差。

| 比較項目 | FinRL（Stage 1.0） | FinRL-X（Stage 3.0） |
|---|---|---|
| 方法範式 | 傳統 DRL | AI 原生（ML + DRL + LLM） |
| 架構 | 三層耦合 | 完全模組化解耦 |
| 策略輸出 | 買賣股數 | 目標投資組合權重 |
| 訊號整合 | 以數值特徵為主 | 支援語意與 Agentic 流程 |

---

## 3. FinGPT 與情緒增強交易

### 3.1 金融 LLM 的必要性
通用 LLM 對金融術語與脈絡理解有限。FinGPT 透過資料中心化與指令微調，使模型更適用於金融情境。

### 3.2 情緒整合方式
- 狀態增強（State Augmentation）
- 決策修正（Action Refinement）

### 3.3 目標不匹配（Objective Mismatch）
關鍵風險：NLP 任務最佳化目標不等於金融效用最大化。

可行作法：
- 將 Prompt 視為離散超參數
- 直接以資訊係數（Information Coefficient）優化

### 3.4 消融實驗結果（PeerJ）
| 組別 | 年化報酬率 | 夏普比率 | 最大回撤 |
|---|---:|---:|---:|
| 完整架構（LLM + MAS + DRL） | 53.87% | 1.702 | -12.54% |
| 無 LLM（僅技術指標） | 42.19% | 1.286 | -14.83% |
| 無 DRL（純情緒規則） | 38.52% | 1.124 | -16.35% |
| 僅 DRL 基準 | 31.84% | 0.893 | -18.74% |

結論：情緒訊號可同時提升報酬與抗回撤能力。

---

## 4. 演算法比較：PPO、SAC、TD3

### 4.1 PPO
- 更新穩定，適合高噪音金融資料
- 加入情緒特徵後，Sharpe 常具優勢

### 4.2 SAC
- 探索能力強（熵最大化）
- 高維狀態下較易受噪音影響

### 4.3 TD3
- 改善 DDPG 過度估計
- 回撤控制較保守，牛市報酬可能落後 PPO

---

## 5. Alpaca 即時模擬交易部署

### 5.1 關鍵風險：前瞻性偏差
需避免在 $t$ 時點使用 $t+1$ 資訊。

控制原則：
- 情緒資料時間戳嚴格對齊
- 採滾動視窗（Walk-forward）訓練與驗證

### 5.2 系統流水線
1. 數據層：抓取日資料並計算技術指標
2. 情緒層：以本地 FinGPT 產生情緒向量
3. 策略層：載入最佳模型推論目標權重
4. 執行層：透過 Alpaca API 下單再平衡

---

## 6. FinRL 競賽趨勢
- 異質資料融合（新聞、SEC 文件、社群）
- 市場機制辨識（Regime Detection）
- 對抗式文本下的穩健性與安全性

---

## 7. 對提案的補強建議

### 7.1 模型一致性
Phase 1 的情緒標籤來源需與 Phase 2 本地推理模型對齊，避免語意分佈漂移。

### 7.2 獎勵函數強化
建議使用：

$$
R_t = \Delta Portfolio\_Value - \lambda \cdot \sigma_t - \gamma \cdot Max\_Drawdown
$$

其中 $\lambda, \gamma$ 為超參數，用於抑制過度交易。

### 7.3 基準擴展
除 Buy-and-Hold 與 MVO，建議加入「無情緒 FinRL」並比較不同產業的資訊邊際效應。

---

## 8. 結論
FinRL + FinGPT + Alpaca 的組合具備研究與工程可行性。若能解決模型一致性與前瞻性偏差，專題可延伸為具發表潛力的研究成果與可驗證的實務策略。

---

## 9. 參考文獻（References）
1. Sentiment-Enhanced DRL Trading System with Live Paper Trading（專題提案）.
2. Zhang, B., Yang, H., & Liu, X. Y. (2023). Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models. arXiv.
3. PeerJ Study (2026). Adaptive LLM-based multi-agent systems for enhanced quantitative trading. PeerJ Computer Science.
4. Liu, X. Y., et al. (2024). FinRL Contests 2023-2025: Benchmarking Financial Reinforcement Learning. NeurIPS/arXiv.
5. AI4Finance Foundation. FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading. GitHub Repository.
6. Liu, X. Y., et al. (2020). FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance. NeurIPS Workshop.
7. arXiv (2026). When Valid Signals Fail: Objective Mismatch and News Latency in LLM-DRL Trading Policies.
8. Yang, H., et al. (2026). FinRL-X: Modular and Deployment-Consistent Trading Architecture. PAKDD Workshop.
9. Yang, H., et al. (2023). FinGPT: Open-Source Financial Large Language Models. IJCAI FinLLM.
10. Ke, Z. T., Kelly, B. T., & Xiu, D. (2019). Predicting Returns With Text Data. NBER Working Paper.
11. Alpaca Learn. How to Start Paper Trading with Alpaca's Trading API.
12. AlgoTrading101. Alpaca Trading API Guide: Automation and Engineering Practices.
13. MDPI (2025). Double Deep Q-Network, A2C, and PPO in Algorithmic Trading Modeling.
14. MDPI (2025). Long-term Reward Functions with Drawdown and Volatility Penalties in DRL Trading.
