# GBP/USD Forex Price Prediction — Deep Learning Comparative Study

A controlled comparison of **LSTM**, **BiLSTM**, **GRU**, and a hybrid **CNN-BiLSTM-Attention** architecture for one-hour-ahead GBP/USD exchange rate forecasting, using a 21-feature technical indicator set and leakage-free preprocessing.

> MSc Project — Bahir Dar Institute of Technology, Faculty of Computing, Department of AI and Data Science

---

## Results Summary

| Model | RMSE | MAE | MAPE% | R² | Dir. Acc% | Return% | MaxDD% |
|---|---|---|---|---|---|---|---|
| LSTM | 0.00968 | 0.00628 | 0.794 | 0.9834 | 48.56 | -3.61 | -6.84 |
| BiLSTM | 0.00985 | 0.00669 | 0.844 | 0.9828 | 48.56 | -4.81 | -8.66 |
| **GRU** | **0.00961** | **0.00634** | **0.800** | **0.9837** | 48.50 | -4.35 | -7.08 |
| **CNN-BiLSTM-Attn** | 0.01023 | 0.00709 | 0.896 | 0.9815 | **49.50** | **-1.03** | **-4.13** |

**GRU** wins on regression accuracy. **CNN-BiLSTM-Attention** wins on directional accuracy and risk-adjusted trading metrics. No single model dominates every metric — see [Results](#results) for full discussion.

---

## Project Structure

```
forex-gbpusd-prediction/
├── data/
│   └── gbpusd_hourly_newyorkTimezone.csv   # raw GBP/USD H1 data, Jun 2024 – May 2026
├── forex_run.py                            # full pipeline: load → features → train → evaluate
├── requirements.txt
├── results/                                # generated after running (not tracked in git)
│   ├── comparison.csv
│   ├── backtest.csv
│   ├── predictions.png
│   ├── learning_curves.png
│   ├── rmse_bar.png
│   └── equity_curves.png
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Asset | GBP/USD |
| Timeframe | 1 Hour (H1) |
| Timezone | New York (UTC−4) |
| Date range | June 16, 2024 – May 15, 2026 |
| Rows | 11,825 |
| Columns | Datetime, Open, High, Low, Close |
| Location | `data/gbpusd_hourly_newyorkTimezone.csv` |

The raw file is included in this repo under `data/`. No external download is needed to reproduce results.

---

## Quickstart

```bash
git clone https://github.com/<your-username>/forex-gbpusd-prediction.git
cd forex-gbpusd-prediction

pip install -r requirements.txt

python forex_run.py
```

Results (metrics tables + plots) will be saved to `results/`.

---

## Method Overview

**1. Feature engineering** — 21 features built from raw OHLC: returns, SMA/EMA, RSI, MACD, Bollinger Band width, ATR, candle body/range.

**2. Preprocessing** (leakage-free):
- Missing value fill → forward-fill + backward-fill
- Duplicate timestamp removal
- Outlier removal — IQR × 4 on log returns
- Chronological 70/15/15 train/val/test split (no shuffling)
- MinMaxScaler **fit on training data only**
- Sliding window of 60 timesteps → predict next close

**3. Models** — all share input shape `(60, 21)`, Adam optimizer, MSE loss, EarlyStopping + ReduceLROnPlateau:
- `LSTM` — 2 stacked layers, 64 units
- `BiLSTM` — 2 stacked bidirectional layers
- `GRU` — 2 stacked layers, 64 units
- `CNN-BiLSTM-Attention` — Conv1D → BiLSTM → Bahdanau attention → Dense

**4. Evaluation** — RMSE, MAE, MAPE, R², Directional Accuracy, plus a simple directional backtest (Win Rate, Total Return, Max Drawdown).

---

## Requirements

```
pandas
numpy
scikit-learn
tensorflow
matplotlib
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Configuration

Edit the config block at the top of `forex_run.py`:

```python
CSV_PATH   = "data/gbpusd_hourly_newyorkTimezone.csv"
TARGET     = "close"      # "close" or "direction"
WINDOW     = 60
UNITS      = 64
DROPOUT    = 0.2
BATCH      = 32
EPOCHS     = 100
```

---

## Citation

If you use this code or dataset, please cite this repository:

```bibtex
@misc{daget2026gbpusd,
  author       = {Amanuel Daget and Robel Alemante and Mengistu Tessema},
  title        = {GBP/USD Forex Price Prediction Using LSTM, BiLSTM, GRU, and CNN-BiLSTM-Attention},
  year         = {2026},
  institution  = {Bahir Dar Institute of Technology},
  howpublished = {\url{https://github.com/<your-username>/forex-gbpusd-prediction}}
}
```


## License

This project is released under the MIT License. The dataset is provided for academic and research use.

---

## Author

**Amanuel Daget** — BDU1807098

Msc Student in AI and Data Science,
Bahir Dar Institute of Technology.
Bahir Dar, Ethiopia
