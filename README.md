# GGAL Trading Bot

![GGAL Trading Bot](https://firebasestorage.googleapis.com/v0/b/leonardo-blog.firebasestorage.app/o/proyectos%2Flogo.webp?alt=media&token=53ca6d46-6501-4207-bb7b-f8c26c8c0c72)

## What is this?

**GGAL Trading Bot** is a professional algorithmic trading application built specifically for the ticker GGAL (Grupo Financiero Galicia).  
It uses machine learning to predict price moves and generate buy/sell signals—helping you make informed trading decisions, or even automate your whole strategy if that’s your vibe.

Think of it as a mashup of technical analysis, machine learning, and backtesting, all wrapped up in an interface so intuitive your grandma could use it (trust me, I tried).

---

## Main Features

- **Historical Data Analysis:** Download and visualize years of GGAL price history.
- **Technical Indicators:** Auto-calculates RSI, volatility, and moving average crossovers.
- **Predictive Model:** Decision tree algorithm to forecast price moves (up/down).
- **Backtesting:** Test your strategies on historical data with all the juicy performance stats.
- **GUI:** Visual, point-and-click interface for data, signals, results (no need to fight with the console).
- **Live Trading:** Generate real-time signals as the market moves.
- **Customization:** Tweak your model’s parameters and strategy settings any way you like.

---

## Screenshots

![Data Screen](https://firebasestorage.googleapis.com/v0/b/leonardo-blog.firebasestorage.app/o/proyectos%2Fdata_screen.webp?alt=media&token=d3e72699-11ae-4466-9b65-6d98c6ac967e)
*View and analyze historical GGAL data*

![Model Screen](https://firebasestorage.googleapis.com/v0/b/leonardo-blog.firebasestorage.app/o/proyectos%2Fmodel_screen.webp?alt=media&token=afb40be9-4b22-48c5-a344-33f9aef02281)
*Train and evaluate predictive models*

![Backtest Screen](https://firebasestorage.googleapis.com/v0/b/leonardo-blog.firebasestorage.app/o/proyectos%2Fbacktest_screen.webp?alt=media&token=dc80b555-d35a-4de4-981b-61c9122695c8)
*Backtest results and performance analysis*

![Trading Screen](https://firebasestorage.googleapis.com/v0/b/leonardo-blog.firebasestorage.app/o/proyectos%2Ftrading_screen.webp?alt=media&token=ad79e8a0-8fdb-4c22-9b02-0d8a9649978c)
*Live trading interface with real-time signals*

---

## Installation

### Requirements

- Python 3.8 or higher
- pip (Python package manager)

### Dependencies

Install these with pip (or use the `requirements.txt`):

```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
scikit-learn>=0.23.0
yfinance>=0.1.63
PyQt5>=5.15.0
joblib>=0.16.0
```

### Step-by-Step Install

1. **Clone the repo:**
   ```bash
   git clone https://github.com/leoprimero/GGAL-Trading-Bot.git
   cd GGAL-Trading-Bot
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv

   # On Windows
   venv\Scriptsctivate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   python src/main.py
   ```

---

## How to Use

### Load Data

1. Go to the **Data** tab, enter the ticker (default is GGAL) and the start date.
2. Click **Load Data** to download and process historical prices.
3. Explore the data in table or chart format.

### Train the Model

1. In the **Model** tab, set your decision tree parameters:
   - Max depth: model complexity
   - Criterion: split method (`entropy` or `gini`)
   - Test size: data portion for validation
2. Click **Train Model** to start training.
3. Check performance metrics (accuracy, recall, F1-score, etc.).
4. Optionally, save your trained model.

### Backtesting

1. In the **Backtest** tab, set your strategy parameters:
   - Confidence threshold: minimum probability for signals
   - Initial capital: simulated account balance
   - Position size: percent of capital per trade
2. Click **Run Backtest** to test the strategy.
3. Analyze results:
   - Performance metrics (total return, Sharpe ratio, etc.)
   - Equity curve
   - Signal visualization on price chart

### Live Trading

1. In the **Live Trading** tab, hit **Update Data** for the latest prices.
2. Review the latest prediction and signal.
3. Set auto-mode and refresh interval if you want automation.
4. Use **Start Trading** and **Stop Trading** to control execution.
5. Check the operations log for your trading history.

---

## Project Structure

```
GGAL-Trading-Bot/
│
├── docs/                      # Documentation
│   └── images/                # Docs images
│
├── notebooks/                 # Exploratory notebooks
│
├── src/                       # Source code
│   ├── data/                  # Data handling modules
│   │   └── data_loader.py
│   ├── models/                # Model training/evaluation
│   │   └── model_trainer.py
│   ├── strategies/            # Trading strategies
│   │   └── trading_strategy.py
│   ├── gui/                   # GUI app
│   │   └── main_app.py
│   ├── utils/                 # General utilities
│   └── main.py                # Main entry point
│
├── tests/                     # Unit/integration tests
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Technical Overview

### Indicators Used

- **RSI (Relative Strength Index):** Measures speed and change of price movements.
- **Volatility:** Standard deviation of daily returns.
- **Moving Average Crosses:** Relations between moving averages of different periods.

### Predictive Model

The bot uses a **decision tree** to classify future price moves:
- **Class 1:** Price is predicted to go up.
- **Class 0:** Price is predicted to go down.

Decision trees are great for trading—they’re interpretable and can handle non-linear relationships between features.

### Trading Strategy

The basic logic:
1. **Buy Signal:** When the model predicts “up” (class 1) with confidence above the threshold.
2. **Sell Signal:** When the model predicts “down” (class 0) with confidence above the threshold.
3. **No Action:** If confidence is below threshold.

---

## Customization & Extending

### Tweak the Parameters

Main knobs for optimization:

- **Prediction Window:** Change `ventana` in `data_loader.py` for different forecast horizons.
- **Tree Depth:** Set `max_depth` in the UI for model complexity.
- **Confidence Threshold:** Adjust to be more aggressive/conservative on signals.

### Add New Indicators

1. Edit `DataLoader` in `data_loader.py` to compute new features.
2. Make sure to add the new indicator to your model’s feature set.

### Implement New Strategies

1. Extend `TradingStrategy` in `trading_strategy.py` or add your own strategy class.
2. Update the GUI to select between strategies.

---

## Risk Disclaimer

This software is for **educational and informational purposes only.**  
Please note:

- **Not Financial Advice:** The signals generated are not investment recommendations.
- **Market Risk:** Stock trading involves risk of capital loss.
- **Model Limitations:** No model can predict all market events.
- **Validation:** Always backtest thoroughly before trading with real money.

---

## Contributions

Contributions are welcome!

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## License

MIT. Use it, fork it, break it, fix it—just don’t blame me if you lose money.

---

## Author

Leonardo I (a.k.a. [@leonardoprimero](https://github.com/leonardoprimero))
