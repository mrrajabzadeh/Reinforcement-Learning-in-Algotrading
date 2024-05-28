# Stock Price Directional Change Detection and Trading Strategy Evaluation

This project leverages reinforcement learning (RL) in two primary stages to develop and evaluate stock trading strategies. The first part detects dynamic directional changes in stock prices, and the second part uses these detections to simulate trading decisions and evaluate their effectiveness against a baseline strategy.

## Project Components

### 1. Dynamic Directional Change Threshold Determination (DC Detection)

**Purpose:**  
Uses RL to dynamically calculate thresholds for detecting directional changes in stock prices, facilitating the identification of potential upturns and downturns.

**Features:**
- **Historical Data Retrieval:** Fetches historical stock price data for predefined indices using `yfinance`.
- **Dynamic Threshold Calculation:** Computes dynamic thresholds using ROC (Rate of Change) metrics.
- **Trend Detection:** Identifies upturns or downturns based on the calculated thresholds.

**Output:**
- Enriched dataset with ROC metrics, dynamic thresholds, and detected trends.
- CSV files of the processed data for further use in trading simulations.

### 2. Trading Strategy and Portfolio Performance Evaluation (Trading Strategy)

**Purpose:**  
Employs the directional change information to simulate trading decisions using an RL model and evaluates these decisions by comparing them to a baseline strategy.

**Features:**
- **State-Based Decision Making:** Decides on actions (buy, sell, hold) using a Q-learning algorithm based on observed states and trends.
- **Portfolio Management:** Manages a simulated portfolio, adjusting holdings based on trading decisions.
- **Performance Evaluation:** Calculates ROI and Sharpe ratio, comparing them to a buy-and-hold strategy.

**Output:**
- Visualizations of stock prices, trading actions, and portfolio values over time.
- Detailed logs of decisions, rewards, and Q-table updates.
- Performance reports comparing the RL strategy to buy-and-hold.

## Prerequisites

Make sure you have Python installed along with the following packages:
- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`

## Execution Instructions

1. **Run the DC Detection script:** Prepares the data with dynamic directional thresholds.
2. **Execute the Trading Strategy script:** Applies the RL model to simulate and evaluate trading strategies.

## Additional Files

- **CSV Outputs:** Files containing detailed metrics, actions, and trading results.
- **Plots:** Visual plots of stock movements, trading decisions, and portfolio changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests to us.
