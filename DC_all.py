# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 11:16:33 2024

@author: mraja
"""

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function to update the current trend and the extreme values (highs and lows)
def update_trend_and_extremes(price, last_dc_event, trend, pl, ph):
    if last_dc_event == 'upturn':
        if trend != 'upward':  # If the trend is just starting, set ph to the current price
            trend = 'upward'
            ph = price
        else:
            ph = max(ph, price)  # If the trend is continuing, update ph as usual
    elif last_dc_event == 'downturn':
        if trend != 'downward':  # If the trend is just starting, set pl to the current price
            trend = 'downward'
            pl = price
        else:
            pl = min(pl, price)  # If the trend is continuing, update pl as usual
    else:
        trend = None
        pl = min(pl, price) if pl is not None else price
        ph = max(ph, price) if ph is not None else price
    return trend, pl, ph

# Function to calculate Rate of Change (ROC) components
def calculate_roc_components(data, day, pl, ph):
    open_t = data['open'][day]
    close_t_minus_1 = data['close'][day - 1] if day > 0 else data['open'][0]
    open_t_minus_1 = data['open'][day - 1] if day > 0 else data['open'][0]

    Upward_ROC = (open_t - ph) / ph if ph != 0 else 0
    Downward_ROC = (open_t - pl) / pl if pl != 0 else 0
    Overnight_ROC = (open_t - close_t_minus_1) / close_t_minus_1 if close_t_minus_1 != 0 else 0
    PreviousDay_ROC = (close_t_minus_1 - open_t_minus_1) / open_t_minus_1 if open_t_minus_1 != 0 else 0

    return Upward_ROC, Downward_ROC, Overnight_ROC, PreviousDay_ROC

# Function to observe the current state based on price changes and ROC components
def observe_state(data, day, pl, ph, trend):
    if day < 5:
        return 'Neutral'  # Not enough data for a 5-day moving average

    price_change = data['close'].diff()
    percentage_change = price_change / data['close'].shift(1)
    five_day_percentage_change_moving_average = percentage_change.rolling(window=5).mean().iloc[day]
    Upward_ROC, Downward_ROC, Overnight_ROC, PreviousDay_ROC = calculate_roc_components(data, day, pl, ph)
    
    # Determine the observed state based on ROC and moving average
    if abs(PreviousDay_ROC) > abs(five_day_percentage_change_moving_average) or abs(Overnight_ROC) > abs(five_day_percentage_change_moving_average):
        if abs(PreviousDay_ROC) > abs(Overnight_ROC):
            return 'Ext_Previous'
        else:
            return 'Ext_Overnight'
    else:
        return 'Neutral'

# Function to check if a state exists in the Q-table, and add it if not
def check_state_exist(q_table, state, actions):
    if state not in q_table.index:
        new_state = pd.Series([0]*len(actions), index=q_table.columns, name=state)
        q_table = q_table.append(new_state)
    return q_table

# Function to select an action based on exploration-exploitation trade-off
def select_action(state, q_table, actions, exploration_rate):
    q_table = check_state_exist(q_table, state, actions)
    
    if np.random.uniform() < exploration_rate:
        action = np.random.choice(actions)  # Exploration
    else:
        state_actions = q_table.loc[state, :]
        action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)  # Exploitation
    
    return action

# Function to update the Q-table based on the observed reward
def learn(q_table, state, action, reward, next_state, learning_rate, discount_factor, actions):
    q_table = check_state_exist(q_table, state, actions)
    q_table = check_state_exist(q_table, next_state, actions)
    
    q_predict = q_table.loc[state, action]
    if next_state != 'terminal':
        q_target = reward + discount_factor * q_table.loc[next_state, :].max()
    else:
        q_target = reward
    q_table.loc[state, action] += learning_rate * (q_target - q_predict)
    return q_table

# Function to calculate the reward based on the selected action
def calculate_reward(action, data, day, pl, ph):
    Upward_ROC, Downward_ROC, Overnight_ROC, PreviousDay_ROC = calculate_roc_components(data, day, pl, ph)
    
    if action == 'DT_Overnight':
        return max(PreviousDay_ROC, Overnight_ROC)
    elif action == 'DT_PreviousDay':
        return max(PreviousDay_ROC, Overnight_ROC)
    elif action == 'DT':
        return 0
    else:
        raise ValueError("Invalid action")

# Function to calculate the dynamic threshold value based on the action and trend
def calculate_dynamic_threshold_value(action, data, day, trend, pl, ph):
    Upward_ROC, Downward_ROC, Overnight_ROC, PreviousDay_ROC = calculate_roc_components(data, day, pl, ph)
    
    if trend == 'upward':
        if action == 'DT_Overnight':
            return Upward_ROC + Overnight_ROC
        elif action == 'DT_PreviousDay':
            return Upward_ROC + PreviousDay_ROC
        elif action == 'DT':
            return Upward_ROC + PreviousDay_ROC + Overnight_ROC
    elif trend == 'downward':
        if action == 'DT_Overnight':
            return Downward_ROC + Overnight_ROC
        elif action == 'DT_PreviousDay':
            return Downward_ROC + PreviousDay_ROC
        elif action == 'DT':
            return Downward_ROC + PreviousDay_ROC + Overnight_ROC
    else:  # No detected upward/downward trend
        min_ROC = min(Upward_ROC, Downward_ROC)
        if action == 'DT_Overnight':
            return min_ROC + Overnight_ROC
        elif action == 'DT_PreviousDay':
            return min_ROC + PreviousDay_ROC
        elif action == 'DT':
            return min_ROC + PreviousDay_ROC + Overnight_ROC

# Function to calculate the dynamic threshold
def calculate_dynamic_threshold(action, data, day, trend, pl, ph, prev_threshold):
    lambda_ = calculate_dynamic_threshold_value(action, data, day, trend, pl, ph)
    
    if lambda_ < 0.02:
        lambda_ = prev_threshold if prev_threshold is not None else lambda_
    
    return lambda_

# Function to detect upturn DC event
def detect_upturn_dc_event(price, ph, lambda_):
    if price >= ph * (1 + abs(lambda_)):
        return "Upturn DC Event Detected"
    else:
        return "No DC Event Detected"

# Function to detect downturn DC event
def detect_downturn_dc_event(price, pl, lambda_):
    if price <= pl * (1 - abs(lambda_)):
        return "Downturn DC Event Detected"
    else:
        return "No DC Event Detected"

np.random.seed(42)  # Set a random seed for reproducibility

# Main code execution
if __name__ == "__main__":
    # Define stock indices and date ranges
    stocks = ["^GSPC", "^IXIC", "^DJI"]  # S&P 500, NASDAQ, Dow Jones
    start_dates = ["2015-07-01", "2015-07-01", "2015-07-01"]
    end_dates = ["2020-07-31", "2020-07-31", "2020-07-31"]

    # Loop through each stock
    for stock, start_date, end_date in zip(stocks, start_dates, end_dates):
        # Download the stock data
        data = yf.download(stock, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data = data[['Date', 'Open', 'Close']]
        data.columns = ['date', 'open', 'close']

        # Initialize Q-learning parameters
        exploration_rate = 0.1
        learning_rate = 0.01
        discount_factor = 0.9

        # Initialize variables for tracking trend and extreme values
        pl = ph = data['close'].iloc[0]
        trend = last_dc_event = None
        pl_updates = ph_updates = 0
        upturn_dc_events_count = 0
        downturn_dc_events_count = 0
        q_table = pd.DataFrame(columns=['DT_Overnight', 'DT_PreviousDay', 'DT'])
        actions = q_table.columns.tolist()

        # Initialize columns to store data analysis results
        data['Upward_ROC'] = pd.Series(dtype='float64')
        data['Downward_ROC'] = pd.Series(dtype='float64')
        data['Overnight_ROC'] = pd.Series(dtype='float64')
        data['PreviousDay_ROC'] = pd.Series(dtype='float64')
        data['pl'] = pd.Series(dtype='float64')
        data['ph'] = pd.Series(dtype='float64')
        data['trend'] = pd.Series(dtype='object')
        data['observed_state'] = pd.Series(dtype='object')
        data['dynamic_threshold'] = pd.Series(dtype='float64')
        data['action'] = pd.Series(dtype='object')
        abs_dynamic_thresholds = []

        prev_threshold = None

        # Iterate through the data day by day
        for day in range(len(data)):
            price = data['close'].iloc[day]
            trend, pl, ph = update_trend_and_extremes(price, last_dc_event, trend, pl, ph)
            data.at[day, 'pl'] = pl
            data.at[day, 'ph'] = ph
            data.at[day, 'trend'] = trend

            # Calculate ROC components
            Upward_ROC, Downward_ROC, Overnight_ROC, PreviousDay_ROC = calculate_roc_components(data, day, pl, ph)
            data.at[day, 'Upward_ROC'] = Upward_ROC
            data.at[day, 'Downward_ROC'] = Downward_ROC
            data.at[day, 'Overnight_ROC'] = Overnight_ROC
            data.at[day, 'PreviousDay_ROC'] = PreviousDay_ROC

            # Observe the state based on ROC components and moving average
            observed_state = observe_state(data, day, pl, ph, trend)
            data.at[day, 'observed_state'] = observed_state
            state = observed_state

            # Select the action based on the Q-table and exploration rate
            action = select_action(state, q_table, actions, exploration_rate)
            data.at[day, 'action'] = action

            # Calculate the dynamic threshold
            dynamic_threshold = calculate_dynamic_threshold(action, data, day, trend, pl, ph, prev_threshold)
            data.at[day, 'dynamic_threshold'] = dynamic_threshold
            prev_threshold = dynamic_threshold
            abs_dynamic_thresholds.append(abs(dynamic_threshold))
            mean_abs_dynamic_threshold = np.mean(abs_dynamic_thresholds)

            # Update the number of times price low or high is updated
            if pl == price:
                pl_updates += 1
            if ph == price:
                ph_updates += 1

            # Determine the trend direction
            if day == 20:
                if ph_updates < pl_updates:
                    trend = 'upward'
                else:
                    trend = 'downward'

            # Detect DC events based on the trend
            if trend == 'upward':
                dc_event = detect_downturn_dc_event(price, pl, abs(dynamic_threshold))
                if dc_event == "Downturn DC Event Detected":
                    downturn_dc_events_count += 1
                    last_dc_event = 'downturn'
            elif trend == 'downward':
                dc_event = detect_upturn_dc_event(price, ph, abs(dynamic_threshold))
                if dc_event == "Upturn DC Event Detected":
                    upturn_dc_events_count += 1
                    last_dc_event = 'upturn'
            else:
                dc_event = "No DC Event Detected"

            data.at[day, 'dc_event'] = dc_event
            if dc_event == "Upturn DC Event Detected":
                last_dc_event = 'upturn'
            elif dc_event == "Downturn DC Event Detected":
                last_dc_event = 'downturn'

            # Calculate the reward and update the Q-table
            reward = calculate_reward(action, data, day, pl, ph)
            next_state = observe_state(data, day + 1, pl, ph, trend) if day + 1 < len(data) else 'terminal'
            q_table = learn(q_table, state, action, reward, next_state, learning_rate, discount_factor, actions)

            # Print the details of each day's analysis
            print(f"Day {day+1} ({data['date'].iloc[day]}): pl: {pl}, ph: {ph}, trend: {trend}, state: {state}")
            print(f"Upward_ROC: {Upward_ROC}, Downward_ROC: {Downward_ROC}, Overnight_ROC: {Overnight_ROC}, PreviousDay_ROC: {PreviousDay_ROC}")
            print(f"Observed State: {observed_state}")
            print(f"Dynamic Threshold: {dynamic_threshold}")
            print(f"DC Event: {dc_event}")
            print(f"Action: {action}, Reward: {reward}")
            print(q_table)

        # Print summary statistics
        print(f"Total Upturn DC Events: {upturn_dc_events_count}")
        print(f"Total Downturn DC Events: {downturn_dc_events_count}")
        print(f"Mean of Absolute Dynamic Threshold: {mean_abs_dynamic_threshold}")

        # Create the plot for the stock prices and DC events
        fig, ax = plt.subplots(figsize=(25, 10))
        ax.plot(data['date'], data['close'], label='Close Price', color='blue')
        upturn_dc_events = data[data['dc_event'] == 'Upturn DC Event Detected']
        ax.scatter(upturn_dc_events['date'], upturn_dc_events['close'], color='green', label='Upturn DC Event')
        downturn_dc_events = data[data['dc_event'] == 'Downturn DC Event Detected']
        ax.scatter(downturn_dc_events['date'], downturn_dc_events['close'], color='red', label='Downturn DC Event')
        ax.set_title(f'Stock Price ({stock}) with DC Events and Trends', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Close Price', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{stock}_plot.png")
        plt.close()

        # Save the data to a CSV file
        data.to_csv(f"DC_Semi_{stock}.csv", index=False)
