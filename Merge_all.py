# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:25:08 2024

@author: mraja
"""

# Import necessary libraries
import os
import yfinance as yf  # Yahoo Finance to fetch financial data
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting library
import matplotlib.dates as mdates  # Handle date formatting in plots

# Function to determine the state based on 'dc_event' and 'action' columns
def observe_state_M(row):
    # Check the 'dc_event' column and classify the state accordingly
    if row['dc_event'] == 'Upturn DC Event Detected':
        if row['action'] == 'DT_Overnight':
            return 'Overnight Upward trend'
        elif row['action'] == 'DT_PreviousDay':
            return 'Previous Day Upward trend'
    elif row['dc_event'] == 'Downturn DC Event Detected':
        if row['action'] == 'DT_Overnight':
            return 'Overnight Downward trend'
        elif row['action'] == 'DT_PreviousDay':
            return 'Previous Day Downward trend'
    elif row['dc_event'] == 'No DC Event Detected':
        # Classify based on 'trend' direction when no event is detected
        if row['action'] == 'DT_Overnight' and row['trend'] == 'upward':
            return 'Neutral Upward trend'
        elif row['action'] == 'DT_Overnight' and row['trend'] == 'downward':
            return 'Neutral Downward trend'
        elif row['action'] == 'DT_PreviousDay' and row['trend'] == 'upward':
            return 'Neutral Upward trend'
        elif row['action'] == 'DT_PreviousDay' and row['trend'] == 'downward':
            return 'Neutral Downward trend'
    return 'Unknown State'  # Default return if no conditions are met

# Function to check if a state exists in the Q-table and add if it doesn't
def check_state_exist_M(q_table_M, state_M, actions_M):
    if state_M not in q_table_M.index:
        # Initialize new state with zeros for all actions if it doesn't exist
        new_state_M = pd.Series([0]*len(actions_M), index=q_table_M.columns, name=state_M)
        q_table_M = q_table_M.append(new_state_M)
    return q_table_M

# Function to select an action based on the current state and Q-table
def select_action_M(state_M, q_table_M, actions_M, exploration_rate_M, cash_M, shares_M, stock_price_M,
                    last_significant_action=None):
    q_table_M = check_state_exist_M(q_table_M, state_M, actions_M)
    
    # Determine allowable actions based on the last significant action
    if last_significant_action == 'Buy':
        allowable_actions_M = ['Hold', 'Sell']
    elif last_significant_action == 'Sell':
        allowable_actions_M = ['Hold', 'Buy']
    else:  # All actions are allowed if last action was None or 'Hold'
        allowable_actions_M = actions_M
        
    if np.random.uniform() < exploration_rate_M:
        # Choose a random action based on exploration rate
        action_M = np.random.choice(allowable_actions_M)
    else:
        # Choose the best action based on Q-values
        state_actions_M = q_table_M.loc[state_M, allowable_actions_M]
        action_M = np.random.choice(state_actions_M[state_actions_M == np.max(state_actions_M)].index)
    
    # Execute the action and update cash and shares accordingly
    if action_M == 'Buy':
        shares_bought_M = cash_M / stock_price_M  # Calculate the number of shares to buy
        cash_M = 0
        shares_M += shares_bought_M
    elif action_M == 'Sell':
        cash_M += shares_M * stock_price_M  # Calculate the cash received from selling
        shares_M = 0
    
    return action_M, cash_M, shares_M

# Learning function to update Q-values based on actions taken and rewards received
def learn_M(q_table_M, state_M, action_M, reward_M, next_state_M, learning_rate_M, discount_factor_M, actions_M):
    q_table_M = check_state_exist_M(q_table_M, state_M, actions_M)
    q_table_M = check_state_exist_M(q_table_M, next_state_M, actions_M)
    
    # Calculate predicted and target Q-values
    q_predict_M = q_table_M.loc[state_M, action_M]
    if next_state_M != 'terminal':
        q_target_M = reward_M + discount_factor_M * q_table_M.loc[next_state_M, :].max()
    else:
        q_target_M = reward_M
    q_table_M.loc[state_M, action_M] += learning_rate_M * (q_target_M - q_predict_M)
    
    return q_table_M

# Function to find the next directional change event in the dataset
def find_next_dc_event(data_M, current_index):
    for next_index in range(current_index + 1, len(data_M)):
        next_row = data_M.iloc[next_index]
        if next_row['dc_event'] in ['Upturn DC Event Detected', 'Downturn DC Event Detected']:
            return next_row['dc_event'], next_row['close']
    return None, None

# Function to calculate the reward based on the action taken and market conditions
def calculate_reward_M(action_M, stock_price_M, buying_price_M, next_dc_event_M, next_dc_event_price_M, portfolio_status_M):
    if pd.isnull(next_dc_event_price_M) or pd.isnull(next_dc_event_M) or buying_price_M is None:
        return 0  # No reward if necessary data is missing or buying price is None

    # Calculate reward based on the action taken and subsequent market conditions
    if action_M == 'Sell':
        return (stock_price_M - next_dc_event_price_M) / stock_price_M if buying_price_M != 0 else 0
    elif action_M == 'Buy':
        return (next_dc_event_price_M - stock_price_M) / stock_price_M if stock_price_M != 0 else 0
    elif action_M == 'Hold':
        # Reward calculation for holding depends on portfolio status and market event direction
        if portfolio_status_M == 'cash':
            if next_dc_event_M == 'Upturn DC Event Detected':
                return (stock_price_M - next_dc_event_price_M) / stock_price_M if stock_price_M != 0 else 0
            elif next_dc_event_M == 'Downturn DC Event Detected':
                return (next_dc_event_price_M - stock_price_M) / stock_price_M if stock_price_M != 0 else 0
        elif portfolio_status_M == 'stock':
            if next_dc_event_M == 'Upturn DC Event Detected':
                return (next_dc_event_price_M - stock_price_M) / stock_price_M if stock_price_M != 0 else 0
            elif next_dc_event_M == 'Downturn DC Event Detected':
                return (stock_price_M - next_dc_event_price_M) / stock_price_M if stock_price_M != 0 else 0
    return 0  # Default reward if none of the above conditions are met

# Initialize parameters for multiple runs of the model
num_runs_M = 2
results_M = []
exploration_rate_M = 0.5  # Exploration rate for random action selection
learning_rate_M = 0.3  # Learning rate for Q-value updates
discount_factor_M = 0.9  # Discount factor for future rewards
np.random.seed(42)  # Set seed for reproducibility

# Lists to store final results
all_rois_final_M = []
all_sharpe_ratios_final_M = []
all_portfolio_values_final_M = []
rewards_M = []

# Define dataset names for analysis
data_names = ["^GSPC", "^IXIC", "^DJI"]  # Example stock indices from Yahoo Finance
results_df = pd.DataFrame(columns=['Dataset', 'Model ROI', 'Model Sharpe Ratio', 'Close Price ROI', 'Close Price Sharpe Ratio'])

for data_name in data_names:
    print(f"Dataset: {data_name}")
    
    # Reset results lists for each dataset
    all_rois_final_M = []
    all_sharpe_ratios_final_M = []
    all_portfolio_values_final_M = []

    for run_M in range(num_runs_M):
        print(f"Run {run_M + 1}/{num_runs_M}")

        # Execute the specific data collection script for the current dataset
        os.system(f"python DC_Sall_{data_name}.py")  # Assumes existence of a script named 'DC_all_<dataset_name>.py'

        # Initialize or reset variables for this run
        q_table_M = pd.DataFrame(columns=['Sell', 'Buy', 'Hold'])
        buying_price_M = None
        portfolio_status_M = 'cash'  # Start with cash in the portfolio
        total_reward_M = 0
        actions_M = ['Sell', 'Buy', 'Hold']

        # Load the dataset
        data_M = pd.read_csv(f"DC_Semi_{data_name}.csv")
        cash_M = 100000  # Set an example initial cash amount
        shares_M = 0  # Start with zero shares
        # Apply the observe_state function to each row to determine the state
        data_M['state_M'] = data_M.apply(observe_state_M, axis=1)
        portfolio_values_M = []     
        if 'action_M' not in data_M.columns:
            data_M['action_M'] = None  # Initialize action column with None if it doesn't exist
        last_significant_action_M = None  # Track the last significant action (Buy or Sell)
        
        # Main loop for the reinforcement learning process
        for index, row in data_M.iterrows():
            state_M = row['state_M']
            stock_price_M = row['close']
            action_M, cash_M, shares_M = select_action_M(
                state_M, q_table_M, actions_M, exploration_rate_M, cash_M, shares_M, stock_price_M, last_significant_action_M)
            data_M.at[index, 'action_M'] = action_M  # Record the action taken
            next_dc_event_M, next_dc_event_price_M = find_next_dc_event(data_M, index)  # Find the next significant market event
            reward_M = calculate_reward_M(action_M, stock_price_M, buying_price_M, next_dc_event_M, next_dc_event_price_M, portfolio_status_M)  # Calculate the reward based on the action taken
            rewards_M.append(reward_M)  # Store the reward
            next_state_M = data_M['state_M'].iloc[index + 1] if index + 1 < len(data_M) else 'terminal'  # Determine the next state
            q_table_M = learn_M(q_table_M, state_M, action_M, reward_M, next_state_M, learning_rate_M, discount_factor_M, actions_M)  # Update Q-values
    
            # Update last significant action, buying price, and portfolio status as needed
            if action_M in ['Buy', 'Sell']:
                last_significant_action_M = action_M  # Update last significant action
            if action_M == 'Buy':
                buying_price_M = stock_price_M  # Update buying price if action is Buy
                portfolio_status_M = 'stock'  # Change portfolio status to stock
            elif action_M == 'Sell':
                portfolio_status_M = 'cash'  # Change portfolio status to cash if action is Sell
    
            total_reward_M += reward_M  # Accumulate total rewards
            # Calculate and store the daily portfolio value
            daily_portfolio_value_M = cash_M + (shares_M * stock_price_M)  # Calculate daily portfolio value
            data_M.at[index, 'daily_portfolio_value_M'] = daily_portfolio_value_M  # Store daily portfolio value
            portfolio_values_M.append(daily_portfolio_value_M)  # Append daily portfolio value to the list
    
        data_M['rewards_M'] = pd.Series(rewards_M)  # Store all rewards in the DataFrame
    
        # Calculate ROI and Sharpe Ratio for this run
        initial_portfolio_value_M = portfolio_values_M[0]  # Initial portfolio value at the start of the run
        final_portfolio_value_M = portfolio_values_M[-1]  # Final portfolio value at the end of the run
        net_trading_gains_M = final_portfolio_value_M - initial_portfolio_value_M  # Calculate net trading gains
        roi_M = net_trading_gains_M / initial_portfolio_value_M  # Calculate return on investment
    
        daily_returns_M = [portfolio_values_M[i] / portfolio_values_M[i-1] - 1 for i in range(1, len(portfolio_values_M))]  # Calculate daily returns
        expected_return_M = np.mean(daily_returns_M)  # Calculate expected daily return
        std_dev_return_M = np.std(daily_returns_M)  # Calculate standard deviation of daily returns
        rf_annual_M = 0.03  # Define the annual risk-free rate
        rf_daily_M = (1 + rf_annual_M) ** (1/252) - 1  # Convert the annual risk-free rate to daily
        if std_dev_return_M == 0:
            sharpe_ratio_M = 0  # If standard deviation is zero, set Sharpe Ratio to zero
        else:
            sharpe_ratio_M = (expected_return_M - rf_daily_M) / (std_dev_return_M * np.sqrt(252))  # Calculate Sharpe Ratio assuming 252 trading days in a year
    
        # Store results for this run in a list
        results_M.append({
            'run': run_M + 1,
            'total_reward': total_reward_M,
            'q_table': q_table_M.copy()  # Copy Q-table to avoid reference issues
        })
    
        print(f"Run {run_M + 1} completed. Total Reward: {total_reward_M}")  # Print the total reward for this run
        
    # After all runs are completed, average the results
    avg_roi_final_M = np.mean(all_rois_final_M)  # Calculate average ROI over all runs
    avg_sharpe_ratio_final_M = np.mean(all_sharpe_ratios_final_M)  # Calculate average Sharpe Ratio over all runs
    avg_portfolio_values_final_M = np.mean(np.array(all_portfolio_values_final_M), axis=0)  # Calculate average portfolio values over all runs

    # Visualization of the average daily portfolio value
    plt.figure(figsize=(10, 5))
    # Plot the average portfolio values against the dates, assuming the dates align with the index of avg_portfolio_values_final_M
    plt.plot(data_M['date'][:len(avg_portfolio_values_final_M)], avg_portfolio_values_final_M, label='Average Portfolio Value')
    plt.title(f'Average Portfolio Value Over Time ({data_name})')  # Set title for the plot
    plt.xlabel('Date')  # Label for the x-axis
    plt.ylabel('Value')  # Label for the y-axis
    plt.legend()  # Display legend
    plt.show()  # Display the plot

    print(f"Average ROI over 20 runs ({data_name}): {avg_roi_final_M:.2%}")  # Print average ROI over 20 runs
    print(f"Average Sharpe Ratio over 20 runs ({data_name}): {avg_sharpe_ratio_final_M:.4f}")  # Print average Sharpe Ratio over 20 runs
    print()


    # Calculate ROI and Sharpe Ratio for the dataset using close price
    initial_portfolio_value_data_M = data_M['close'].iloc[0]  # Initial close price of the stock
    final_portfolio_value_data_M = data_M['close'].iloc[-1]  # Final close price of the stock
    net_trading_gains_data_M = final_portfolio_value_data_M - initial_portfolio_value_data_M  # Calculate net trading gains based on close prices
    roi_data_M = net_trading_gains_data_M / initial_portfolio_value_data_M  # Calculate ROI based on close prices

    daily_returns_data_M = data_M['close'].pct_change().dropna()  # Calculate daily returns based on close prices
    expected_return_data_M = np.mean(daily_returns_data_M)  # Calculate expected daily return based on close prices
    std_dev_return_data_M = np.std(daily_returns_data_M)  # Calculate standard deviation of daily returns based on close prices
    sharpe_ratio_data_M = (expected_return_data_M - rf_daily_M) / (std_dev_return_data_M * np.sqrt(252))  # Calculate Sharpe Ratio based on close prices

    # Append results to the DataFrame
    results_df = results_df.append({
        'Dataset': data_name,
        'Model ROI': avg_roi_final_M,
        'Model Sharpe Ratio': avg_sharpe_ratio_final_M,
        'Close Price ROI': roi_data_M,
        'Close Price Sharpe Ratio': sharpe_ratio_data_M
    }, ignore_index=True)

# Display the comparison table of results
print("Comparison of Results:")
print(results_df)
results_df.to_csv('results_comparison.assignment2.csv', index=False)  # Save results to a CSV file
