# -*- coding: utf-8
"""
Created on Thu Mar  7 11:25:08 2024

@author: mraja
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def observe_state_M(row):
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
        if row['action'] == 'DT_Overnight' and row['trend'] == 'upward':
            return 'Neutral Upward trend'
        elif row['action'] == 'DT_Overnight' and row['trend'] == 'downward':
            return 'Neutral Downward trend'
        elif row['action'] == 'DT_PreviousDay' and row['trend'] == 'upward':
            return 'Neutral Upward trend'
        elif row['action'] == 'DT_PreviousDay' and row['trend'] == 'downward':
            return 'Neutral Downward trend'
    return 'Unknown State'  # Return this if none of the above conditions are met

def check_state_exist_M(q_table_M, state_M, actions_M):
    if state_M not in q_table_M.index:
        # If the state is not in the Q-table, add it with initial values for all actions
        new_state_M = pd.Series([0]*len(actions_M), index=q_table_M.columns, name=state_M)
        q_table_M = q_table_M.append(new_state_M)
    return q_table_M

def select_action_M(state_M, q_table_M, actions_M, exploration_rate_M, cash_M, shares_M, stock_price_M,
                    last_significant_action=None):
    q_table_M = check_state_exist_M(q_table_M, state_M, actions_M)
    
        # Determine the allowable actions based on last significant action (ignoring 'Hold')
    if last_significant_action == 'Buy':
        allowable_actions_M = ['Hold', 'Sell']
    elif last_significant_action == 'Sell':
        allowable_actions_M = ['Hold', 'Buy']
    else:  # If last significant action was None or 'Hold', all actions are allowed
        allowable_actions_M = actions_M
        
    if np.random.uniform() < exploration_rate_M:
        action_M = np.random.choice(allowable_actions_M)
    else:
        state_actions_M = q_table_M.loc[state_M, allowable_actions_M]
        action_M = np.random.choice(state_actions_M[state_actions_M == np.max(state_actions_M)].index)
    
    if action_M == 'Buy':
        shares_bought_M = cash_M / stock_price_M
        cash_M = 0
        shares_M += shares_bought_M
    elif action_M == 'Sell':
        cash_M += shares_M * stock_price_M
        shares_M = 0
    
    return action_M, cash_M, shares_M

def learn_M(q_table_M, state_M, action_M, reward_M, next_state_M, learning_rate_M, discount_factor_M, actions_M):
    q_table_M = check_state_exist_M(q_table_M, state_M, actions_M)
    q_table_M = check_state_exist_M(q_table_M, next_state_M, actions_M)
    
    q_predict_M = q_table_M.loc[state_M, action_M]
    if next_state_M != 'terminal':
        q_target_M = reward_M + discount_factor_M * q_table_M.loc[next_state_M, :].max()
    else:
        q_target_M = reward_M
    q_table_M.loc[state_M, action_M] += learning_rate_M * (q_target_M - q_predict_M)
    
    return q_table_M

def find_next_dc_event(data_M, current_index):
    for next_index in range(current_index + 1, len(data_M)):
        next_row = data_M.iloc[next_index]
        if next_row['dc_event'] in ['Upturn DC Event Detected', 'Downturn DC Event Detected']:
            return next_row['dc_event'], next_row['close']
    return None, None


def calculate_reward_M(action_M, stock_price_M, buying_price_M, next_dc_event_M, next_dc_event_price_M, portfolio_status_M):
    if pd.isnull(next_dc_event_price_M) or pd.isnull(next_dc_event_M) or buying_price_M is None:
        return 0  # Return 0 reward if next DC event or its price is not available or buying price is None

    if action_M == 'Sell':
        return (stock_price_M - next_dc_event_price_M) / stock_price_M if buying_price_M != 0 else 0
    elif action_M == 'Buy':
        return (next_dc_event_price_M - stock_price_M) / stock_price_M if stock_price_M != 0 else 0
    elif action_M == 'Hold':
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
    return 0


num_runs_M = 2
results_M = []
exploration_rate_M = 0.5  # Define your exploration rate here
learning_rate_M = 0.3  # Define your learning rate here
discount_factor_M = 0.9 # Define your discount factor here
np.random.seed(42)

all_rois_final_M = []
all_sharpe_ratios_final_M = []
all_portfolio_values_final_M = []
rewards_M = []


# Define the dataset names
data_names = ["^GSPC", "^IXIC", "^DJI"]
results_df = pd.DataFrame(columns=['Dataset', 'Model ROI', 'Model Sharpe Ratio', 'Close Price ROI', 'Close Price Sharpe Ratio'])

for data_name in data_names:
    print(f"Dataset: {data_name}")
    
    all_rois_final_M = []
    all_sharpe_ratios_final_M = []
    all_portfolio_values_final_M = []

    for run_M in range(num_runs_M):
        print(f"Run {run_M + 1}/{num_runs_M}")

        # Run the script for the current dataset
        os.system(f"python DC_Sall_{data_name}.py")  # This runs 'DC_all_<dataset_name>.py'

        # Initialize or reset variables for this run
        q_table_M = pd.DataFrame(columns=['Sell', 'Buy', 'Hold'])
        buying_price_M = None
        portfolio_status_M = 'cash'  # Assuming starting with cash
        total_reward_M = 0
        actions_M = ['Sell', 'Buy', 'Hold']

    
        # Load the data
        data_M = pd.read_csv(f"DC_Semi_{data_name}.csv")
        cash_M = 100000  # Example initial cash amount
        shares_M = 0  # Example initial shares amount
        # Apply the observe_state function to each row of the dataframe
        data_M['state_M'] = data_M.apply(observe_state_M, axis=1)
        portfolio_values_M = []     
        if 'action_M' not in data_M.columns:
            data_M['action_M'] = None  # Initialize with None
        last_significant_action_M = None  
        
        # Main loop for the learning process
        for index, row in data_M.iterrows():
            state_M = row['state_M']
            stock_price_M = row['close']
            action_M, cash_M, shares_M = select_action_M(
                state_M, q_table_M, actions_M, exploration_rate_M, cash_M, shares_M, stock_price_M, last_significant_action_M)
            data_M.at[index, 'action_M'] = action_M
            next_dc_event_M, next_dc_event_price_M = find_next_dc_event(data_M, index)
            reward_M = calculate_reward_M(action_M, stock_price_M, buying_price_M, next_dc_event_M, next_dc_event_price_M, portfolio_status_M)
            rewards_M.append(reward_M)
            next_state_M = data_M['state_M'].iloc[index + 1] if index + 1 < len(data_M) else 'terminal'
            q_table_M = learn_M(q_table_M, state_M, action_M, reward_M, next_state_M, learning_rate_M, discount_factor_M, actions_M)
    
            # Update last_significant_action_M only if it's Buy or Sell
            if action_M in ['Buy', 'Sell']:
                last_significant_action_M = action_M
                
            # Update buying_price and portfolio_status as needed
            if action_M == 'Buy':
                buying_price_M = stock_price_M
                portfolio_status_M = 'stock'
            elif action_M == 'Sell':
                portfolio_status_M = 'cash'
    
            total_reward_M += reward_M
            # Calculate and store the daily portfolio value in the DataFrame
            daily_portfolio_value_M = cash_M + (shares_M * stock_price_M)
            data_M.at[index, 'daily_portfolio_value_M'] = daily_portfolio_value_M
            portfolio_values_M.append(daily_portfolio_value_M)
    
        data_M['rewards_M'] = pd.Series(rewards_M)
    
        # Calculate ROI and Sharpe Ratio for this run
        initial_portfolio_value_M = portfolio_values_M[0]
        final_portfolio_value_M = portfolio_values_M[-1]
        net_trading_gains_M = final_portfolio_value_M - initial_portfolio_value_M
        roi_M = net_trading_gains_M / initial_portfolio_value_M
    
        daily_returns_M = [portfolio_values_M[i] / portfolio_values_M[i-1] - 1 for i in range(1, len(portfolio_values_M))]
        expected_return_M = np.mean(daily_returns_M)
        std_dev_return_M = np.std(daily_returns_M)
        rf_annual_M = 0.03  # Risk-free rate
        rf_daily_M = (1 + rf_annual_M) ** (1/252) - 1  # Convert annual risk-free rate to daily
        if std_dev_return_M == 0:
            sharpe_ratio_M = 0
        else:
            sharpe_ratio_M = (expected_return_M - rf_daily_M) / (std_dev_return_M * np.sqrt(252))  # Assuming 252 trading days in a year
    
        # Store results for this run
        all_rois_final_M.append(roi_M)
        all_sharpe_ratios_final_M.append(sharpe_ratio_M)
        all_portfolio_values_final_M.append(portfolio_values_M)
       
        # Store results of this run
        results_M.append({
            'run': run_M + 1,
            'total_reward': total_reward_M,
            'q_table': q_table_M.copy()
        })
    
        print(f"Run {run_M + 1} completed. Total Reward: {total_reward_M}")
        
    # After all runs are completed, average the results
    avg_roi_final_M = np.mean(all_rois_final_M)
    avg_sharpe_ratio_final_M = np.mean(all_sharpe_ratios_final_M)
    avg_portfolio_values_final_M = np.mean(np.array(all_portfolio_values_final_M), axis=0)

    # Visualization of the average daily portfolio value
    plt.figure(figsize=(10, 5))
    # Plot the average portfolio values against the dates, assuming the dates align with the index of avg_portfolio_values_final_M
    plt.plot(data_M['date'][:len(avg_portfolio_values_final_M)], avg_portfolio_values_final_M, label='Average Portfolio Value')
    plt.title(f'Average Portfolio Value Over Time ({data_name})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    print(f"Average ROI over 20 runs ({data_name}): {avg_roi_final_M:.2%}")
    print(f"Average Sharpe Ratio over 20 runs ({data_name}): {avg_sharpe_ratio_final_M:.4f}")
    print()


   # Calculate ROI and Sharpe Ratio for the dataset using close price
    initial_portfolio_value_data_M = data_M['close'].iloc[0]
    final_portfolio_value_data_M = data_M['close'].iloc[-1]
    net_trading_gains_data_M = final_portfolio_value_data_M - initial_portfolio_value_data_M
    roi_data_M = net_trading_gains_data_M / initial_portfolio_value_data_M

    daily_returns_data_M = data_M['close'].pct_change().dropna()
    expected_return_data_M = np.mean(daily_returns_data_M)
    std_dev_return_data_M = np.std(daily_returns_data_M)
    sharpe_ratio_data_M = (expected_return_data_M - rf_daily_M) / (std_dev_return_data_M * np.sqrt(252))

    results_df = results_df.append({
        'Dataset': data_name,
        'Model ROI': avg_roi_final_M,
        'Model Sharpe Ratio': avg_sharpe_ratio_final_M,
        'Close Price ROI': roi_data_M,
        'Close Price Sharpe Ratio': sharpe_ratio_data_M
    }, ignore_index=True)

# Display the comparison table
print("Comparison of Results:")
print(results_df)
results_df.to_csv('results_comparison.assignment2.csv', index=False)




