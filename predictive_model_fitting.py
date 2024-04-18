'''
Estimating predictive models on the non-standardized master file
Last Update: 4/17/2024
'''

# Import libraries:
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error, r2_score
# Import my functions from PCA script:
from pca_model_fitting import split_data, model_performance_metrics, resid_plot, pred_vs_actuals_plot

# Directory variables:
repo_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
source_data_dir = os.path.join(repo_dir, "analysis-data")
results_dir = os.path.join(repo_dir, 'analysis-results')

# Big teams list:
big_teams = ['Arsenal', 'Manchester City', 'Manchester Utd', 'Tottenham', 'Chelsea', 'Liverpool', 'Real Madrid', 'Barcelona', 'Atlético Madrid', 'Paris S-G', 'Juventus', 'Milan', 'Inter']

# League ID and name mapping variables:
league_ids = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'ITA-Serie A']
league_names = ['Premier League', 'La Liga', 'Ligue 1', 'Serie A']
league_names_dict = dict(zip(league_ids, league_names))


# Function that adds the log(market value) column to the data:
def add_log_mkt_val_col(df):
    df['log_mkt_val'] = np.log(df['market_value_in_eur'])
    return df


# Function that adds the "Big Team" dummy variable to the data:
def add_big_team_col(df):
    df['Big Team'] = df['team'].isin(big_teams).astype(int)
    return df


# Function that adds a clean league name column based on league id:
def add_league_name_col(df):
    df['league_name'] = df['league'].map(league_names_dict)
    return df


def main():
    
    # Import master file:
    raw_df = pd.read_csv(os.path.join(source_data_dir, 'master_file_with_dummies.csv')).drop(columns='Unnamed: 0')

    # Add the 'Big Team' column:
    raw_df = add_big_team_col(raw_df)
    
    # Add league name, generate dummies, drop all string columns:
    raw_df = add_league_name_col(raw_df)
    league_dummies_df = pd.get_dummies(raw_df['league_name'], drop_first=False)
    master_df = pd.concat([raw_df, league_dummies_df], axis=1)
    master_df = master_df.drop(columns=['league_name', 'league', 'season', 'team', 'player_name', 'nationality', 'position'])

    # Add the LN(market value) column:
    master_df = add_log_mkt_val_col(master_df)
    
    # Conduct the train-test split with logged market value as the target:
    X_train_df_log, X_test_df_log, y_train_df_log, y_test_df_log = split_data(master_df, 'log_mkt_val')

    # Conduct the train-test split with market value as the target:
    X_train_df, X_test_df, y_train_df, y_test_df = split_data(master_df)

    # ------------ Start of OLS model fitting with both targets ------------
    train_ols_model = sm.OLS(y_train_df, X_train_df)
    train_ols_model_log = sm.OLS(y_train_df_log, X_train_df_log)
    train_ols_results = train_ols_model.fit()
    train_ols_results_log = train_ols_model_log.fit()

    # OLS Regression Output:
    # print(train_ols_results.summary())
    print(train_ols_results_log.summary())

    # Use the OLS model to predict the TEST data:
    ols_test_pred = train_ols_results.predict(X_test_df)
    ols_test_pred_log = train_ols_results_log.predict(X_test_df_log)

    # # Print OLS error metrics:
    # model_performance_metrics(y_test_df, ols_test_pred, 'OLS')
    model_performance_metrics(y_test_df_log, ols_test_pred_log, 'OLS - Logged Outcome')

    # # OLS residual plot:
    # resid_plot(train_ols_results, model_name='OLS')
    resid_plot(train_ols_results_log, model_name='OLS - Logged Outcome')

    # # Plot OLS test predictions versus actuals:
    # pred_vs_actuals_plot(y_test_df, ols_test_pred, model_name='OLS')
    pred_vs_actuals_plot(y_test_df_log, ols_test_pred_log, model_name='OLS - Logged Outcome')

    # ------------ Start of Lasso regression model fitting ------------


if __name__ == "__main__":
    main()
