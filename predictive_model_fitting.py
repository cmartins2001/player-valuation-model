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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
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


# Function that fits/predicts OLS an model on a dataframe:
def fit_and_run_OLS(df, target_var, clean_y_name):
    '''
    This function accepts a master dataframe with dummies, conducts a test-train split,
    fits/predicts an OLS regression, then outputs error metrics and model performance
    visuals into the proper GitHub directories.
    '''

    # Call the test-train split function:
    X_train, X_test, y_train, y_test = split_data(df, target_var, test_size=.2, seed_value=42)

    # Train OLS model:
    train_ols_model = sm.OLS(y_train, X_train)
    train_ols_results = train_ols_model.fit()

    # Print output:
    print(train_ols_results.summary())

    # Predict on test data:
    ols_test_pred = train_ols_results.predict(X_test)

    # Print Error Metrics:
    model_performance_metrics(y_test, ols_test_pred, f'OLS - Y = {clean_y_name}')

    # Residual plot:
    resid_plot(train_ols_results, model_name=f'OLS - Y = {clean_y_name}')

    # Predicted vs actuals plot:
    pred_vs_actuals_plot(y_test, ols_test_pred, model_name=f'OLS - Y = {clean_y_name}')


# Function that standardizes all non-dummy variables in the master file:
def standardize_numeric_vars(df):

    # Initialize the scaler:
    scaler = StandardScaler()

    # Define scaling column range:
    first_dummy_loc = df.columns.get_loc('Alavés')
    data_to_scale = df.iloc[:, :first_dummy_loc] # everything but the first dummy and onwards

    # Fit and transform the selected columns:
    scaled_data = scaler.fit_transform(data_to_scale)

    # Create a DataFrame from the scaled data:
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns[:first_dummy_loc])

    # Concatenate the scaled_df with the dummy columns:
    final_scaled_df = pd.concat([scaled_df, df.iloc[:, first_dummy_loc:]], axis=1)

    return final_scaled_df


# Function that fits and runs a Lasso regression model:
def fit_and_run_LASSO(df, target_var, clean_y_name):

    # Create a standardized version of the master dataframe:
    standardized_df = standardize_numeric_vars(df)

    # Conduct the test-train split with the standardized dataframe:
    X_train, X_test, y_train, y_test = split_data(standardized_df, target_var, test_size=.2, seed_value=42)

    # Initialize the Lasso model:
    lasso_model = Lasso(alpha=0.1)

    # Fit on training data:
    trained_lasso = lasso_model.fit(X_train, y_train)

    # Extract important features:
    selected_features = X_train.columns[trained_lasso.coef_ != 0]

    # Predict on test data:
    pred_lasso = trained_lasso.predict(X_test)

    # Lasso performance metrics:
    model_performance_metrics(y_test, pred_lasso, f'Lasso - Y = {clean_y_name}')

    # Predicted vs. actuals plot:
    pred_vs_actuals_plot(y_test, pred_lasso, f'Lasso - Y = {clean_y_name}')

    # Return important lasso features for analysis:
    return selected_features



def main():
    
    # Import master file:
    raw_df = pd.read_csv(os.path.join(source_data_dir, 'master_file_with_dummies.csv')).drop(columns='Unnamed: 0')

    # Add the LN(market value) column:
    raw_df = add_log_mkt_val_col(raw_df)
    
    # Add the 'Big Team' column:
    raw_df = add_big_team_col(raw_df)
    
    # Add league name, generate dummies, drop all string columns:
    raw_df = add_league_name_col(raw_df)
    league_dummies_df = pd.get_dummies(raw_df['league_name'], drop_first=True) # one-hot encoding, drop first to create a reference category
    master_df = pd.concat([raw_df, league_dummies_df], axis=1)
    master_df = master_df.drop(columns=['league_name', 'league', 'season', 'team', 'player_name', 'nationality', 'position'])

    # ------------ Start of OLS model fitting with both targets ------------

    # Iterate OLS model fitting over different outcomes:
    # for outcome, clean_outcome in zip(['market_value_in_eur', 'log_mkt_val'], ['Market Value (euros)', 'ln(Market Value)']):
    #     fit_and_run_OLS(master_df, outcome, clean_outcome)

    # ------------ Start of Lasso regression model fitting ------------

    # test the lasso fitting function:
    fit_and_run_LASSO(master_df, 'market_value_in_eur', 'Market Value (euros)')
    fit_and_run_LASSO(master_df, 'log_mkt_val', 'LN(Market Value)')


if __name__ == "__main__":
    main()
