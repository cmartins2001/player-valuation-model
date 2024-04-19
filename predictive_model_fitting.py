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
from sklearn.ensemble import RandomForestRegressor
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

    # Drop the Y column that was NOT selected to avoid prediction bias:
    if target_var == 'market_value_in_eur':
        if 'log_mkt_val' in df.columns:
            df = df.drop(columns='log_mkt_val')
    elif target_var == 'log_mkt_val':
        if 'market_value_in_eur' in df.columns:
            df = df.drop(columns='market_value_in_eur')
    else:
        print('\n***INVALID TARGET VARIABLE***\n')

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

    # # Concatenate the scaled_df with the dummy columns:
    # final_scaled_df = pd.concat([scaled_df, df.iloc[:, first_dummy_loc:]], axis=1)

    return scaled_df # TESTING TO SEE IF LASSO WORKS BETTER IF I EXCLUDE THE DUMMIES AND LN(MKT VALUE) COLUMN


# Function that fits and runs a Lasso regression model:
def fit_and_run_LASSO(df, target_var, clean_y_name):

    # Create a standardized version of the master dataframe:
    standardized_df = standardize_numeric_vars(df)

    # Conduct the test-train split with the standardized dataframe:
    X_train, X_test, y_train, y_test = split_data(standardized_df, target_var, test_size=.2, seed_value=42)

    # Initialize the Lasso model:
    lasso_model = Lasso(max_iter=1000) # alpha=0.1, use this if can't tune the parameter; max_iter=1000 to avoid convergence error

    # Create CV grid search space:
    search_space = {'alpha' : [0.001, 0.01, 0.1, 1, 10]}

    # Initialize GridSearchCV:
    grid_search = GridSearchCV(lasso_model, search_space, cv=5, scoring='neg_mean_squared_error')

    # Fit GridSearchCV on training data:
    grid_search.fit(X_train, y_train)

    # Get the best model after grid search:
    best_lasso_model = grid_search.best_estimator_

    # Print the parameter used:
    print(f'\nLasso Parameter Selected: {grid_search.best_params_}\n')

    # Fit on training data:
    # trained_lasso = lasso_model.fit(X_train, y_train)

    # Extract important features:
    # selected_features = X_train.columns[trained_lasso.coef_ != 0]
    selected_features = X_train.columns[best_lasso_model.coef_ != 0]

    # Predict on test data:
    # pred_lasso = trained_lasso.predict(X_test)
    pred_lasso = best_lasso_model.predict(X_test)

    # Lasso performance metrics:
    model_performance_metrics(y_test, pred_lasso, f'Tuned Lasso - Y = {clean_y_name}')

    # Predicted vs. actuals plot:
    pred_vs_actuals_plot(y_test, pred_lasso, f'Tuned Lasso - Y = {clean_y_name}')

    # Return important lasso features for analysis:
    return selected_features


# Function that fits a random forest regressor on a dataframe:
def fit_and_run_RANDOMFOREST(df, target_var, clean_y_name):

    # Drop the Y column that was NOT selected:
    if target_var == 'market_value_in_eur':
        df = df.drop(columns='log_mkt_val')
    elif target_var == 'log_mkt_val':
        df = df.drop(columns='market_value_in_eur')
    else:
        print('\n***INVALID TARGET VARIABLE***\n')
    
    # Conduct the test-train split with the standardized dataframe:
    X_train, X_test, y_train, y_test = split_data(df, target_var, test_size=.2, seed_value=42)

    # Create CV grid search space:
    search_space = {'n_estimators' : [25, 50, 100], #, 200, 500],
                    'max_depth' : [None, 3, 5], #, 10, 20],
                    'min_samples_split' : [2, 5, 9] #, 10]
                    }
    
    # Initialize the RF model with the tuned parameters; CV process is commented out for reference:
    rf_model = RandomForestRegressor(n_estimators=100, min_samples_split=2, max_depth=None, random_state=42) # n_estimators=100, min_samples_split=5, max_depth=None

    # Initialize GridSearchCV:
    # grid_search = GridSearchCV(estimator=rf_model, 
    #                            param_grid=search_space, 
    #                            cv=5, 
    #                            scoring='neg_mean_squared_error',
    #                            refit=True,
    #                            n_jobs=-1) # use all available CPU cores
    
    # Fit GridSearchCV on training data:
    # grid_search.fit(X_train, y_train)

    # # Best estimator:
    # best_rf = grid_search.best_estimator_

    # # Print best parameters:
    # print(f"\nRF Parameters Used: \n{grid_search.best_params_}\n")

    # # Predict on test data:
    # pred_rf = best_rf.predict(X_test)

    # Train the model on the training data:
    train_rf_results = rf_model.fit(X_train, y_train)

    # Extract the feature importances:
    rf_importances = train_rf_results.feature_importances_

    # Get feature names
    feature_names = X_train.columns

    # Combine feature names and importances into a DataFrame
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_importances})

    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Print top N important features
    top_n = 20  # Set the number of top features to display
    print(f"\nTuned Random Forest Model - Top {top_n} Important Features:\n")
    print(feature_importance_df.head(top_n))

    # Predict the test data:
    pred_rf = train_rf_results.predict(X_test)

    # Error metrics:
    model_performance_metrics(y_test, pred_rf, f'Tuned Random Forest - Y = {clean_y_name}')

    # Predicted vs. actuals:
    pred_vs_actuals_plot(y_test, pred_rf, f'Tuned Random Forest - Y = {clean_y_name}')

    # Output the important features dataframe for use later in OLS model building:
    return feature_importance_df


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

    # Fit and run untuned LASSO models on both outcomes:
    # for outcome, clean_outcome in zip(['market_value_in_eur', 'log_mkt_val'], ['Market Value (euros)', 'ln(Market Value)']):    
    #     fit_and_run_LASSO(master_df, outcome, clean_outcome)
    #     print(f'Lasso Model with Y = {clean_outcome} - Important Features: \n{fit_and_run_LASSO(master_df, outcome, clean_outcome)}')

    # ------------ Start of Random Forest model fitting ------------
    # for outcome, clean_outcome in zip(['market_value_in_eur', 'log_mkt_val'], ['Market Value (euros)', 'ln(Market Value)']):
    #     fit_and_run_RANDOMFOREST(master_df, outcome, clean_outcome)

    # ------------ Start of OLS modeling with Random Forests' important features ------------
    
    # Slice master dataframe to only the RF features and the two outcome variables:
    # important_feature_list = fit_and_run_RANDOMFOREST(master_df, 'market_value_in_eur', 'Market Value (euros)')['Feature'].to_list()[0:25] # Top 25 most important features only
    
    # # Hard coded the RF features list to avoid running the algorithm every time:
    # important_feature_list = ['Big Team', 'age', 'npxG+xAG', 'onxG', 'xG+/-', 'SCA', 'Premier League', 'Att 3rd.1', 'Sh', 'GCA', 'Dis', '+/-', 'onG', 'PPM', 'SoT/90', 'Att Pen', 'Cmp%.3', 'PassLive', 'npxG+xAG.1', 'Sh/90', 'G+A', 'xAG', 'TotDist.1', '+/-90', '1/3']

    # # Add the outcomes to the list:
    # important_feature_list.append('market_value_in_eur')
    # important_feature_list.append('log_mkt_val')

    # # Slice the main dataframe:
    # rf_feature_sliced_df = master_df[important_feature_list] 

    # # Run an OLS model using the sliced dataframe:
    # for outcome, clean_outcome in zip(['market_value_in_eur', 'log_mkt_val'], ['MV (euros) - RF Features Only', 'ln(Market Value) - RF Features Only']):
    #     fit_and_run_OLS(rf_feature_sliced_df, outcome, clean_outcome)

    # ------------ Start of OLS modeling on position-specific datasets ------------

    # Re-import master file:
    pos_raw_df = pd.read_csv(os.path.join(source_data_dir, 'master_file_with_dummies.csv')).drop(columns='Unnamed: 0')

    # Add the LN(market value) column:
    pos_raw_df = add_log_mkt_val_col(pos_raw_df)
    
    # Add the 'Big Team' column:
    pos_raw_df = add_big_team_col(pos_raw_df)
    
    # Add league name, generate dummies, drop all string columns:
    pos_raw_df = add_league_name_col(pos_raw_df)
    pos_league_dummies_df = pd.get_dummies(pos_raw_df['league_name'], drop_first=True) # one-hot encoding, drop first to create a reference category
    pos_master_df = pd.concat([pos_raw_df, pos_league_dummies_df], axis=1)
    pos_master_df = pos_master_df.drop(columns=['league_name', 'league', 'season', 'team', 'player_name', 'nationality'])

    # Import the position-level data:
    for pos in ['DF' 'MF', 'FW']:

        # Slice the master dataframe by position:
        pos_df = pos_master_df[pos_master_df['position'].str.contains(pos)]

        # Remove the position column and print status messages:
        pos_df = pos_df.drop(columns='position')
        print(f'\nWorking on the {pos} data...\n')
        print(f'\n{pos} Dataframe Row total: {pos_df.shape[0]}\n')

        # Run the OLS model on the logged outcome:
        fit_and_run_OLS(pos_df, 'log_mkt_val', f'ln(Market Value) - {pos} Only')
    

if __name__ == "__main__":
    main()
