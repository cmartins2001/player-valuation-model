'''
Principle Component Analysis (PCA) workflow script
Last Update: 4/17/2024
'''

# Import libraries:
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error, r2_score

# Directory variables:
repo_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
source_data_dir = os.path.join(repo_dir, "analysis-data")
results_dir = os.path.join(repo_dir, 'analysis-results')


# Function that conducts a train-test split:
def split_data(df, target_var_name : str = 'market_value_in_eur', test_size : float = 0.2, seed_value : int = 42):
    
    # Create x, y vars:
    y = df[target_var_name]
    X = df.drop(columns=target_var_name) # everything but the target

    # Convert all feature column names to strings for PCA fitting:
    X.columns = X.columns.astype(str)
    
    # Create test and train datasets:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed_value)

    return X_train, X_test, y_train, y_test


# Function that plots a heatmap of PCA components and original features:
def plot_PCA_heatmap(components, x_train_df, file_suffix : str = 'pca'):
    
    # Create the dataframe:
    pca_components_df = pd.DataFrame(components, columns=x_train_df.columns)

    # Generate and store a heatmap of the PCA components and original features:
    plt.figure(figsize=(12, 8))
    sns.heatmap(pca_components_df, cmap='coolwarm', fmt=".2f") #annot=True, 
    plt.title('Loadings of Original Features on Principal Components')
    plt.xlabel('Original Features')
    plt.ylabel('Principal Components')
    # Save the plot as a file:
    plt.savefig(os.path.join(os.path.join(source_data_dir, 'heatmaps'), f'{file_suffix}_heatmap.png'))
    plt.close()


# Function that FITS an OLS model on training PCA data:
def fit_ols_on_pca(x_train_pca, x_test_pca, y_train):
    '''
    This function returns the .fit() method of the OLS() statsmodels call.
    The OLS output can be printed in main() using "print(fit_ols_on_pca(args..).summary())"
    '''

    # Concatenate a constant column to each dataframe:
    x_train_constant = sm.add_constant(x_train_pca)
    x_test_constant = sm.add_constant(x_test_pca)

    # Fit the OLS model:
    train_ols_model = sm.OLS(y_train, x_train_pca) # Y from training data principle components from training data

    # Model results:
    train_ols_results = train_ols_model.fit()

    return train_ols_results


# Function that generates a residual plot for a trained model fit:
def resid_plot(model_results, model_name : str):

    # Create a residuals dataframe:
    residuals_df = pd.DataFrame({f'{model_name} Fitted Values' : model_results.fittedvalues,
                                  f'{model_name} Residuals' : model_results.resid})
    
    plt.figure(figsize=(8, 6))
    sns.residplot(x=f'{model_name} Fitted Values', y=f'{model_name} Residuals', data=residuals_df, color='cornflowerblue')
    plt.title(f'{model_name} Model - Residual Plot')
    plt.xlabel(f'{model_name} Fitted Values')
    plt.ylabel(f'{model_name} Residuals')
    # Save the plot as a file:
    plt.savefig(os.path.join(os.path.join(results_dir, 'performance-eval-graphs'), f'{model_name}_residplot.png'))
    plt.close()


# Function that plots model predictions versus test actuals of the target variable:
def pred_vs_actuals_plot(test_actuals, model_predictions, model_name : str):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=test_actuals, y=model_predictions, color='cornflowerblue', alpha=0.5)
    sns.lineplot(x=test_actuals, y=test_actuals, color='orangered', linestyle='--', label='Perfect Predictions')
    plt.title(f'{model_name} Predictions vs Actuals')
    plt.xlabel('Actuals')
    plt.ylabel(f'{model_name} Predictions')
    plt.savefig(os.path.join(os.path.join(results_dir, 'performance-eval-graphs'), f'{model_name}_pred_vs_actuals.png'))
    plt.close()


def main():

    # Import standardized master file:
    scaled_df = pd.read_csv(os.path.join(source_data_dir, 'master_file_standardized.csv')).drop(columns='Unnamed: 0')

    # Conduct the train-test split:
    X_train_df, X_test_df, y_train_df, y_test_df = split_data(scaled_df)

    # Create an instance of the PCA model:
    pca = PCA(0.95)

    # Fit the model on the training features:
    pca_fit = pca.fit(X_train_df)

    # Generate a seaborn heatmap:
    # plot_PCA_heatmap(pca_fit.components_, X_train_df, file_suffix='pca')

    # Transform the training and testing features using the learned transformation:
    X_train_pca = pca.transform(X_train_df)
    X_test_pca = pca.transform(X_test_df)

    # ------------ Start of PCA-OLS model fitting ------------

    # Add a constant column to the pca feature dataframes:
    X_train_pca_with_constant = sm.add_constant(X_train_pca)
    X_test_pca_with_constant = sm.add_constant(X_test_pca)

    # # # Fit the OLS model:
    # train_ols_model = sm.OLS(y_train_df, X_train_pca_with_constant) # Y from training data against principle components from training data
    # train_ols_results = train_ols_model.fit()

    # # # OLS Regression Output:
    # print(train_ols_results.summary())

    # # Use the OLS model to predict the TEST data:
    # ols_test_pred = train_ols_results.predict(X_test_pca_with_constant)

    # # OLS Error Metrics:
    # ols_mae = mean_absolute_error(y_test_df, ols_test_pred)
    # ols_mse = mean_squared_error(y_test_df, ols_test_pred)
    # ols_rmse = np.sqrt(ols_mse)
    # ols_mape = mean_absolute_percentage_error(y_test_df, ols_test_pred)
    # ols_MedAE = median_absolute_error(y_test_df, ols_test_pred)
    # ols_errors = [ols_mae, ols_mse, ols_rmse, ols_mape, ols_MedAE]

    # # Print OLS error metrics:
    # for var, name in zip(ols_errors, ['MAE', 'MSE', 'RMSE', 'RMSE', 'MAPE', 'MedAE']):
    #     print(f'\nOLS with PCA Component Predictors - {name}: {var:.4f}\n')

    # OLS performance evaluation visuals:

    # Plot OLS training model residuals:
    # resid_plot(train_ols_results, model_name='PCA-OLS')

    # Plot OLS test predictions versus actuals:
    # pred_vs_actuals_plot(y_test_df, ols_test_pred, model_name='PCA-OLS')

    # ------------ Start of PCA-Random Forest model fitting ------------

    # Initialize the random forest model:
    rf_pca = RandomForestRegressor(n_estimators = 100, random_state = 42)

    # Train the random forest model on the training PCA data:
    train_rf_results = rf_pca.fit(X_train_pca, y_train_df)

    # Predict the test PCA data:
    rf_pred = train_rf_results.predict(X_test_pca)

    # RF Error Metrics:
    rf_mae = mean_absolute_error(y_test_df, rf_pred)
    rf_mse = mean_squared_error(y_test_df, rf_pred)
    rf_rmse = np.sqrt(rf_mse)
    rf_mape = mean_absolute_percentage_error(y_test_df, rf_pred)
    rf_MedAE = median_absolute_error(y_test_df, rf_pred)
    rf_r_squared = r2_score(y_test_df, rf_pred)
    rf_errors = [rf_mae, rf_mse, rf_rmse, rf_mape, rf_MedAE]

    # Print OLS error metrics:
    for var, name in zip(rf_errors, ['MAE', 'MSE', 'RMSE', 'RMSE', 'MAPE', 'MedAE']):
        print(f'\nRandom Forest with PCA Component Predictors - {name}: {var:.4f}\n')

    # Random Forest performance evaluation visuals:
    pred_vs_actuals_plot(y_test_df, rf_pred, 'PCA-Random Forest')


if __name__ == '__main__':
    main()
