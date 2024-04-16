'''
Principle Component Analysis (PCA) workflow script
Last Update: 4/16/2024
'''

# Import libraries:
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Directory variables:
repo_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
source_data_dir = os.path.join(repo_dir, "analysis-data")


# Function that conducts a train-test split:
def split_data(df, target_var_name : str = 'market_value_in_eur', test_size : float = 0.2):
    
    # Create x, y vars:
    y = df[target_var_name]
    X = df.drop(columns=target_var_name) # everything but the target

    # Convert all feature column names to strings for PCA fitting:
    X.columns = X.columns.astype(str)
    
    # Create test and train datasets:
    seed_value = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed_value)

    return X_train, X_test, y_train, y_test


# Function that plots a heatmap of PCA components and original features:
def plot_PCA_heatmap(components, x_train_df):
    
    # Create the dataframe:
    pca_components_df = pd.DataFrame(components, columns=x_train_df.columns)

    # Plot a heatmap of the PCA components and original features:
    plt.figure(figsize=(12, 8))
    sns.heatmap(pca_components_df, cmap='coolwarm', fmt=".2f") #annot=True, 
    plt.title('Loadings of Original Features on Principal Components')
    plt.xlabel('Original Features')
    plt.ylabel('Principal Components')
    plt.show()


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


def main():

    # Import standardized master file:
    scaled_df = pd.read_csv(os.path.join(source_data_dir, 'master_file_standardized.csv'))

    # Position-level iteration:
    for position in ['DF', 'MF', 'FW']:

        position_scaled_df = scaled_df[scaled_df]
        # do this in the get_analysis script... 

    # Conduct the train-test split:
    X_train_df = split_data(scaled_df)[0]
    X_test_df = split_data(scaled_df)[1]
    y_train_df = split_data(scaled_df)[2]
    y_test_df = split_data(scaled_df)[3]

    # Create an instance of the PCA model:
    pca = PCA(0.95)

    # Fit the model on the training features:
    pca.fit(X_train_df)

    # Generate a seaborn heatmap:
    plot_PCA_heatmap(pca.components_, X_train_df)
    plt.savefig(os.path.join(os.path.join(source_data_dir, 'heatmaps'), f'pca_heatmap.png'))
    plt.close()

    # Transform the training and testing features using the learned transformation:
    X_train_pca = pca.transform(X_train_df)
    X_test_pca = pca.transform(X_test_df)
