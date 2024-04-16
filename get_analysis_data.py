'''
Script that cleans the fbref-dw master file and generates two outputs:
(1) a cleaned master file with the covariate dummies
(2) a standardized master file with only quantitative data for PCA
Last update: 4/16/2024
'''

# Import libraries:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# Graph output styling from matplotlib:
plt.style.use('fivethirtyeight')

# Directory variables:
repo_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
source_data_dir = os.path.join(repo_dir, "fbref-dw-merges")
output_dir = os.path.join(repo_dir, "analysis-data")

# League ID and name mapping variables:
league_ids = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'ITA-Serie A']
league_names = ['Premier League', 'La Liga', 'Ligue 1', 'Serie A']
league_names_dict = dict(zip(league_ids, league_names))

# Big teams list:
big_teams = ['Arsenal', 'Manchester City', 'Manchester Utd', 'Tottenham', 'Chelsea', 'Liverpool', 'Real Madrid', 'Barcelona', 'Atlético Madrid', 'Paris S-G', 'Juventus', 'Milan', 'Inter']

# Playing time duplicate columns (to be removed from the master file):
playing_time_duplicates = ['Min.1', 'Mn/MP', 'Min%', '90s', 'Starts.1', 'Mn/Start', 'Compl', 'Subs', 'Mn/Sub', 'unSub']

# Function that writes the output data to a CSV file:
def make_csv(dir, df, file_name):
    file_path = os.path.join(dir, f'{file_name}.csv')
    return df.to_csv(file_path)


# Function that removes unnamed columns:
def remove_unnamed_cols(df):

    # Create list of unnamed columns:
    columns = df.columns
    unnamed_cols = [col for col in columns if "Unnamed" in col]

    # Create a new df:
    new_df = (df
              .drop(columns=unnamed_cols)
              )

    return new_df


# Function that removes YOB column so that age can be standardized later:
def remove_YOB_col(df):
    return (df.drop(columns='YOB'))


# Function that removes unnecessary DW player/team ID columns:
def remove_dw_info(df):
    return (df.drop(columns=['team_id', 'player_id', 'league_id']))


# Function that removes duplicate FBref playing time columns:
def remove_playing_time_cols(df):
    return (df.drop(columns=playing_time_duplicates))


# Function that slices the master file based on an optimal playing time cutoff:
def playing_time_slice(df, cutoff : int = 8):
    return (df[df['90s_r'] >= cutoff]
            .reset_index()
            .drop(columns='index')
            )


# # Function that adds the log(market value) column to the data:
# def add_log_mkt_val_col(df):
#     df['log_mkt_val'] = np.log(df['market_value_in_eur'])
#     return df


# # Function that adds the "Big Team" dummy variable to the data:
# def add_big_team_col(df):
#     df['Big Team'] = df['team'].isin(big_teams).astype(int)
#     return df


# # Function that adds a clean league name column based on league id:
# def add_league_name_col(df):
#     df['league_name'] = df['league'].map(league_names_dict)
#     return df


def main():

    # Import the master file and make a copy:
    master_df = pd.read_csv(os.path.join(source_data_dir, 'master_file.csv'))
    master_df_copy = master_df.copy(deep=True)

    # Generate histogram/KDE plots for the playing time variables (optional):
    playing_time_cols = ['Starts', 'Min', '90s_r']
    # for col in playing_time_cols:
    #     plt.figure(figsize=(9,5))
    #     sns.histplot(data=master_df_copy, x=col, kde=True)
    #     plt.title(f'{col} KDE Plot')
    #     if col == 'Min':
    #         plt.axvline(x=720, color='r', linestyle='--')
    #         plt.show()
    #     elif col == '90s_r':
    #         plt.axvline(x=8, color='r', linestyle='--')
    #         plt.show()
    #     else:
    #         plt.axvline(x=10, color='r', linestyle='--')
    #         plt.show()

    # Test the proposed playing time cutoff measures (optional):
    playing_time_dict = {'Starts' : 10, 'Min' : 720, '90s_r' : 8} # use this to alter the playing time metric cutoffs if necessary
    # for key, val in playing_time_dict.items():

    #     # Slice the master file by the cutoff value:
    #     cut_df = master_df_copy[master_df_copy[key] >= val]
        
    #     # Compute and print the data loss (as a % of the original row total):
    #     data_loss = ((7121 - cut_df.shape[0]) / 7121)*100
    #     print(f'\nData Loss from the {key} Column Cutoff: {data_loss:.2f}%')

    ### CONCLUSION: WILL USE THE 90s_r >= 8 cutoff since it's associated with the least amount of data loss ###

    # Slice and clean the master data file:
    cleaned_master_df = remove_unnamed_cols(master_df_copy)
    cleaned_master_df = remove_YOB_col(cleaned_master_df)
    cleaned_master_df = remove_dw_info(cleaned_master_df)
    cleaned_master_df = remove_playing_time_cols(cleaned_master_df)
    cleaned_master_df = playing_time_slice(cleaned_master_df)

    # Add extra columns (optional):
    # cleaned_master_df = add_big_team_col(cleaned_master_df)
    # cleaned_master_df = add_league_name_col(cleaned_master_df)
    # cleaned_master_df = add_log_mkt_val_col(cleaned_master_df)

    # Iterate the get_dummies() process over the variables of interest:
    dummy_cols = ['team', 'season', 'nationality', 'position']

    # Initialize the list of dataframes for concatenation, starting with the clean master file:
    dfs_to_concat = [cleaned_master_df]

    for col in dummy_cols:

        # Create the dummy variable dataframe:
        dummy_df = pd.get_dummies(cleaned_master_df[col], drop_first=False)

        # Add it to the list of dataframes to concat:
        dfs_to_concat.append(dummy_df)

    # Concatenate the dataframes:
    cleaned_master_with_dummies_df = pd.concat(dfs_to_concat, axis=1)

    # Remove all NaN values from the data:
    final_master_df = cleaned_master_with_dummies_df.dropna().reset_index().drop(columns='index')

    # Send the first output file to a CSV in the analysis-data directory:
    make_csv(output_dir, final_master_df,"master_file_with_dummies")
    print(f'First file created: Row Total = {final_master_df.shape[0]}')

    # ------------ Start of the standardized DF creation for PCA process ------------

    # Initialize StandardScaler from sklearn:
    scaler = StandardScaler()

    # Print cleaned dataframe column datatypes to ensure they are all at least floats:
    # print('Index')
    # for index, column in enumerate(final_master_df.columns):
    #     print(f"{index}\t{column}: {final_master_df[column].dtype}")

    # NOTE: THE NON-QUANT VARIABLES LIKE PLAYER NAME AND LEAGUE ARE STORE AS 'OBJECT' DTYPES, AS OPPOSED TO STRINGS, IDK WHY 

    # Determine the appropriate column range for standardization: columns [0:5] don't need to be standardized
    # Columns [6:145] need to be standardized
    # Columns [146:] do not need to be standardized or included in the PCA, they are dummies

    # Data to standardize:
    data_to_scale = final_master_df.iloc[:, 6:146].values   # this is hard-coded based on the above code chunk

    # Fit and transform the selected columns:
    scaled_data = scaler.fit_transform(data_to_scale)

    # Create a DataFrame from the scaled data:
    scaled_df = pd.DataFrame(scaled_data, columns=final_master_df.columns[6:146])

    # Send standardized master DF to a CSV in the source directory:
    make_csv(output_dir, scaled_df, 'master_file_standardized')
    print(f'Second file created: Row Total = {scaled_df.shape[0]}')

    # ------------ Final Step: Position-level standardized dataframes ------------

    for position in ['DF', 'MF', 'FW']:

        # Slice the final master DF:
        position_df = final_master_df[final_master_df['position'] == position]

        # Initialize StandardScaler from sklearn:
        scaler = StandardScaler()

        # Data to standardize:
        position_data_to_scale = position_df.iloc[:, 6:146].values

        # Fit and transform the selected columns:
        position_scaled_data = scaler.fit_transform(position_data_to_scale)

        # Create a DataFrame from the scaled data:
        position_scaled_df = pd.DataFrame(position_scaled_data, columns=position_df.columns[ 6:146])

        # Send standardized position DF to a CSV in the source directory:
        make_csv(output_dir, position_scaled_df, f'{position}_standardized')
        print(f'{position} file created: Row Total = {position_scaled_data.shape[0]}')


if __name__ == '__main__':
    main()
