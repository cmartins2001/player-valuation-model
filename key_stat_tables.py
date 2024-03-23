'''
Python script for creating league-level statistics tables.
Last Updated: 3/23/2024
'''

# Import libraries:
import pandas as pd
import os
import seaborn as so
import matplotlib.pyplot as plt
# Graph output styling from matplotlib:
plt.style.use('fivethirtyeight')

# Global variables:
repo_dir = os.getcwd()  # Directory of the script
merged_data_dir = os.path.join(repo_dir, "Merged Data")   # Path to the Merged Data folder
export_dir = os.path.join(repo_dir, "Summary Tables")     # Path to the Summary Data folder
league_ids = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'GER-Bundesliga', 'ITA-Serie A'] # For league-level iteration

# Create a dictionary of key statistics and their respective aggregation methods:
test_stat_dict = {'age': 'mean',
                  'Gls.1': 'mean',  # goals/90
                  'Cmp%.1': 'mean', # medium pass completion rate
                  'SoT%': 'mean',   # shot-on-target %
                  'SCA90' : 'mean', # shot-creating actions per 90 
                  'Tkl%': 'mean',   # successful tackle %
                  'Succ%': 'mean'   # successful take-on %
                  }


# Function that imports data from Github data folder:
def import_merged_data(file_path):
    df = pd.read_excel(file_path)
    return df


# Function that removes unnamed columns:
def remove_unnamed_cols(df):

    # Create list of unnamed columns:
    columns = df.columns
    unnamed_cols = [col for col in columns if "Unnamed" in col]

    # Create a new df and set index:
    new_df = (df
              .drop(columns=unnamed_cols)
              .set_index('league')
              )

    return new_df


# Function that writes the output data to an Excel file:
def make_xl(path, df, file_name):
    file_path = os.path.join(path, f'{file_name}.xlsx')
    return df.to_excel(file_path, index=True)           # Remove index=True if getting permission error


# Function that generates a dataframe with team-level statistics filtered by position and season:
def key_stats_table(df, stat_dict, pos, season):
    return (df[df['position'].str.contains(pos) & (df['season'] == season)]
              .groupby('team')
              .aggregate(stat_dict)
              .reset_index()
              )


# Position and season variables for filtering:
filter_pos = 'MF'
filter_season = 2223

# Make a list of imported dataframes:
league_df_list = [import_merged_data(os.path.join(merged_data_dir, f"{league}_full_merge.xlsx")) for league in league_ids]

# Make a list of clean, copied dataframes:
cleaned_league_df_list = [remove_unnamed_cols((league_df.copy(deep=True))).dropna(subset=['position']) for league_df in league_df_list]

# Create a midfielder summary statistics table for each league in the 22/23 season:
mid_2223_stat_df_list = [key_stats_table(clean_league_df, test_stat_dict, filter_pos, filter_season) for clean_league_df in cleaned_league_df_list]

# Send the summary tables to Excel:
for table, league in zip(mid_2223_stat_df_list, league_ids):
    make_xl(export_dir, table, file_name=f"{league}_{filter_pos}_{filter_season}_keystats")
