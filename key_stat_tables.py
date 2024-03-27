'''
Python script for creating league-level statistics tables.
Last Updated: 3/25/2024
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
season_ids = ['1718', '1819', '1920', '2021', '2122', '2223'] # For season-level iteration

# Create a dictionary of key statistics and their respective aggregation methods:
test_stat_dict = {'age': 'mean',
                  'Gls.1': 'mean',  # goals/90
                  'Cmp%.1': 'mean', # medium pass completion rate
                  'SoT%': 'mean',   # shot-on-target %
                  'SCA90' : 'mean', # shot-creating actions per 90 
                  'Tkl%': 'mean',   # successful tackle %
                  'Succ%': 'mean'   # successful take-on %
                  }

# Create a statistics dictionary for the DEF category:
def_stat_dict = {'Tkl': 'mean',        # total tackles
                  'TklW': 'mean',      # total tackles won
                  'Def 3rd': 'mean',   # total tackles in def. 3rd
                  'Mid 3rd': 'mean',   # total tackles in mid. 3rd
                  'Blocks.1' : 'mean', # total blocks
                  'Sh.3': 'mean',     # total shots blocked
                  }

# Create a statistics dictionary for the MID category:
mid_stat_dict = {'Gls.1': 'mean',       # goals/90
                  'Ast.1': 'mean',      # assists/90
                  'G-PK.1': 'mean',     # npg per 90
                  'G+A-PK': 'mean',   # npg+a per 90
                  'npxG+xAG.1' : 'mean',
                  'SoT%': 'mean',
                  'Cmp%': 'mean',       # overall pass completion
                  'Cmp%.1': 'mean',     # mid-range pass completion
                  'SCA90': 'mean',
                  'GCA90': 'mean',
                  'Tkl%': 'mean',       # successful tackle rate
                  'PrgC.1': 'mean',     # total progressive carries
                  }

# Create a statistics dictionary for the FW category:
fw_stat_dict = {'Gls.1': 'mean',        # goals/90
                  'Ast.1': 'mean',      # assists/90
                  'G-PK.1': 'mean',     # npg per 90
                  'G+A-PK': 'mean',   # npg+a per 90
                  'npxG+xAG.1' : 'mean', # per 90
                  'npxG.1': 'mean',     # per 90
                  'G/Sh': 'mean',       # overall pass completion
                  'G/SoT': 'mean',      # mid-range pass completion
                  'np:G-xG': 'mean',    # per 90
                  'A-xAG': 'mean',      # total A - xAG 
                  'Crs': 'mean',        # total crosses
                  'GCA90': 'mean',
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
              .round(2) # Round averages to two decimal places
              .reset_index()
              )


# Function that generates a dataframe with season-team-level statistics filtered by position:
def key_stats_table_all_seasons(df, stat_dict, pos):
    return (df[df['position'].str.contains(pos)]
              .groupby(['season', 'team'])
              .aggregate(stat_dict)
              .round(2) # Round averages to two decimal places
              .reset_index()
              )


# Make a list of imported dataframes:
league_df_list = [import_merged_data(os.path.join(merged_data_dir, f"{league}_full_merge.xlsx")) for league in league_ids]

# Make a list of clean, copied dataframes:
cleaned_league_df_list = [remove_unnamed_cols((league_df.copy(deep=True))).dropna(subset=['position']) for league_df in league_df_list]

# Create a nested for loop for each position:
positions = ['DEF', 'MF', 'FW']
seasons = [1718, 1819, 1920, 2021, 2122, 2223]
stat_dicts = [def_stat_dict, mid_stat_dict, fw_stat_dict]

for pos, stat_dict in zip(positions, stat_dicts):

    # File destination path:
    pos_export_dir = os.path.join(export_dir, pos)

    # Create a list of summary statistics table for each league for the position in the loop:
    stat_df_list = [key_stats_table_all_seasons(clean_league_df, stat_dict, pos) for clean_league_df in cleaned_league_df_list]

    # Send the summary tables to their respective directories:
    for df, league in zip(stat_df_list, league_ids):
        make_xl(pos_export_dir, df, file_name=f"{league}_{pos}_keystats")
