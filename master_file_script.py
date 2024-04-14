'''
Script for Creating a Master Data File
Last Updated: 4/13/2024
'''

# Libraries:
import pandas as pd
import os

# Global variables:
repo_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
source_data_dir = os.path.join(repo_dir, 'fbref-dw-merges')
# League IDs for iteration (top 4 only):
fbref_league_ids = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'ITA-Serie A']
dw_league_ids = [ 'GB1', 'ES1', 'FR1', 'IT1']
seasons = [1718, 1819, 1920, 2021, 2122, 2223]


# Function that creates a CSV from a pandas df:
def make_csv(df, dir, file_name):
    file_path = os.path.join(dir, f'{file_name}.csv')
    return df.to_csv(file_path, index=True)


# Import data from fbref-dw-merges directory:
def import_merged_data(league):
    return (pd.read_csv(os.path.join(source_data_dir, f"{league}_fbref_dw_merge.csv")))


# Function that drops the DW team name column and renames the FBref team name column:
def fix_team_name_columns(df):
    return (df
            .drop('team_y', axis=1)
            .rename(columns={'team_x' : 'team'})
            )


# Function that merges csv files from the fbref-dw-merges directory:
def merge_data_files(df_list):
    return(pd.concat(df_list))


def main():

    # Import the fbref-dw merged csv files:
    raw_fbref_dw_dfs = [import_merged_data(league) for league in fbref_league_ids]

    # Clean the team column names:
    clean_fbref_dw_dfs = [fix_team_name_columns(df) for df in raw_fbref_dw_dfs]

    # Concatenate into a single file:
    master_df = pd.concat(clean_fbref_dw_dfs)

    # Send a csv output to the fbref-dw merge directory:
    make_csv(master_df, source_data_dir, 'master_file')


if __name__ == '__main__':
    main()
