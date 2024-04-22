'''
Web scraping FBref using the soccerdata API
Last Updated: 4/22/2024
'''

# Import the soccerdata module
import soccerdata as sd
import pandas as pd
import os

# Directory variables for data export:
repo_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
export_dir = os.path.join(repo_dir, "fbref-data")

# List of league IDs, seasons, and statistic types for bulk scraping:
league_ids = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'GER-Bundesliga', 'ITA-Serie A']
stat_types = ['standard', 'shooting', 'passing', 'passing_types', 'goal_shot_creation', 'defense', 'possession', 'playing_time']
season_ids = ['1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223']


# Generate a pandas dataframe using a method:
def get_player_stats(league: str, season : str, stat_type: str = 'passing'):

    # Create an instance of the FBref class:
    sd_inst = sd.FBref(leagues=league, seasons=season)

    # Return a Pandas dataframe:
    return sd_inst.read_player_season_stats(stat_type=stat_type)


# Function that creates a CSV from a pandas df:
def make_csv(df, dir, file_name):
    file_path = os.path.join(dir, f'{file_name}.csv')
    return df.to_csv(file_path, index=True)


# Try making two statistics dataframes and joining on the player name column:
test_df1 = get_player_stats('ENG-Premier League', '1819', 'passing')
test_df2 = get_player_stats('ENG-Premier League', '1819', 'shooting')

def join_dfs(df1, df2):
    merged_df = pd.merge(df1, df2, on='player')
    return merged_df


def main():
    
    # Iterate over the lists and create a series of dataframes with output data:
    # WARNING: Attempting to iterate over all leagues and seasons will likely not worl. Break it up if needed:
    for league in league_ids:
        for season in season_ids:

            # Create a list of statistic dataframes:
            df_list = [get_player_stats(league=league, season=season, stat_type=stat) for stat in stat_types]

            # Sort the multi index of each dataframe for more efficient merging later on:
            for df in df_list:
                df = df.sort_index(level=['league', 'season', 'team', 'player'], inplace=True)

            # Define initial dataframe:
            joined_df = df_list[0]

            # Join each subsequent dataframe from the list to the previous one:
            for df, stat in zip(df_list[1:], stat_types[1:]):

                # Overwrite the initialized dataframe by joining the current dataframe to past joins:
                joined_df = joined_df.join(df, how='outer', lsuffix=f'_r', rsuffix=f'_{stat}')

            # Send final season-level output to Excel:
            make_csv(joined_df, os.path.join(export_dir, league), file_name=f'{league}_{season}_full_join') # send to the league-specific folder within the fbref-data directory; was doing this manually in File Explorer prior to 4/22/24

    print('Success.')


if "__name__" == "__main__":
    main()
