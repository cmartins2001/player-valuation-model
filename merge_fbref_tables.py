'''
Combining FBref data tables pulled using the soccerdata API
Last Updated: 4/22/2024
'''

# Import libraries:
import pandas as pd
import os

# Global variables:
repo_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
fbref_data_dir = os.path.join(repo_dir, "fbref-data")   # Path to the fbref-data folder
export_dir = os.path.join(repo_dir, "Merged Data")      # Path to the export folder
season_ids = ['1718', '1819', '1920', '2021', '2122', '2223']
league_ids = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'GER-Bundesliga', 'ITA-Serie A']
index_col_names = ['league', 'season', 'team']
SKIP_ROWS = [1] 
new_cols_dict = {'Unnamed: 0': 'league', 'Unnamed: 1': 'season', 'Unnamed: 2': 'team', 'Unnamed: 3': 'player', 
            'Unnamed: 4': 'nationality', 'Unnamed: 5': 'position', 'Unnamed: 6': 'age', 'Unnamed: 7': 'YOB'}

# Global functions:


# Function that creates a CSV from a pandas df:
def make_csv(df, dir, file_name):
    file_path = os.path.join(dir, f'{file_name}.csv')
    return df.to_csv(file_path, index=True)


# Function for importing and cleaning data at the league-season level:
def import_and_clean(path):

    # Create the dataframe:
    df = (pd
        .read_excel(path, header=SKIP_ROWS)
        .rename(columns=new_cols_dict)
        .drop([0])
        )
    
    # Clean index column names and set index to league:
    for name in index_col_names:
        df[name] = df[name].fillna(method='ffill')
    return df


# Function for merging two pandas dataframes:
def concatenate_dfs(df_list):
    merged_df = pd.concat(df_list)
    return merged_df


def main():

    # Import, clean and concatenate data:
    for league in league_ids:

        # Import the league-season-specific dataframes into a list: 
        league_df_list = [import_and_clean(path=os.path.join(fbref_data_dir, league, f"{league}_{season}_full_join.xlsx")).set_index('league') for season in season_ids]

        # Concatenate the dataframes from the list:
        league_merged_df = concatenate_dfs(league_df_list)
        print(f'\nThe {league} merge is complete, proceeed to dimensions check.\n')

        # Check row and column counts:
        print(f'\n{league} Merge Row Count: {league_merged_df.shape[0]}\n')
        print(f'\n{league} Merge Column Count: {league_merged_df.shape[1]}\n')

        # Send the merged dataframe to excel:
        user_bool = int(input("Enter 1 if DF dimensions OK, 0 otherwise: "))
        if user_bool == 1:
            print(f'\nThe {league} merge was successful. Sending to Excel...\n')
            make_csv(league_merged_df, export_dir, file_name=f'{league}_full_merge')
        else:
            print(f'\nThe {league} merge was unsuccessful. Proceeding to next league...\n')

    print(f'\nProcess complete. See {export_dir} for results.\n')


if __name__ == "__main__":
    main()
