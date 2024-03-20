'''
Combining FBref data tables pulled using the soccerdata API
Last Updated: 3/20/2024
'''

# Import libraries:
import pandas as pd
import os

# Global variables:
dir = 'C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model'
eng_dir = 'C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model/ENG-Premier League'
export_dir = 'C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model/Merged Data'
season_ids = ['1718', '1819', '1920', '2021', '2122', '2223']
league_ids = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'GER-Bundesliga', 'ITA-Serie A']
index_col_names = ['league', 'season', 'team']
SKIP_ROWS = [1] 
new_cols_dict = {'Unnamed: 0': 'league', 'Unnamed: 1': 'season', 'Unnamed: 2': 'team', 'Unnamed: 3': 'player', 
            'Unnamed: 4': 'nationality', 'Unnamed: 5': 'position', 'Unnamed: 6': 'age', 'Unnamed: 7': 'YOB'}

# Global functions:


# Function that writes the output data to an Excel file:
def make_xl(path, df, file_name):
    file_path = os.path.join(path, f'{file_name}.xlsx')
    return df.to_excel(file_path, index=True)           # Remove index=True if getting permission error


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


# Import, clean and concatenate data:
for league in league_ids:

    #Initialize empty list:
    league_df_list = []

    print(f'\nWorking on the {league} dataframe merge...\n')

    for season in season_ids:
        # Dynamic file path:
        file_path = f"C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model/{league}/{league}_{season}_full_join.xlsx"

        # Import and clean the data:
        df = import_and_clean(file_path)
        df = df.set_index('league')

        # Add to list of datafrmes:
        league_df_list.append(df)
        print(f'\nSuccessfully added the {season} dataframe for the {league}.\n')

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
        make_xl(export_dir, league_merged_df, file_name=f'{league}_full_merge')
    else:
        print(f'\nThe {league} merge was unsuccessful. Proceeding to next league...\n')

print(f'\nProcess complete. See {export_dir} for results.\n')
