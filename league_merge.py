'''
Combining FBref data tables pulled using the soccerdata API
Last Updated: 2/12/2024
'''

# Import libraries:
import pandas as pd
import os

# Global variables:
dir = 'C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model'
eng_dir = 'C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model/ENG-Premier League'
export_dir = 'C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model/Merged Data'
season_ids = ['1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223']
league_ids = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'GER-Bundesliga', 'ITA-Serie A']
test_path = "C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model/ENG-Premier League/ENG-Premier League_1920_full_join.xlsx"
index_col_names = ['league', 'season', 'team']
SKIP_ROWS = [1] 

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
    
    # Clean index column names:
    for name in index_col_names:
        df[name] = df[name].fillna(method='ffill')
    
    return df


# Variables for data cleaning:
new_cols_dict = {'Unnamed: 0': 'league', 'Unnamed: 1': 'season', 'Unnamed: 2': 'team', 'Unnamed: 3': 'player', 
            'Unnamed: 4': 'nationality', 'Unnamed: 5': 'position', 'Unnamed: 6': 'age', 'Unnamed: 7': 'YOB'}

# Initliaze an empty list:
df_list = []

# Data cleansing iteration at the league-season level:
for league in league_ids:
    for id in season_ids:

        # Define file path and read into pandas:
        file_path = f"C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model/{league}/{league}_{id}_full_join.xlsx"
        df = import_and_clean(file_path)

