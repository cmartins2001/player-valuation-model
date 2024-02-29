'''
Combining FBref data tables pulled using the soccerdata API
Last Updated: 2/12/2024
'''

# Import libraries:
import pandas as pd

# Global variables:
dir = 'C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model'
eng_dir = 'C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model/ENG-Premier League'
season_ids = ['1516', '1617', '1718', '1819', '1920', '2021', '2122', '2223']
league_ids = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'GER-Bundesliga', 'ITA-Serie A']
test_path = "C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model/ENG-Premier League/ENG-Premier League_1920_full_join.xlsx"

# Test resetting the index to see what happens to the multi-index columns:
test_df = (pd
           .read_excel(test_path)
           .reset_index()
           )

print(test_df)

# for season, league in zip(season_ids, league_ids):
#     df = pd.read_csv(f'C:/Users/cmart/OneDrive - Bentley University/Research/Player Valuation Model/ENG-Premier League _{season}')


