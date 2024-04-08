'''
Script that merges transferMRKT data to existing FBref data
Last Updated: 4/6/2024
'''

# Libraries:
import pandas as pd
import os

# GLOBAL VARIABLES:
repo_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
transferMKT_dir = os.path.join(repo_dir, 'transferMKT-data') # For storing the TMKT files from Data World
export_dir = os.path.join(repo_dir, 'fbref-dw-merges') # For final export files
dw_top5_leagues = [ 'GB1', 'ES1', 'FR1', 'GR1', 'IT1'] # For top-5 league filtering
fbref_top5_leagues = ['ENG-Premier League', 'ESP-La Liga', 'FRA-Ligue 1', 'GER-Bundesliga', 'ITA-Serie A'] # For iteration zipping

# DW column dictionaries for renaming:
appearances_col_dict = {'competition_id' : 'league_id', 'player_club_id' : 'team_id'}
player_val_col_dict = {'current_club_id' : 'team_id', 'player_club_domestic_competition_id' : 'league_id'}
clubs_col_dict = {'club_id' : 'team_id','name' : 'team', 'domestic_competition_id' : 'league_id'}
# List of these dictionaries:
col_rename_dicts = [appearances_col_dict, player_val_col_dict, clubs_col_dict]

# DW column lists for slicing:
appearances_cols = ['player_id', 'team_id', 'date', 'player_name', 'league_id']
player_val_cols = ['player_id', 'date', 'team_id', 'market_value_in_eur', 'league_id']
clubs_cols = ['team_id', 'team', 'league_id']
# List of these lists:
dw_cols_lists = [appearances_cols, player_val_cols, clubs_cols]

# DW dictionaries for groupby.agg() call:
appearances_agg_dict = {'player_id' : 'first', 'league_id' : 'first'}
player_val_agg_dict = {'market_value_in_eur' : 'mean', 'league_id' : 'first'}
clubs_agg_dict = {} # empty because no agg needed. need list length=3 for iteration zipping
# List of these dictionaries:
aggregation_dicts =[appearances_agg_dict, player_val_agg_dict, clubs_agg_dict]

# GLOBAL FUNCTIONS:


# Function that creates a CSV from a pandas df:
def make_csv(df, dir, file_name):
    file_path = os.path.join(dir, f'{file_name}.csv')
    return df.to_csv(file_path, index=True)


# Function that converts a datetime column to soccer season format (ex: 1819):
def calculate_season(date):
    year = date.year
    month = date.month
    if month in range(7,12):
        return ((year - 2000) * 100) + (year - 1999)
    else:
        return ((year - 2001) * 100) + (year - 2000)
    

# Function that standardizes column names for DW dataframes:
def standardize_col_names(df, col_dict):

    # Check if there is a date column in the DF:
    if 'date' in df.columns:
        return (df
                .rename(columns=col_dict)
                .sort_values('date', ascending=True)
                )
    else:
        return (df.
                rename(columns=col_dict)
                )


# Function that filters DW dataframes by date/league:
def filter_data_world(df, league_id):

    # Determine if the passed DF has a 'date' column:
    if 'date' in df.columns:
    
        # Filter by date and league id:
        return (df[(df['league_id'] == league_id) & (df['date'] >='2017-07-01')])
    
    else:

        # Filter by league id only:
        return (df[df['league_id'] == league_id])


# Function that slices DW dataframes by a list of columns:
def slice_data_world(df, cols):
    return (df[cols])


# Function that adds a 'season' column to the DW dataframes:
def add_season_column(df):
    
    # Convert the current 'date' column to date format:
    df['date'] = pd.to_datetime(df['date'])

    # Add the season column by calling the calculate_season() function:
    df['season'] = df['date'].apply(lambda x: calculate_season(x))

    # Drop the date column for testing:
    df.drop('date', axis=1, inplace=True)

    return df


# Function that aggregates DW dataframes to the season-team-player-level:
def aggregate_data_world(df, agg_dict, groupby_cols):
    return (df
            .groupby(groupby_cols)
            .agg(agg_dict)
            .reset_index()
            )


# Function that merges the three aggregated DW dataframes:
def merge_dw_dfs(df1, df2, df3):
    '''
    Use this function to merge the three aggregate DW dataframes in this order:
    df1 = appearances, df2 = player valuations, and df3 = clubs
    '''

    # Merge appearances and player valuations:
    first_merge = pd.merge(df1, df2, on=['league_id', 'season', 'team_id', 'player_id'])

    # Merge the first merge and the clubs DF:
    second_merge = pd.merge(first_merge, df3, on=['team_id', 'league_id'])

    # Return the 17/18+ season data only:
    return (second_merge[second_merge['season'] >= 1718])


# Function that imports and standardizes fbref-data:
def import_and_standardize_FBref(file_name):
    
    # Import to pandas from the merged data directory:
    fbref_df = pd.read_excel(os.path.join(repo_dir, f"Merged Data\{file_name}.xlsx"))

    # Rename the player column to 'player_name' for merging:
    return (fbref_df.rename(columns={'player' : 'player_name'}))


# Function that merges the FBref and DW data for a single league:
def merge_fbref_dw(fbref_df, dw_df):
    return (pd.merge(fbref_df, dw_df, on=['season', 'player_name']))


def main():

    # Start by importing the data from data world:
  
    # Create a pandas df for the DW appearances file:
    appearances_df = pd.read_csv('https://query.data.world/s/2t4a5mgcrt7xb32ifpci2wijahs7fq?dws=00000')

    # Create a pandas dataframe for the DW player valuation file:
    player_val_df = pd.read_csv('https://query.data.world/s/bxh6i5g3kll34aqabzszjbecgdzabm?dws=00000')

    # Create a pandas for the clubs file:
    clubs_df = pd.read_csv('https://query.data.world/s/4iac2yo5mskcbmy6xnsvahtxe5eakd?dws=00000')

    # Send the three raw files to the TMKT directory as CSVs:
    data_world_dfs = [appearances_df, player_val_df, clubs_df]
    dw_file_names = ['DW_appearances', 'DW_player_vals', 'DW_clubs']

    for df, name in zip(data_world_dfs, dw_file_names):
        make_csv(df, transferMKT_dir, name)

    # Next, clean the raw DW data:
    dw_raw_dfs = [appearances_df, player_val_df, clubs_df]

    # Standardize col names:
    standardized_dfs = [standardize_col_names(df, col_dict) for df, col_dict in zip(dw_raw_dfs, col_rename_dicts)]

    # Filter by league and date:
    for dw_league, fbref_league in zip(dw_top5_leagues, fbref_top5_leagues):

        # Initialize empty list of dataframes:
        league_dw_df_list = []

        # Conduct cleaning and add the 3 DFs to the empty list:
        for index, (df, cols_list, agg_dict) in enumerate(zip(standardized_dfs, dw_cols_lists, aggregation_dicts)):
            
            # Filter the standardized DF:
            filtered_df = filter_data_world(df, dw_league)

            # Slice the filtered DF by necessary columns only:
            sliced_df = slice_data_world(filtered_df, cols_list)

            # Aggregate the sliced DF -- first 2 DW DFs only:
            if index == 0:
                
                # Add the season column to the sliced DF:
                sliced_df = add_season_column(sliced_df)

                # Aggregate the dataframe:
                agg_df = aggregate_data_world(df=sliced_df, agg_dict=agg_dict, groupby_cols=['season', 'team_id', 'player_name'])
                league_dw_df_list.append(agg_df)
            elif index == 1:
                
                # Add the season column to the sliced DF:
                sliced_df = add_season_column(sliced_df)

                # Aggregate the dataframe:
                agg_df = aggregate_data_world(df=sliced_df, agg_dict=agg_dict, groupby_cols=['season', 'team_id', 'player_id',])
                league_dw_df_list.append(agg_df)
            else:
                league_dw_df_list.append(sliced_df)

        # Merge the 3 cleaned, league-level DW DFs:
        dw_league_merge = merge_dw_dfs(league_dw_df_list[0], league_dw_df_list[1], league_dw_df_list[2])

        # Import the corresponding fbref league data:
        fbref_league_df = import_and_standardize_FBref(file_name=f"{fbref_league}_full_merge")

        # Merge the DW merge to the fbref DF:
        fbref_dw_league_merge = merge_fbref_dw(fbref_league_df, dw_league_merge)

        # Send the final league merge to the export directory:
        make_csv(fbref_dw_league_merge, export_dir, file_name=f"{fbref_league}_fbref_dw_merge")


if __name__ == '__main__':
    main()
