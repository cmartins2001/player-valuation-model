'''
Importing transferMKT data to pandas from data world
Last Updated: 4/2/2024
'''

import pandas as pd
import os

repo_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script


# Function that creates a CSV from a pandas df:
def make_csv(df, dir, file_name):
    file_path = os.path.join(dir, f'{file_name}.csv')
    return df.to_csv(file_path, index=True)


# Create a pandas dataframe:
df = pd.read_csv('https://query.data.world/s/bxh6i5g3kll34aqabzszjbecgdzabm?dws=00000')

# Send the dataframe to the Github repo as a CSV:
make_csv(df, repo_dir, file_name="transferMKT_player_valuation")
