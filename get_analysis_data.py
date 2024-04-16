'''
Script that cleans the fbref-dw master file and generates two outputs:
(1) a cleaned master file with the covariate dummies
(2) a standardized master file with only quantitative data for PCA
Last update: 4/16/2024
'''

# Import libraries:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Graph output styling from matplotlib:
plt.style.use('fivethirtyeight')

# Global variables:

