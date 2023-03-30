import os
import pandas as pd
import numpy as np
import pycaret as pyc

data_path = os.path.join(os.path.dirname(__file__), 'data/db_game.csv')
df = pd.read_csv(data_path)

print(df)

