"""Reader reads data of different formats and return objects that are suitable to be read by others"""

import pandas as pd
import os

def read_beijing_data(folder_path):

    # https://explorebj.com/subway/

    #data_filenames = ["7.csv", "8.csv", "9.csv"]
    data_filenames = ["7.csv"]
    cols_filename = "fields.csv"

    files_paths = [folder_path + f for f in data_filenames]
    cols_path = folder_path + cols_filename

    colnames = list(pd.read_csv(cols_path))

    df = pd.concat([pd.read_csv(f, names=colnames) for f in files_paths])

    return df






