import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def get_data():
    try:

        df=pd.read_csv('Info.txt',delim_whitespace=True)
        print("Creating dataframe")
        print("\n")

    except:

        print("Errorr :The requested file wasn't found")

    data=list((df["REFNUM"].values))
    df["SEVERITY"].replace({"B": 0, "M":1, np.nan:0}, inplace=True)
    labels=list((df["SEVERITY"].values))

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test
