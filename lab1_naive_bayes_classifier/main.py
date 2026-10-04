## import necessary libraries
import pandas as pd
import numpy as np


## main function
def main(case=1):
    # load the dataset
    if case == 0:
        data = pd.read_csv("weather/weather.data", sep=r'\s+', names=["outlook", "temperature", "humidity", "windy", "play"])
    elif case == 1:
        data = pd.read_csv("breast_cancer/breast-cancer.data", names=["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"])
    else:
        raise ValueError("Invalid case number. Please choose 0 for weather data or 1 for breast cancer data.")



if __name__ == "__main__":
    main()