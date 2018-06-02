import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset = pd.read_csv("ACS_15_5YR_S1501.csv")
dataser = dataset[['GEO.display-label', 'HC02_EST_VC17', 'HC02_EST_VC18']]