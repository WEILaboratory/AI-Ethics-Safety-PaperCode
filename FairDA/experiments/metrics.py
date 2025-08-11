import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

############# Importing the datasets
train = pd.read_csv("dataset/hr/train.csv")
test = pd.read_csv("dataset/hr/test.csv")

#### Visualizing the null values using missingo function
import missingno as msno

#### Visualizing the null values using missingo function
msno.matrix(train)
msno.bar(test, color="y", figsize=(10, 8))

plt.show()

print("end")
