import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from fcmeans import FCM

df = pd.read_csv("Mall_Customers.csv")
print(df.shape)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.describe(include="all"))

#Which columns are identifiers, categorical descriptors and numerical measurements?

#Why should CustomerID normally be excluded from distance-based clustering?
#CustomerID is an identifier, not a meaningful measure of customer behaviour. and -Means groups similar customers based on distance, so including CustomerID can create misleading groups because the ID numbers do not show how similar customers are.
# What could be lost if Gender is excluded? What could be problematic if it is included?
#The analysis may miss differences in customer behaviour associated with gender. ANDThe resulting segments may be less useful for understanding demographic patterns.
#Do missing values or extreme values require attention?Missing values: The algorithm may not be able to calculate distances if data is missing. #Extreme values: Very large or small values can pull clusters away from where most customers belong.




