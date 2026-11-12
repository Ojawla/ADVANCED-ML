#kmeans is my bseline
# iadded gender which is categorical if  is just numerical use teachers way
#fuzzy cmeans si my main
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

'''
df = pd.read_csv("Mall_Customers.csv")
df = pd.read_csv("Mall_Customers.csv")
print(df.shape)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.describe(include="all"))


#why kmeans justify

#Start with three numerical features if they exist in your file. Keep feature selection explicit; use only features you can justify
#These three features are chosen because they capture different aspects of customer behaviour and characteristics, allowing the clustering algorithm to identify meaningful customer segment
#we can see how age impact how incomeimapct how spendign scor eimpact
#added one categorical feature so cna try hot encoding
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]
X = df[features].copy()

# One-Hot Encoding for one column if is multiple look on top
X = pd.get_dummies(X, columns=["Gender"], dtype=int)

print(X.head())
#justification
#One-Hot Encoding was performed on the Gender column because it contains categorical data. This converted the categories into separate binary numerical columns (0 and 1), making the feature suitable for use by the clustering algorithm.


#scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled shape:", X_scaled.shape)
print("Approx. means:", X_scaled.mean(axis=0).round(3))
print("Approx. standard deviations:", X_scaled.std(axis=0).round(3))

#why scale
#Feature scaling was performed using StandardScaler to standardise the selected features to a common scale. This is important because K-Means uses Euclidean distance to measure similarities between customers. Without scaling, features with larger numerical ranges could disproportionately influence the clustering results

#tChoose K using evidence elbiw and shilouette
# :#this step we test different number of kk from 2 to 11 and created a elbow which will allow us pick by Helps identify where adding more clusters produces diminishing improvements in inertia.
#we look for the point where increasing the number of clusters produces much smaller reductions in inertia


#followin that we also did shilouete to ceck eh shilowtter the /...
## The Silhouette score was used alongside the Elbow method to evaluate how well-separated the clusters were.
# This helps identify a suitable number of clusters, as higher Silhouette scores generally indicate better-defined clusters.
ks = range(2, 11)
inertias, silhouettes = [], []
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(ks), inertias, marker="o")
axes[0].set(title="Elbow plot", xlabel="K", ylabel="Inertia")
axes[1].plot(list(ks), silhouettes, marker="o")
axes[1].set(title="Silhouette by K", xlabel="K", ylabel="Silhouette score")
plt.tight_layout()
plt.show()
pd.DataFrame({"K": list(ks), "inertia": inertias,
"silhouette": silhouettes}).round(3)

#kmeans clustering
k = 5 # replace with the value you can justify
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)
# Profile clusters in original feature units
print(df.groupby("kmeans_cluster")[features].mean(numeric_only=True).round(1))
print(df["kmeans_cluster"].value_counts().sort_index())
#Visualise two dimensions. This plot is a projection, so it cannot show every aspect of a higher-dimensional result.
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=features[1], y=features[2],
hue="kmeans_cluster", palette="tab10")
plt.title("K-means customer segments")
plt.tight_layout()
plt.show()
'''
#teachers way no numericcal
df = pd.read_csv("Mall_Customers.csv")
print(df.shape)
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.describe(include="all"))

#select feature and scale
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled shape:", X_scaled.shape)
print("Approx. means:", X_scaled.mean(axis=0).round(3))
print("Approx. standard deviations:", X_scaled.std(axis=0).round(3))

#choose k with vidence
#Use the elbow plot as a diagnostic, not a rule that identifies the true number of clusters. Add silhouette scores to
#provide a second view.
ks = range(2, 11)
inertias, silhouettes = [], []
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(ks), inertias, marker="o")
axes[0].set(title="Elbow plot", xlabel="K", ylabel="Inertia")
axes[1].plot(list(ks), silhouettes, marker="o")
axes[1].set(title="Silhouette by K", xlabel="K", ylabel="Silhouette score")
plt.tight_layout()
plt.show()
pd.DataFrame({"K": list(ks), "inertia": inertias,
"silhouette": silhouettes}).round(3)

#fit in the hard clustering
k = 6 # replace with the value you can justify
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)
# Profile clusters in original feature units
print(df.groupby("kmeans_cluster")[features].mean().round(1))
print(df["kmeans_cluster"].value_counts().sort_index())
#Visualise two dimensions. This plot is a projection, so it cannot show every aspect of a higher-dimensional result.
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x=features[1], y=features[2],
hue="kmeans_cluster", palette="tab10")
plt.title("K-means customer segments")
plt.tight_layout()
plt.show()

#parameter
#pick k is 6 represents the number of clusters (groups) that K-means will create. used shilouete and elbow
#random_state=42 si result is reproducible if rerun code
#n_init=10 Runs the K-means algorithm ten times using different initial starting positions for the cluster centres#
#fit: Trains the K-means model using the scaled customer data#
# we visualised using scatter plot each cluster

# graph
# cluster 0These customers earn approximately $40k–$67k and have spending scores around 40–60. They represent customers with moderate purchasing behaviour
#cluster 1 These customers have relatively moderate incomes and spending scores. They show fairly similar spending behaviour to Cluster 0, although the groups are separated by the algorithm.
#Cluster 2 – High-income, low-spending customers These customers earn approximately $70k–$140k but have relatively low spending scores. They may represent an opportunity for targeted marketing campaigns to encourage more purchases.
#cluster 3ese customers have high annual incomes and high spending scores. They are potentially valuable customers and could be targeted with premium products, loyalty programmes and exclusive offers.
#cluster 4 low income high spender These customers earn relatively less but have high spending scores. They may be enthusiastic shoppers who respond well to discounts, promotions and affordable products.

# table
#hese are relatively older customers with moderate income and average spending behaviour. Businesses could target them with products and services suited to their needs and preferences
#hese are younger customers with moderate income and average spending behaviour. Businesses could attract them through digital marketing, discounts and loyalty programmes.
#These customers earn a high income but have a low spending score. They represent a potential opportunity for businesses to increase purchases through personalised offers and targeted marketing

''' interpretation
The K-means clustering algorithm successfully grouped 200 customers into six distinct segments based on their age, annual income and spending score.
The results reveal different customer behaviours and purchasing patterns. Cluster 3 represents young, high-income and high-spending customers, making them an attractive segment for premium marketing strategies. Cluster 2 consists of high-income customers with low spending scores, presenting an opportunity to increase sales through personalised marketing. Cluster 4 represents young customers with lower incomes but high spending scores, while Cluster 5 contains customers with low income and low spending behaviour.
Clusters 0 and 1 represent customers with moderate income and spending levels but different age profiles.'''

#hierachal
from sklearn.cluster import AgglomerativeClustering
hier = AgglomerativeClustering(n_clusters=k, linkage="ward")
hier_labels = hier.fit_predict(X_scaled)
print(pd.Series(hier_labels).value_counts().sort_index())