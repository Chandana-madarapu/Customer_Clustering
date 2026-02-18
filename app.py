import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from datetime import datetime

st.title("Customer Segmentation using K-Means Clustering")

# ==========================================
# LOAD DATASET FROM REPO
# ==========================================
df = pd.read_excel("marketing_campaign1.xlsx")

st.subheader("Dataset Preview")
st.write(df.head())

# ==========================================
# HANDLE MISSING INCOME
# ==========================================
df['Income'] = df['Income'].fillna(df['Income'].median())

# ==========================================
# AGE FEATURE
# ==========================================
current_year = datetime.now().year
df['Age'] = current_year - df['Year_Birth']
df.drop(columns=['Year_Birth'], inplace=True)

# ==========================================
# CUSTOMER TENURE
# ==========================================
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Customer_Tenure_Years'] = (datetime.now() - df['Dt_Customer']).dt.days/365
df.drop(columns=['Dt_Customer'], inplace=True)

# ==========================================
# TOTAL SPENDING
# ==========================================
spending_cols = [
    'MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds'
]
df['Total_Spending'] = df[spending_cols].sum(axis=1)
df.drop(columns=spending_cols, inplace=True)

# ==========================================
# CHILDREN COUNT
# ==========================================
df['Children_Count'] = df['Kidhome'] + df['Teenhome']
df.drop(columns=['Kidhome','Teenhome'], inplace=True)

# ==========================================
# EDA VISUALS
# ==========================================
st.subheader("Income Distribution")
fig, ax = plt.subplots()
sns.histplot(df['Income'], kde=True, ax=ax)
st.pyplot(fig)

# ==========================================
# CLUSTERING FEATURES
# ==========================================
features = ['Income','Age','Total_Spending','Children_Count','Recency']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# ELBOW METHOD
# ==========================================
st.subheader("Elbow Method")
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1,11), wcss, marker='o')
st.pyplot(fig)

# ==========================================
# KMEANS
# ==========================================
k = st.slider("Select number of clusters", 2, 10, 4)

model = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = model.fit_predict(X_scaled)

st.subheader("Cluster Distribution")
st.bar_chart(df['Cluster'].value_counts())

# ==========================================
# CLUSTER PROFILE
# ==========================================
st.subheader("Cluster Profiles")
st.write(df.groupby("Cluster")[features].mean().round(2))

# ==========================================
# VALIDATION METRICS
# ==========================================
st.subheader("Cluster Validation")
st.write("Silhouette Score:", round(silhouette_score(X_scaled, df['Cluster']),3))
st.write("Davies Bouldin Index:", round(davies_bouldin_score(X_scaled, df['Cluster']),3))
st.write("Calinski Harabasz Score:", round(calinski_harabasz_score(X_scaled, df['Cluster']),2))

# ==========================================
# PCA
# ==========================================
st.subheader("PCA Visualization")
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
ax.scatter(pca_data[:,0], pca_data[:,1], c=df['Cluster'])
st.pyplot(fig)
