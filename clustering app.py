
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["OMP_NUM_THREADS"] = "1"
df = pd.read_csv('CustomerMarketing.csv')
df.info()
df.head(5)
df.isnull().sum()
#The income feature has around 24 null values.
df['Income'].isna().sum()
median_income = df['Income'].median()
df['Income'] = df['Income'].fillna(median_income)
df['Income'].isna().sum()
                                    
from datetime import datetime
current_year = datetime.now().year
df['Age'] = current_year - df['Year_Birth']
df[['Year_Birth','Age']].head(5)
df.drop(columns=['Year_Birth'],inplace=True)
summary = df.describe().T
summary['mode'] = df.mode().iloc[0]
summary
sns.histplot(df['Income'],kde=True)
plt.title('Income Distribution')
plt.show()
sns.boxplot(y=df['Income'])
plt.title('Income Distribution')
plt.show()

df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'],format='%d-%m-%Y')
today = datetime.now()
df['Customer_Tenure_Years'] = (today - df['Dt_Customer']).dt.days/365
df['Customer_Tenure_Years'] = df['Customer_Tenure_Years'].round(1)
df['Customer_Tenure_Years'].head()
df.drop(columns=['Dt_Customer'],inplace=True)
sns.histplot(df['Recency'], kde=True)
plt.title("Recency Distribution")
plt.xlabel("Days Since Last Purchase")
plt.ylabel("Count")
plt.show()
sns.boxplot(y=df['Recency'])
plt.title("Boxplot of Recency")
plt.show()


spending_cols = [
    'MntWines',
    'MntFruits',
    'MntMeatProducts',
    'MntFishProducts',
    'MntSweetProducts',
    'MntGoldProds'
]
df['Total_Spending'] = df[spending_cols].sum(axis=1)
df['Total_Spending'].head(5)
df['Total_Spending'].describe()
sns.violinplot(y=df['Total_Spending'])
plt.title("Violin Plot of Total Spending")
plt.show()
sns.boxplot(y=df['Total_Spending'])
plt.title("Boxplot of Total Spending")
plt.show()


df.drop(columns=spending_cols, inplace=True)
#DEALS FEATURE
channel_cols = ['NumWebPurchases',
    'NumCatalogPurchases',
    'NumStorePurchases',
    'NumWebVisitsMonth']

for col in channel_cols:
    sns.histplot(df[col],kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

    sns.boxplot(y=df[col])
    plt.title(f"Distribution of {col}")
    plt.show()


df['Total_Purchases'] = (
    df['NumWebPurchases'] +
    df['NumCatalogPurchases'] +
    df['NumStorePurchases']
)

sns.histplot(df['Total_Purchases'], kde=True)
plt.title("Total Purchases Distribution")
plt.show()

sns.histplot(df['NumDealsPurchases'], kde=True)
plt.title("Distribution of Deal-Based Purchases")
plt.show()

sns.boxplot(y=df['NumDealsPurchases'])
plt.title("Boxplot of Deal-Based Purchases")
plt.show()


df['Deal_purchase_ratio'] = df['NumDealsPurchases'] / (df['Total_Purchases'] + 1)
df['Deal_purchase_ratio'].head()
#The Deal Purchase Ratio tells us what percentage of purchases depend on deals.
df.drop(columns=[
    'NumWebPurchases',
    'NumCatalogPurchases',
    'NumStorePurchases'
], inplace=True)
#CAMPAIGN FEATURES
campaign_cols = ['AcceptedCmp1',
    'AcceptedCmp2',
    'AcceptedCmp3',
    'AcceptedCmp4',
    'AcceptedCmp5',
    'Response']
df['Campaign_Response_Count'] = df[campaign_cols].sum(axis=1)
df[['Campaign_Response_Count']].head()
df['Campaign_Response_Count'].value_counts()
sns.histplot(df['Campaign_Response_Count'], discrete=True)
plt.title("Campaign Response Count Distribution")
plt.show()
#The Campaign_Response_Count feature ranges from 0 to 5, indicating varying levels of customer responsiveness to marketing campaigns, 
#with the majority of customers showing low to moderate engagement.
df.drop(columns=campaign_cols, inplace=True)
df['Complain'].value_counts()
#The complain feature indicates a few people shows dissatisfaction. Hence keeping the feature as is.
df['Z_CostContact'].value_counts()
df['Z_Revenue'].value_counts()
df.drop(columns=['Z_CostContact','Z_Revenue'], inplace=True)
#Z_CostContact and Z_Revenue were removed since they contained constant values across all observations and provided no discriminatory power for clustering.
#CATEGORICAL FEATURES
df['Education'].value_counts()
#Ordinal Encoding because education follows a natural progression
edu_map = {
    'Basic': 1,
    '2n Cycle': 2,
    'Graduation': 3,
    'Master': 4,
    'PhD': 5
}
df['Education_Level'] = df['Education'].map(edu_map)
df[['Education', 'Education_Level']].head()
df['Education_Level'].value_counts()
df.drop(columns=['Education'], inplace=True)
sns.countplot(x='Education_Level', data=df)
plt.title("Education Level Distribution")
plt.show()
sns.violinplot(x='Education_Level', y='Total_Spending', data=df)
plt.title("Total Spending by Education Level")
plt.show()
#Customers with higher education levels tend to exhibit higher median total spending. In particular, customers with a Master’s degree show 
#the highest spending tendency among all education groups.
df['Marital_Status'].value_counts()
#Merging odd lables into sensible groups to reduce the noise
df['Marital_Status'] = df['Marital_Status'].replace({
    'Alone': 'Single',
    'YOLO': 'Single',
    'Absurd': 'Single'
})
sns.countplot(x='Marital_Status', data=df)
plt.title("Marital Status Distribution")
plt.show()
df = pd.get_dummies(df, columns=['Marital_Status'], drop_first=True)
df['Children_Count'] = df['Kidhome'] + df['Teenhome']
sns.histplot(df['Children_Count'], discrete=True)
plt.title("Total Children in Household")
plt.show()
sns.violinplot(x='Children_Count', y='Total_Spending', data=df)
plt.title("Total Spending by Children Count")
plt.show()
df.drop(columns=['Kidhome','Teenhome'], inplace=True)
#Violin plot analysis shows that customers without children exhibit the highest median total spending, while spending gradually decreases as 
#the number of children in the household increases, indicating greater discretionary purchasing power among child-free customers.
#APPLYING STANDARD SCALING TECHNIQUE 
selected_features = [
    "Income",
    "Age",
    "Total_Spending",
    "Children_Count",
    "Recency"
]

X = df[selected_features]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X.head()
print(df.columns)
X_scaled[:5]
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
#The correlation heatmap indicates a strong positive relationship between Total_Purchases and Total_Spending, and a moderate positive relationship 
#between Income and Total_Spending. Children_Count shows a negative correlation with Total_Spending, indicating lower discretionary spending in larger 
#households. Overall, most features exhibit low inter-correlation, suggesting minimal redundancy and suitability for clustering.
pd.DataFrame(X_scaled).describe()
from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
#The WCSS curve shows a sharp decrease up to K=4, after which the rate of decrease slows down, indicating diminishing returns. Hence, K=4 was selected as the optimal number of clusters.
n_clusters = 4
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters
df['Cluster'].value_counts()
cluster_profile = df.groupby('Cluster').mean()
cluster_profile
cluster_profile.round(2)
#KMeans clustering identified four distinct customer segments: (1) Premium High-Value Customers with high income, high spending, and strong campaign 
#responsiveness; (2) Value-Conscious Loyalists who purchase frequently but are price-sensitive; (3) Budget Occasional Buyers with moderate engagement 
#and spending; and (4) Dormant Customers characterized by low purchase frequency and low spending.
from sklearn.metrics import silhouette_score
sil = silhouette_score(X_scaled, df['Cluster'])
sil
from sklearn.metrics import davies_bouldin_score

dbi = davies_bouldin_score(X_scaled, df['Cluster'])
dbi
from sklearn.metrics import calinski_harabasz_score

chi = calinski_harabasz_score(X_scaled, df['Cluster'])
chi
#KMeans clustering was evaluated using internal validation metrics. The silhouette score of approximately 0.21 indicates overlapping yet meaningful 
#clusters, which is expected in real-world customer behavior datasets. The Davies–Bouldin Index of 1.36 suggests moderate cluster separation, 
#while a Calinski–Harabasz score of 644 indicates reasonable cluster compactness and separation. Based on these metrics and strong business 
#interpretability of cluster profiles, K=4 was selected as the final clustering solution.
profile_cols = [
    'Income',
    'Total_Spending',   
    'Age',
    'Children_Count',   
    'Recency'
]
cluster_summary = (
    df.groupby('Cluster')[profile_cols]
      .mean()
      .reset_index()
)
cluster_summary['Income'] = cluster_summary['Income'].round(0)
cluster_summary['Total_Spending'] = cluster_summary['Total_Spending'].round(0)
cluster_summary['Age'] = cluster_summary['Age'].round(0)
cluster_summary['Children_Count'] = cluster_summary['Children_Count'].round(1)
cluster_summary['Recency'] = cluster_summary['Recency'].round(0)
print("Cluster Profiles (Average Values per Group):")
display(cluster_summary)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
plot_df["Cluster"] = kmeans.labels_

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set1",
    alpha=0.7
)

plt.title("Customer Segmentation - Cluster Separation (PCA View)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()
Cluster 3 — Premium High-Value Customers

Profile
Highest Income (76K)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("Customer Segmentation using K-Means Clustering")


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Marketing Dataset", type=["xlsx", "csv"])


if uploaded_file is not None:

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())


    # -----------------------------
    # DATA CLEANING
    # -----------------------------
    df = df.dropna()

    # Remove non-useful columns if present
    drop_cols = ["ID", "Dt_Customer"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])


    # -----------------------------
    # SELECT NUMERIC FEATURES
    # -----------------------------
    numeric_df = df.select_dtypes(include=np.number)


    # Remove constant columns (no clustering value)
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]


    st.subheader("Numeric Features Used for Clustering")
    st.write(numeric_df.columns.tolist())


    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)


    # -----------------------------
    # SELECT NUMBER OF CLUSTERS
    # -----------------------------
    k = st.slider("Select Number of Clusters", 2, 10, 3)


    # -----------------------------
    # APPLY KMEANS
    # -----------------------------
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(scaled_data)

    df["Cluster"] = clusters


    # -----------------------------
    # SHOW CLUSTERED DATA
    # -----------------------------
    st.subheader("Clustered Dataset")
    st.write(df.head())


    # -----------------------------
    # CLUSTER COUNT
    # -----------------------------
    st.subheader("Cluster Distribution")
    st.bar_chart(df["Cluster"].value_counts())


    # -----------------------------
    # PCA VISUALIZATION
    # -----------------------------
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    plt.figure()
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Customer Segments (PCA View)")

    st.pyplot(plt)
