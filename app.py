import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

st.title("Customer Segmentation using K-Means Clustering")

# ====================================================
# LOAD DATASET (from repo)
# ====================================================
df = pd.read_excel("marketing_campaign1.xlsx")

# ====================================================
# DATA PREPROCESSING (same as training)
# ====================================================
df['Income'] = df['Income'].fillna(df['Income'].median())

current_year = datetime.now().year
df['Age'] = current_year - df['Year_Birth']
df.drop(columns=['Year_Birth'], inplace=True)

df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Customer_Tenure'] = (datetime.now() - df['Dt_Customer']).dt.days / 365
df.drop(columns=['Dt_Customer'], inplace=True)

spending_cols = [
    'MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds'
]
df['Total_Spending'] = df[spending_cols].sum(axis=1)
df.drop(columns=spending_cols, inplace=True)

df['Children_Count'] = df['Kidhome'] + df['Teenhome']
df.drop(columns=['Kidhome','Teenhome'], inplace=True)

# ====================================================
# TRAIN MODEL
# ====================================================
features = ['Income','Age','Total_Spending','Children_Count','Recency']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# ====================================================
# USER INPUT SECTION
# ====================================================
st.header("Enter customer details to predict cluster")

income = st.number_input("Income", value=50000)
age = st.number_input("Age", value=40)
spending = st.number_input("Total Spending", value=500)
children = st.number_input("Children Count", value=1)
recency = st.number_input("Recency (days since last purchase)", value=30)

# ====================================================
# PREDICTION
# ====================================================
if st.button("Predict Cluster"):

    new_data = pd.DataFrame([[income, age, spending, children, recency]],
                            columns=features)

    new_scaled = scaler.transform(new_data)
    cluster = kmeans.predict(new_scaled)[0]

    # Segment interpretation
    if cluster == 0:
        segment = "Premium High Value Customers"
        desc = "High income, high spending, low price sensitivity"

    elif cluster == 1:
        segment = "Value Conscious Loyalists"
        desc = "Frequent buyers but price sensitive"

    elif cluster == 2:
        segment = "Budget Occasional Buyers"
        desc = "Moderate engagement and spending"

    else:
        segment = "Dormant Customers"
        desc = "Low purchase frequency and low spending"

    st.success(f"Segment: {segment}")
    st.write(desc)
