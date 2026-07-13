import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide"
)

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------

sales = pd.read_csv("train.csv")
anomalies = pd.read_csv(
    "weekly_anomaly_detection.csv",
    header=None
)

anomalies = anomalies.iloc[:,0].str.split(",", expand=True)

anomalies.columns = [
    "Order Date",
    "Sales",
    "Isolation",
    "RollingMean",
    "RollingStd",
    "ZScore",
    "Rolling_Z"
]

# Remove the duplicated header row
anomalies = anomalies.iloc[1:].reset_index(drop=True)

# Convert data types
anomalies["Order Date"] = pd.to_datetime(anomalies["Order Date"])
import csv

clusters = pd.read_csv(
    "product_clusters.csv",
    header=None
)

# Parse the CSV line correctly, respecting quoted commas
rows = clusters.iloc[:, 0].apply(lambda x: next(csv.reader([x])))

clusters = pd.DataFrame(rows.tolist())

clusters.columns = [
    "Sub-Category",
    "Total Sales",
    "Growth Rate",
    "Volatility",
    "Average Order Value",
    "Cluster",
    "PCA1",
    "PCA2",
    "Demand Segment"
]

# Remove duplicated header row
clusters = clusters.iloc[1:].reset_index(drop=True)

# Convert numeric columns
numeric_cols = [
    "Total Sales",
    "Growth Rate",
    "Volatility",
    "Average Order Value",
    "Cluster",
    "PCA1",
    "PCA2"
]

for col in numeric_cols:
    clusters[col] = pd.to_numeric(clusters[col])
# Load forecast results

category_forecast = pd.read_csv(
    "category_forecast.csv",
    header=None
)

category_forecast = category_forecast.iloc[:,0].str.split(
    ",",
    expand=True
)

category_forecast.columns = [
    "Month",
    "Furniture",
    "Technology",
    "Office Supplies"
]

category_forecast = category_forecast.iloc[1:]


region_forecast = pd.read_csv(
    "region_forecast.csv",
    header=None
)

region_forecast = region_forecast.iloc[:,0].str.split(
    ",",
    expand=True
)

region_forecast.columns = [
    "Month",
    "West",
    "East"
]
anomalies["Order Date"] = pd.to_datetime(anomalies["Order Date"])

region_forecast = region_forecast.iloc[1:]
for col in ["Furniture","Technology","Office Supplies"]:
    category_forecast[col] = category_forecast[col].astype(float)

for col in ["West","East"]:
    region_forecast[col] = region_forecast[col].astype(float)

sales["Order Date"] = pd.to_datetime(
    sales["Order Date"],
    dayfirst=True)

sales["Year"] = sales["Order Date"].dt.year
sales["Month"] = sales["Order Date"].dt.month_name()
sales["Month Number"] = sales["Order Date"].dt.month

sales = sales.sort_values("Order Date")

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)
# =====================================================
# PAGE 1
# =====================================================

if page == "Sales Overview":

    st.title("📈 Sales Overview Dashboard")
    st.markdown("---")
    total_sales = sales["Sales"].sum()
    total_orders = sales["Order ID"].nunique()
    avg_order = sales["Sales"].mean()
    c1,c2,c3 = st.columns(3)
    c1.metric(
        "Total Sales",f"${total_sales:,.0f}")

    c2.metric(
        "Orders",total_orders)

    c3.metric(
        "Average Order Value",f"${avg_order:.2f}")

    st.markdown("---")
    yearly = sales.groupby("Year")["Sales"].sum()

    fig,ax = plt.subplots(figsize=(8,5))

    yearly.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Total Sales by Year")
    ax.set_ylabel("Sales")
    st.pyplot(fig)
    monthly = sales.groupby(pd.Grouper( key="Order Date", freq="ME" ))["Sales"].sum()

    fig,ax = plt.subplots(figsize=(12,5))
    monthly.plot(ax=ax)

    ax.set_title("Monthly Sales Trend")
    ax.grid(True)
    st.pyplot(fig)
    st.markdown("---")

    region = st.selectbox("Select Region",["All"] +sorted(sales["Region"].unique()))
    category = st.selectbox("Select Category",["All"] +sorted( sales["Category"].unique()))

    filtered = sales.copy()

    if region != "All":

        filtered = filtered[
            filtered["Region"] == region
        ]

    if category != "All":

        filtered = filtered[
            filtered["Category"] == category
        ]
    st.subheader("Sales by Region")

    region_sales = filtered.groupby("Region")["Sales"].sum()

    fig,ax = plt.subplots(figsize=(8,4))

    region_sales.plot(kind="bar",ax=ax)
    st.pyplot(fig)
    st.subheader("Sales by Category")

    category_sales = filtered.groupby("Category")["Sales"].sum()

    fig,ax = plt.subplots(figsize=(8,4))
    category_sales.plot(kind="bar",ax=ax)

    st.pyplot(fig)
# =====================================================
# PAGE 2
# =====================================================

elif page == "Forecast Explorer":
    st.title("📈 Forecast Explorer")
    st.markdown("### Forecast using the Best Model (XGBoost)")
    forecast_type = st.selectbox("Forecast Type",["Category","Region"])

    if forecast_type=="Category":

        option = st.selectbox(
            "Select Category",
            ["Furniture","Technology","Office Supplies" ]
        )

        horizon = st.slider(

            "Forecast Horizon (Months)",
            1,
            3,
            3
        )

        values = category_forecast[option].iloc[:horizon]
        months = category_forecast["Month"].iloc[:horizon]
    else:
        option = st.selectbox(
            "Select Region",
            ["West","East"]
        )
        horizon = st.slider(
            "Forecast Horizon (Months)",
            1,
            3,
            3
        )
        values = region_forecast[option].iloc[:horizon]
        months = region_forecast["Month"].iloc[:horizon]
    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(months,values,marker='o',linewidth=3)
    ax.set_title(f"{option} Forecast")
    ax.set_xlabel("Forecast Month")
    ax.set_ylabel("Sales")
    ax.grid(True)
    st.pyplot(fig)
    st.subheader("Forecast Values")
    forecast_table = pd.DataFrame({"Month":months,"Forecast Sales":values})
    st.dataframe(forecast_table)
    st.markdown("---")
    st.subheader("Best Model Performance")
    c1,c2 = st.columns(2)
    c1.metric("MAE","15130.97")
    c2.metric("RMSE","18016.89")
    st.success("Best Model Selected: XGBoost")
# =====================================================
# PAGE 3
# =====================================================

elif page == "Anomaly Report":

    st.title("🚨 Weekly Sales Anomaly Report")

    st.markdown(
        "Detected anomalous weeks using Isolation Forest and Rolling Z-Score."
    )
    st.subheader("Anomaly Detection Chart")

    st.image(
    "Charts/isolation_forest_anomalies.png",
    use_container_width=True
)
    st.image(
    "Charts/zscore_anomalies.png",
    use_container_width=True
)
    st.subheader("Detected Anomalies")

    cols = [
    "Order Date",
    "Sales",
    "Isolation",
    "Rolling_Z"
]

    available_cols = [c for c in cols if c in anomalies.columns]

    st.dataframe(
    anomalies[available_cols],
    use_container_width=True
)
    st.metric(
    "Total Detected Anomalies",
    len(anomalies)
)
    st.subheader("Business Interpretation")

    st.markdown("""

Possible reasons for anomalies include:

- 🎉 Festival sales (Diwali, Christmas, New Year)
- 🛍 Promotional discount campaigns
- 📦 Bulk corporate orders
- 🚚 Supply chain disruptions
- 📉 Seasonal demand fluctuations
- 📈 Inventory clearance events

Isolation Forest detects machine learning-based anomalies,
whereas Rolling Z-Score identifies statistically unusual weeks.
Using both methods provides a more robust anomaly detection strategy.

""")
# =====================================================
# PAGE 4
# =====================================================

elif page == "Product Demand Segments":

    st.title("📦 Product Demand Segmentation")

    st.markdown(
        "K-Means clustering groups product sub-categories according to their demand characteristics."
    )

    st.subheader("Demand Clusters")

    st.image(
        "charts/product_clusters.png",
        use_container_width=True
    )

    st.subheader("Sub-Category Demand Segments")

    st.dataframe(
        clusters[
            ["Sub-Category", "Demand Segment"]
        ],
        use_container_width=True
    )

    st.subheader("Products per Cluster")

    count = clusters["Demand Segment"].value_counts()

    st.bar_chart(count)

    st.subheader("Recommended Stocking Strategy")

    st.markdown("""
### 🟢 High Volume, Stable Demand
Maintain high inventory levels.

### 🔵 Growing Demand
Increase stock gradually.

### 🟠 Low Volume, High Volatility
Maintain limited inventory.

### 🔴 Declining Demand
Reduce inventory and clear stock.
""")

    st.subheader("Cluster Summary")

    summary = clusters.groupby(
        "Demand Segment"
    ).size().reset_index(name="Number of Products")

    st.dataframe(summary)