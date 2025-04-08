import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

st.set_page_config(layout="wide")
st.title("📈 Qdrant UMAP Stock Pattern Explorer")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

df = load_data(ticker, start_date, end_date)
st.write(f"Loaded {len(df)} rows of data for **{ticker}**")

# Vectorize price data
window_size = 30
price_vectors, start_dates, cum_returns, volatilities = [], [], [], []

for i in range(len(df) - window_size):
    window = df["Return"].iloc[i:i + window_size].values
    normed = (window - window.mean()) / (window.std() + 1e-6)
    price_vectors.append(normed.tolist())
    start_dates.append(str(df.index[i].date()))
    cum_returns.append(np.sum(window))
    volatilities.append(np.std(window))

# Qdrant in-memory
client = QdrantClient(":memory:")
collection_name = "price_patterns"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=window_size, distance=Distance.COSINE)
)

payload = [{"start_date": d, "cum_return": r, "volatility": v} for d, r, v in zip(start_dates, cum_returns, volatilities)]
client.upload_collection(
    collection_name=collection_name,
    vectors=price_vectors,
    payload=payload,
    ids=list(range(len(price_vectors)))
)

# UMAP projection
umap_model = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
vectors_2d = umap_model.fit_transform(price_vectors)

# KMeans clustering
n_clusters = 30
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(price_vectors)

# Visualization DataFrame
viz_df = pd.DataFrame(vectors_2d, columns=["x", "y"])
viz_df["date"] = pd.to_datetime(start_dates)
viz_df["vector"] = price_vectors
viz_df["year"] = viz_df["date"].dt.year
viz_df["qdrant_id"] = list(range(len(viz_df)))
viz_df["cluster"] = cluster_labels
viz_df["cumulative_return"] = cum_returns
viz_df["volatility"] = volatilities

# Filter by year
selected_year = st.sidebar.selectbox("Filter by Year", ["All"] + sorted(viz_df["year"].unique().astype(str).tolist()))
filtered_df = viz_df if selected_year == "All" else viz_df[viz_df["year"] == int(selected_year)]

# Cluster selection
highlight_cluster = st.sidebar.selectbox("Highlight Cluster (optional)", ["None"] + list(range(n_clusters)))

# Color mode selection
color_mode = st.sidebar.radio("Color By", ["Cluster", "Cumulative Return", "Volatility"])

if color_mode == "Cluster":
    marker_color = filtered_df["cluster"]
    colorbar_title = "Cluster"
    colorscale = "Turbo"
elif color_mode == "Cumulative Return":
    marker_color = filtered_df["cumulative_return"]
    colorbar_title = "Cumulative Return"
    colorscale = "Viridis"
else:
    marker_color = filtered_df["volatility"]
    colorbar_title = "Volatility"
    colorscale = "Plasma"

# Plot
st.subheader("📊 UMAP Projection")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered_df["x"],
    y=filtered_df["y"],
    mode="markers",
    marker=dict(
        size=6,
        color=marker_color,
        colorscale=colorscale,
        colorbar=dict(title=colorbar_title),
    ),
    text=filtered_df.apply(lambda row: f"Date: {row['date'].date()}<br>Cluster: {row['cluster']}<br>CumReturn: {row['cumulative_return']:.2f}<br>Volatility: {row['volatility']:.4f}", axis=1),
    hoverinfo="text",
    name="All Patterns"
))

# Highlight cluster if selected
if highlight_cluster != "None":
    cluster_points = filtered_df[filtered_df["cluster"] == int(highlight_cluster)]
    fig.add_trace(go.Scatter(
        x=cluster_points["x"],
        y=cluster_points["y"],
        mode="markers",
        marker=dict(symbol="circle", size=10, color="black", opacity=0.5),
        name=f"Cluster {highlight_cluster}"
    ))

fig.update_layout(
    xaxis_title="UMAP Dimension 1",
    yaxis_title="UMAP Dimension 2"
)

# Pattern selection
selected_idx = st.number_input(
    "Click a pattern index to search similar (0 to {})".format(len(filtered_df)-1),
    min_value=0,
    max_value=len(filtered_df)-1,
    step=1
)

# Run similarity search on user input
if st.button("🔍 Search Similar Patterns"):
    query_vector = filtered_df.iloc[selected_idx]["vector"]
    query_date = filtered_df.iloc[selected_idx]["date"]

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5
    )

    match_ids = [res.id for res in results]
    matches = filtered_df[filtered_df["qdrant_id"].isin(match_ids)]

    # Plot matches and query
    fig.add_trace(go.Scatter(
        x=[filtered_df.iloc[selected_idx]["x"]],
        y=[filtered_df.iloc[selected_idx]["y"]],
        mode="markers",
        marker=dict(symbol="star", size=14, color="black"),
        name="Query"
    ))

    fig.add_trace(go.Scatter(
        x=matches["x"],
        y=matches["y"],
        mode="markers",
        marker=dict(symbol="diamond", size=10, color="red"),
        name="Top Matches"
    ))

    # Line chart for similarity
    st.subheader("📈 Pattern Similarity Plot")
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.plot(query_vector, label=f"Query: {query_date}", color="black", linewidth=2)

    for i, res in enumerate(results):
        ax.plot(res.vector, label=f"Match {i+1}: {res.payload['start_date']} (Score: {res.score:.2f})")

    ax.set_xlabel("Days")
    ax.set_ylabel("Normalized Return")
    ax.set_title("Price Pattern Similarity")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig2)

st.plotly_chart(fig, use_container_width=True)
