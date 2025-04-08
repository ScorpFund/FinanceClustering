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
st.title("📈 Qdrant UMAP Stock Pattern Explorer (3D Edition)")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))
color_by = st.sidebar.selectbox("Color By", ["Cluster", "Volatility", "Cumulative Return"])
highlight_cluster = st.sidebar.selectbox("Highlight Cluster", ["None"] + [str(i) for i in range(30)])

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
price_vectors, start_dates, volatilities, cumulative_returns = [], [], [], []

for i in range(len(df) - window_size):
    window = df["Return"].iloc[i:i + window_size]
    normed = (window - window.mean()) / (window.std() + 1e-6)
    price_vectors.append(normed.tolist())
    start_dates.append(str(df.index[i].date()))
    volatilities.append(window.std())
    cumulative_returns.append((window + 1).prod() - 1)

# Qdrant in-memory
client = QdrantClient(":memory:")
collection_name = "price_patterns"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=window_size, distance=Distance.COSINE)
)

payload = [{"start_date": d} for d in start_dates]
client.upload_collection(
    collection_name=collection_name,
    vectors=price_vectors,
    payload=payload,
    ids=list(range(len(price_vectors)))
)

# UMAP 3D projection
umap_model = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", n_components=3, random_state=42)
vectors_3d = umap_model.fit_transform(price_vectors)

# KMeans clustering
n_clusters = 30
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(price_vectors)

# Visualization DataFrame
viz_df = pd.DataFrame(vectors_3d, columns=["x", "y", "z"])
viz_df["date"] = pd.to_datetime(start_dates)
viz_df["vector"] = price_vectors
viz_df["year"] = viz_df["date"].dt.year
viz_df["qdrant_id"] = list(range(len(viz_df)))
viz_df["cluster"] = cluster_labels
viz_df["volatility"] = volatilities
viz_df["cum_return"] = cumulative_returns

# Filter by year
selected_year = st.sidebar.selectbox("Filter by Year", ["All"] + sorted(viz_df["year"].unique().astype(str).tolist()))
filtered_df = viz_df if selected_year == "All" else viz_df[viz_df["year"] == int(selected_year)]

# Choose color values
if color_by == "Cluster":
    color_vals = filtered_df["cluster"]
    colorscale = "Turbo"
elif color_by == "Volatility":
    color_vals = filtered_df["volatility"]
    colorscale = "Viridis"
else:
    color_vals = filtered_df["cum_return"]
    colorscale = "Cividis"

# Tooltips
hover_text = filtered_df.apply(
    lambda row: f"Date: {row['date'].date()}<br>Volatility: {row['volatility']:.4f}<br>Return: {row['cum_return']:.2%}<br>Cluster: {row['cluster']}", axis=1
)

# Plot 3D UMAP
st.subheader("🧭 UMAP 3D Projection")
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=filtered_df["x"],
    y=filtered_df["y"],
    z=filtered_df["z"],
    mode="markers",
    marker=dict(
        size=4,
        color=color_vals,
        colorscale=colorscale,
        colorbar=dict(title=color_by),
        opacity=0.8,
    ),
    text=hover_text,
    name="Patterns",
    hovertemplate="%{text}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>"
))

# Highlight selected cluster
if highlight_cluster != "None":
    cluster_idx = int(highlight_cluster)
    highlight_df = filtered_df[filtered_df["cluster"] == cluster_idx]

    fig.add_trace(go.Scatter3d(
        x=highlight_df["x"],
        y=highlight_df["y"],
        z=highlight_df["z"],
        mode="markers",
        marker=dict(symbol='circle', size=6, color="red"),
        name=f"Highlighted Cluster {cluster_idx}"
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="UMAP X",
        yaxis_title="UMAP Y",
        zaxis_title="UMAP Z"
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    height=700
)

# Pattern selection and similarity
selected_idx = st.number_input(
    "Click a pattern index to search similar (0 to {})".format(len(filtered_df) - 1),
    min_value=0,
    max_value=len(filtered_df) - 1,
    step=1
)

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

    # Plot query + matches
    fig.add_trace(go.Scatter3d(
        x=[filtered_df.iloc[selected_idx]["x"]],
        y=[filtered_df.iloc[selected_idx]["y"]],
        z=[filtered_df.iloc[selected_idx]["z"]],
        mode="markers",
        marker=dict(symbol='circle', size=10, color="black"),
        name="Query"
    ))

    fig.add_trace(go.Scatter3d(
        x=matches["x"],
        y=matches["y"],
        z=matches["z"],
        mode="markers",
        marker=dict(symbol='circle', size=8, color="lime"),
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

# Final 3D plot
st.plotly_chart(fig, use_container_width=True)
