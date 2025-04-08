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

# UMAP + KMeans
embedding = UMAP(n_components=3, random_state=42).fit_transform(price_vectors)
k = 10
clusters = KMeans(n_clusters=k, random_state=42).fit_predict(embedding)

# Color mapping
if color_by == "Cluster":
    color_vals = clusters
elif color_by == "Volatility":
    color_vals = volatilities
else:
    color_vals = cumulative_returns

# Convert to numpy
embedding = np.array(embedding)
color_vals = np.array(color_vals)

# Sanity check to avoid None/length mismatch
if embedding.shape[0] == len(color_vals):
    scatter = go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=color_vals,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=start_dates,
        hoverinfo='text'
    )
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='UMAP-1',
            yaxis_title='UMAP-2',
            zaxis_title='UMAP-3'
        )
    )
    fig = go.Figure(data=[scatter], layout=layout)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Mismatch in embedding and color dimensions — check data integrity.")
