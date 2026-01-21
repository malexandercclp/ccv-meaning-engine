"""
CCV Meaning Engine - Network View

Visualize the recurrence buffer as an interactive network graph.
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core import (
    CCVProcessor, AXES, VALENCES,
    get_valence_tagged_axis
)

# ============================================================================
# PAGE CONFIGURATION (FIXED)
# ============================================================================

st.set_page_config(
    page_title="Network View - CCV",
    page_icon="🕸️",   # valid emoji
    layout="wide"
)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'processor' not in st.session_state:
    st.session_state.processor = CCVProcessor()

processor = st.session_state.processor


# ============================================================================
# HELPERS
# ============================================================================

def get_valence_color(valence):
    return {'+': '#22c55e', '-': '#ef4444', '0': '#6366f1'}.get(valence, '#6366f1')


def create_network_graph(buffer, filter_node="Full System", layout_type="spring"):
    G = buffer.graph
    
    if filter_node != "Full System" and filter_node in G:
        neighbors = set(G.predecessors(filter_node)) | set(G.successors(filter_node))
        neighbors.add(filter_node)
        G = G.subgraph(neighbors)
    
    if len(G.nodes()) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data yet. Process some experiences to see the network.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='#9898a8')
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=600,
        )
        return fig
    
    # Layout selection
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    
    # Edge traces
    edge_traces = []
    for u, v, key, data in G.edges(keys=True, data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        valence = data.get('valence', '0')
        weight = data.get('weight', 1.0)
        
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=max(0.5, min(4, weight)),
                color=get_valence_color(valence)
            ),
            opacity=0.5,
            hoverinfo='text',
            hovertext=f"{u} → {v}<br>Type: {key}<br>Weight: {weight:.2f}",
            showlegend=False
        ))
    
    # Node trace
    node_x, node_y, node_text = [], [], []
    node_colors, node_sizes, hover_texts = [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node.upper())
        
        node_data = G.nodes[node]
        sentiment = node_data.get('sentiment', 0)
        
        if sentiment > 0.2:
            node_colors.append('#22c55e')
        elif sentiment < -0.2:
            node_colors.append('#ef4444')
        else:
            node_colors.append('#6366f1')
        
        degree = G.degree(node)
        node_sizes.append(max(25, min(60, 20 + degree * 5)))
        
        hover_texts.append(
            f"<b>{node.upper()}</b><br>"
            f"Sentiment: {sentiment:.2f}<br>"
            f"Connections: {degree}<br>"
            f"In: {G.in_degree(node)} | Out: {G.out_degree(node)}"
        )
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=hover_texts,
        text=node_text,
        textposition='top center',
        textfont=dict(size=11, color='#e8e8ed', family='Inter'),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='#1a1a25'),
            opacity=0.95
        ),
        showlegend=False
    )
    
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.markdown("""
    <h2 style="margin-bottom: 0.25rem;">Network View</h2>
    <p style="color: #6a6a7a; margin-top: 0;">
        Visualize the recurrence buffer as a network graph
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        all_nodes = ["Full System"] + sorted(processor.buffer.list_objects())
        filter_node = st.selectbox("Filter by object:", options=all_nodes, index=0)
    
    with col2:
        layout_type = st.selectbox(
            "Layout:",
            options=["spring", "circular", "kamada_kawai"],
            index=0
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    
    fig = create_network_graph(processor.buffer, filter_node, layout_type)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
    
    st.markdown("---")
    st.markdown("#### Network Statistics")
    
    G = processor.buffer.graph
    
    if len(G.nodes()) > 0:
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        col_s1.metric("Nodes", len(G.nodes()))
        col_s2.metric("Edges", len(G.edges()))
        
        density = nx.density(G) if len(G.nodes()) > 1 else 0
        col_s3.metric("Density", f"{density:.3f}")
        
        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
        col_s4.metric("Avg Degree", f"{avg_degree:.1f}")
        
    else:
        st.info("No data in the buffer yet. Process some experiences first.")


if __name__ == "__main__":
    main()
