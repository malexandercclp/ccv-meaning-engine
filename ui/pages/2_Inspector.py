"""
CCV Meaning Engine - Object Inspector
"""

import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import sys
import html

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core import CCVProcessor, AXES, get_valence_tagged_axis

st.set_page_config(
    page_title="Object Inspector - CCV",
    page_icon="🔍",   # FIXED
    layout="wide"
)

if 'processor' not in st.session_state:
    st.session_state.processor = CCVProcessor()

processor = st.session_state.processor

def get_valence_color(valence):
    return {'+': '#22c55e', '-': '#ef4444', '0': '#6366f1'}.get(valence, '#6366f1')

def main():
    st.markdown("## Object Inspector")

    objects = processor.buffer.list_objects()
    if not objects:
        st.info("No objects in buffer yet.")
        return

    selected = st.selectbox("Select object", objects)
    summary = processor.inspect_object(selected)
    if not summary:
        st.error("Could not load object.")
        return

    st.markdown("---")
    st.markdown(f"### {selected.upper()}")

    axes_data = summary.get("axes", {})

    for axis in AXES:
        axis_info = axes_data.get(axis)
        if not axis_info:
            continue

        st.markdown(f"**{axis.upper()}**")
        cols = st.columns(3)

        for i, valence in enumerate(['+', '0', '-']):
            with cols[i]:
                data = axis_info.get(valence)
                if not data:
                    st.markdown(f"*[{valence}] empty*")
                    continue

                raw = data.get("content", "")
                content = html.escape(raw)

                cryst = data.get("crystallization", 0.5)
                entropy = data.get("entropy", 0.1)

                st.markdown(f"""
                <div style="border-left: 3px solid {get_valence_color(valence)};
                            padding: 0.75rem; background:#15151f;">
                    <p>{content}</p>
                    <small>C:{cryst:.2f} | E:{entropy:.2f}</small>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
