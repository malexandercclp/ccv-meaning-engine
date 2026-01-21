"""
CCV Meaning Engine - Main Application

Multi-page Streamlit app for the CCV system.
"""

import streamlit as st
from pathlib import Path
import sys
import html

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import CCVProcessor, get_config, SystemState

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CCV Meaning Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SHARED STYLING
# ============================================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
    }
    
    .result-card {
        background: #15151f;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .object-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .object-new {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid #22c55e;
        color: #86efac;
    }
    
    .object-existing {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid #6366f1;
        color: #a5b4fc;
    }
    
    .axis-card {
        background: #1a1a25;
        border-left: 3px solid;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    if 'processor' not in st.session_state:
        st.session_state.processor = CCVProcessor()
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None

init_session_state()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_state_color(state):
    return {
        SystemState.ENGAGED: '#22c55e',
        SystemState.TENTATIVE: '#eab308',
        SystemState.DISENGAGED: '#f97316',
        SystemState.WITHDRAWN: '#ef4444',
    }.get(state, '#6366f1')

def get_state_emoji(state):
    return {
        SystemState.ENGAGED: '🟢',
        SystemState.TENTATIVE: '🟡',
        SystemState.DISENGAGED: '🟠',
        SystemState.WITHDRAWN: '🔴',
    }.get(state, '⚪')

def get_valence_color(valence):
    return {'+': '#22c55e', '-': '#ef4444', '0': '#6366f1'}.get(valence, '#6366f1')

def get_action_style(action):
    styles = {
        'created': ('✨', '#22c55e', 'Created'),
        'reinforced': ('💎', '#6366f1', 'Reinforced'),
        'contradicted': ('⚡', '#ef4444', 'Contradicted'),
        'accumulated': ('📥', '#eab308', 'Accumulated'),
    }
    return styles.get(action, ('•', '#6a6a7a', action))

# ============================================================================
# RESULT DISPLAY
# ============================================================================

def display_result(result):
    """Display the processing result with full details."""
    
    state_color = get_state_color(result.system_state)
    state_emoji = get_state_emoji(result.system_state)
    
    # === HEADER: System State ===
    st.markdown(f"""
    <div class="result-card" style="border-left: 4px solid {state_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="color: #6a6a7a; font-size: 0.8rem; text-transform: uppercase;">Input Processed</span>
                <div style="color: #e8e8ed; font-size: 1rem; margin-top: 0.25rem; font-style: italic;">
                    "{html.escape(result.input_text[:100])}{'...' if len(result.input_text) > 100 else ''}"
                </div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 2rem;">{state_emoji}</span>
                <div style="color: {state_color}; font-weight: 600; text-transform: uppercase;">
                    {result.system_state.value}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === CCV METRICS ===
    st.markdown("#### 📊 CCV Dynamics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clarity", f"{result.clarity:.3f}", help="Overall navigability of the buffer")
    col2.metric("Arousal", f"{result.arousal:.3f}", help="Initial attention signal from novelty/mismatch")
    col3.metric("Valence Intensity", f"{result.valence_intensity:.3f}", help="How significant this feels")
    col4.metric("Curiosity", f"{result.curiosity:.3f}", help="Engagement with novelty (gated by clarity)")
    
    st.markdown("---")
    
    # === OBJECTS SECTION ===
    st.markdown("#### 🎯 Objects Affected")
    
    col_new, col_existing = st.columns(2)
    
    with col_new:
        st.markdown("**New Objects Created:**")
        if result.new_objects:
            for obj in result.new_objects:
                st.markdown(f'<span class="object-tag object-new">✨ {html.escape(obj)}</span>', unsafe_allow_html=True)
        else:
            st.markdown("*None - all objects already existed*")
    
    with col_existing:
        st.markdown("**Existing Objects Updated:**")
        existing = [o for o in result.affected_objects if o not in result.new_objects]
        if existing:
            for obj in existing:
                st.markdown(f'<span class="object-tag object-existing">{html.escape(obj)}</span>', unsafe_allow_html=True)
        else:
            st.markdown("*None*")
    
    st.markdown("---")
    
    # === AXIS TRAVERSALS ===
    st.markdown("#### 🔄 Axis Traversals")
    
    if result.object_updates:
        for obj_update in result.object_updates:
            valence_color = get_valence_color(obj_update.valence)
            new_badge = "🆕 " if obj_update.is_new else ""
            
            with st.expander(
                f"{new_badge}**{obj_update.object_name.upper()}** — "
                f"Valence: `{obj_update.valence}` | "
                f"Sentiment: `{obj_update.sentiment:.2f}` | "
                f"Mismatch: `{obj_update.mismatch:.3f}`",
                expanded=True
            ):
                if obj_update.axis_updates:
                    for axis_update in obj_update.axis_updates:
                        icon, color, label = get_action_style(axis_update.action)
                        
                        # Calculate deltas
                        e_delta = axis_update.entropy_after - axis_update.entropy_before
                        c_delta = axis_update.crystallization_after - axis_update.crystallization_before
                        
                        e_delta_str = f"+{e_delta:.3f}" if e_delta > 0 else f"{e_delta:.3f}"
                        c_delta_str = f"+{c_delta:.3f}" if c_delta > 0 else f"{c_delta:.3f}"
                        
                        blocked = "🚫 BLOCKED" if axis_update.blocked else ""
                        
                        st.markdown(f"""
                        <div class="axis-card" style="border-color: {color};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 1.1rem;">{icon}</span>
                                    <code style="color: {valence_color}; font-weight: bold; font-size: 1rem;">
                                        {axis_update.axis}{obj_update.valence}
                                    </code>
                                    <span style="color: {color}; margin-left: 0.5rem; text-transform: uppercase; font-size: 0.8rem;">
                                        {label}
                                    </span>
                                    <span style="color: #ef4444; font-weight: bold; margin-left: 0.5rem;">{blocked}</span>
                                </div>
                                <div style="color: #6a6a7a; font-size: 0.8rem;">
                                    mismatch: {axis_update.mismatch:.3f}
                                </div>
                            </div>
                            <div style="margin-top: 0.5rem; display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.85rem;">
                                <div>
                                    <span style="color: #6a6a7a;">Entropy:</span>
                                    <span style="color: #e8e8ed;">{axis_update.entropy_before:.3f} → {axis_update.entropy_after:.3f}</span>
                                    <span style="color: {'#ef4444' if e_delta > 0 else '#22c55e'};">({e_delta_str})</span>
                                </div>
                                <div>
                                    <span style="color: #6a6a7a;">Crystallization:</span>
                                    <span style="color: #e8e8ed;">{axis_update.crystallization_before:.3f} → {axis_update.crystallization_after:.3f}</span>
                                    <span style="color: {'#22c55e' if c_delta > 0 else '#ef4444'};">({c_delta_str})</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("*No axis updates recorded*")
    else:
        st.info("No object updates recorded")
    
    # === SPECIAL EVENTS ===
    has_special_events = (
        result.blocked_axes or 
        result.competition_events or 
        result.spillover_opportunities or
        (hasattr(result, 'friction_events') and result.friction_events)
    )
    
    if has_special_events:
        st.markdown("---")
        st.markdown("#### ⚠️ Special Events")
        
        # Cross-valence friction (NEW)
        if hasattr(result, 'friction_events') and result.friction_events:
            st.markdown("**🔥 Cross-Valence Friction** (contradictions across valence slots):")
            for friction in result.friction_events:
                st.markdown(f"""
                <div style="
                    background: rgba(249, 115, 22, 0.1);
                    border-left: 3px solid #f97316;
                    padding: 0.75rem;
                    margin: 0.5rem 0;
                    border-radius: 0 6px 6px 0;
                ">
                    <strong>{friction.object_name.upper()}</strong> [{friction.axis}]<br>
                    <span style="color: #22c55e;">NEW ({friction.new_valence}):</span> "{html.escape(friction.new_content)}"<br>
                    <span style="color: #ef4444;">vs EXISTING ({friction.opposing_valence}):</span> "{html.escape(friction.opposing_content)}"<br>
                    <span style="color: #6a6a7a; font-size: 0.85rem;">
                        Confidence: {friction.contradiction_confidence:.2f} | 
                        Entropy added: +{friction.entropy_added_to_new:.3f} (new), +{friction.entropy_added_to_opposing:.3f} (existing)
                    </span>
                </div>
                """, unsafe_allow_html=True)
        
        if result.blocked_axes:
            st.markdown("**🚫 Blocked Axes** (entropy threshold exceeded):")
            for blocked in result.blocked_axes:
                st.markdown(f"- `{blocked}`")
        
        if result.competition_events:
            st.markdown("**⚔️ Valence Competitions:**")
            for comp in result.competition_events:
                st.markdown(
                    f"- **{comp.object_name}**: `{comp.axis}{comp.dominant_valence}` "
                    f"dominated `{comp.axis}{comp.subordinate_valence}` "
                    f"(crystallization: {comp.dominant_crystallization:.2f})"
                )
        
        if result.spillover_opportunities:
            st.markdown("**🔀 Spillover Opportunities** (alternative routes when blocked):")
            for spill in result.spillover_opportunities:
                st.markdown(f"- `{spill}`")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    processor = st.session_state.processor

    # === SIDEBAR ===
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem; color: #8b5cf6;">●</span>
            <div>
                <strong style="color: #e8e8ed;">CCV Engine</strong><br>
                <span style="color: #6a6a7a; font-size: 0.75rem;">v1.0</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        
        status = processor.get_buffer_status()
        st.metric("Total Objects", status['num_objects'])
        st.metric("Global Clarity", f"{status['clarity']:.2f}")
        st.metric("Connections", status['num_edges'])
        st.metric("Blocked Axes", status['blocked_axes'])

        st.markdown("---")
        st.markdown("**Navigation**")
        st.page_link("app.py", label="Process Input", icon="⚡")
        st.page_link("pages/1_Network.py", label="Network View", icon="🕸️")
        st.page_link("pages/2_Inspector.py", label="Object Inspector", icon="🔍")
        st.page_link("pages/3_Settings.py", label="Settings & Data", icon="⚙️")

    # === MAIN CONTENT ===
    st.markdown("## 🧠 Process Experience")
    st.markdown("*Enter an experience to process through the CCV meaning engine*")
    
    # Input area
    experience = st.text_area(
        "Experience Input",
        label_visibility="collapsed",
        placeholder="e.g., 'Coffee gives me energy in the morning' or 'My father was always critical of me'",
        height=100
    )
    
    # Process button
    if st.button("⚡ Process Experience", type="primary", use_container_width=False):
        if experience.strip():
            with st.spinner("Processing through CCV engine..."):
                result = processor.process_input(experience)
                st.session_state.last_result = result
                st.session_state.history.append(result)
            st.rerun()
        else:
            st.warning("Please enter an experience to process.")
    
    # === DISPLAY RESULT ===
    st.markdown("---")
    
    if st.session_state.last_result is not None:
        st.markdown("### 📋 Processing Result")
        display_result(st.session_state.last_result)
    else:
        st.markdown("""
        <div style="
            background: #15151f;
            border: 2px dashed #2a2a3a;
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            color: #6a6a7a;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🧠</div>
            <div style="font-size: 1.2rem; color: #9898a8;">No experiences processed yet</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">
                Enter an experience above and click "Process Experience" to see how the CCV system interprets it.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # === HISTORY ===
    if len(st.session_state.history) > 1:
        st.markdown("---")
        with st.expander(f"📜 Processing History ({len(st.session_state.history)} total)", expanded=False):
            for i, hist in enumerate(reversed(st.session_state.history[:-1])):
                if i >= 10:
                    st.markdown(f"*...and {len(st.session_state.history) - 11} more*")
                    break
                emoji = get_state_emoji(hist.system_state)
                st.markdown(
                    f"{emoji} **{hist.input_text[:50]}...** — "
                    f"C:{hist.clarity:.2f} A:{hist.arousal:.2f} "
                    f"*({hist.timestamp})*"
                )


if __name__ == "__main__":
    main()
