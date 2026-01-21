"""
CCV Meaning Engine - Settings & Data

Configuration parameters, backup, export, and data management.
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core import (
    CCVProcessor, CCVConfig, get_config, set_config, get_preset, PRESETS
)

# ============================================================================
# PAGE CONFIGURATION (FIXED)
# ============================================================================

st.set_page_config(
    page_title="Settings - CCV",
    page_icon="⚙️",   # FIXED: valid emoji
    layout="wide"
)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'processor' not in st.session_state:
    st.session_state.processor = CCVProcessor()

processor = st.session_state.processor


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.markdown("""
    <h2 style="margin-bottom: 0.25rem;">Settings & Data</h2>
    <p style="color: #6a6a7a; margin-top: 0;">Configure parameters, manage backups, and export data</p>
    """, unsafe_allow_html=True)
    
    config = get_config()
    
    col_settings, col_data = st.columns([1, 1])
    
    # ================= SETTINGS =================
    with col_settings:
        st.markdown("### Parameters")
        
        st.markdown("**Load Preset**")
        preset_names = list(PRESETS.keys())
        selected_preset = st.selectbox(
            "Preset:",
            options=["(current)"] + preset_names,
            label_visibility="collapsed"
        )
        
        if selected_preset != "(current)":
            if st.button("Load Preset", use_container_width=True):
                new_config = get_preset(selected_preset)
                set_config(new_config)
                st.success(f"Loaded '{selected_preset}' preset!")
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("**Entropy Dynamics**")
        st.markdown(
            '<p style="color: #6a6a7a; font-size: 0.8rem;">Controls how contradiction affects axes</p>',
            unsafe_allow_html=True
        )
        
        entropy_threshold = st.slider(
            "Threshold (axis blocks above this)",
            0.3, 0.95, config.entropy.threshold, 0.05
        )
        
        entropy_multiplier = st.slider(
            "Multiplier (contradiction accumulation rate)",
            1.0, 6.0, config.entropy.multiplier, 0.5
        )
        
        entropy_decay = st.slider(
            "Decay (reduction on successful traversal)",
            0.9, 0.99, config.entropy.decay, 0.01
        )
        
        st.markdown("---")
        
        st.markdown("**Crystallization Dynamics**")
        st.markdown(
            '<p style="color: #6a6a7a; font-size: 0.8rem;">Controls how beliefs strengthen</p>',
            unsafe_allow_html=True
        )
        
        cryst_boost = st.slider(
            "Boost (reinforcement amount)",
            0.01, 0.2, config.crystallization.boost, 0.01
        )
        
        st.markdown("---")
        
        st.markdown("**Valence Dynamics**")
        
        dominance_threshold = st.slider(
            "Dominance Threshold (gap for competition)",
            0.1, 0.5, config.valence.dominance_threshold, 0.05
        )
        
        competition_penalty = st.slider(
            "Competition Penalty Ratio",
            0.05, 0.3, config.valence.competition_penalty, 0.05
        )
        
        st.markdown("---")
        
        st.markdown("**Curiosity Dynamics**")
        
        clarity_floor = st.slider(
            "Clarity Floor (minimum for curiosity)",
            0.1, 0.5, config.curiosity.clarity_floor, 0.05
        )
        
        st.markdown("---")
        
        if st.button("Apply All Changes", type="primary", use_container_width=True):
            config.entropy.threshold = entropy_threshold
            config.entropy.multiplier = entropy_multiplier
            config.entropy.decay = entropy_decay
            config.crystallization.boost = cryst_boost
            config.valence.dominance_threshold = dominance_threshold
            config.valence.competition_penalty = competition_penalty
            config.curiosity.clarity_floor = clarity_floor
            set_config(config)
            st.success("Parameters updated!")
    
    # ================= DATA =================
    with col_data:
        st.markdown("### Data Management")
        
        status = processor.get_buffer_status()
        
        st.markdown(f"""
        <div style="
            background: #15151f;
            border: 1px solid #2a2a3a;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <p style="color: #6a6a7a; font-size: 0.75rem; text-transform: uppercase;">Buffer Statistics</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                <div>Objects<br><strong>{status['num_objects']}</strong></div>
                <div>Edges<br><strong>{status['num_edges']}</strong></div>
                <div>Clarity<br><strong>{status['clarity']:.3f}</strong></div>
                <div>Blocked Axes<br><strong>{status['blocked_axes']}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("**Backup**")
        backup_name = st.text_input("Backup name (optional):")
        
        if st.button("Create Backup", use_container_width=True):
            processor.buffer.create_backup(backup_name or None)
            st.success("Backup created!")
        
        st.markdown("---")
        
        st.markdown("**Export**")
        export_json = json.dumps(processor.buffer.to_dict(), indent=2)
        
        st.download_button(
            "Download JSON Export",
            export_json,
            file_name=f"ccv_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Restore from backup
        st.markdown("**Restore from Backup**")
        backups = processor.buffer.list_backups()
        
        if backups:
            backup_options = {b['name']: b['path'] for b in backups}
            selected_backup = st.selectbox(
                "Select backup:",
                options=list(backup_options.keys()),
                label_visibility="collapsed"
            )
            
            if st.button("Restore Selected Backup", use_container_width=True):
                try:
                    processor.buffer.restore_backup(backup_options[selected_backup])
                    st.success(f"Restored from backup: {selected_backup}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to restore: {e}")
        else:
            st.markdown("*No backups available*")
        
        st.markdown("---")
        
        # DANGER ZONE - Clear Buffer
        st.markdown("**⚠️ Danger Zone**")
        
        st.markdown("""
        <div style="
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        ">
            <p style="color: #fca5a5; margin: 0; font-size: 0.9rem;">
                <strong>Clear Buffer</strong> will permanently delete all objects, connections, 
                and accumulated data from the recurrence buffer. A backup will be created automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Two-step confirmation for clear
        if 'confirm_clear' not in st.session_state:
            st.session_state.confirm_clear = False
        
        if not st.session_state.confirm_clear:
            if st.button("🗑️ Clear Buffer...", use_container_width=True, type="secondary"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("Are you sure? This cannot be undone (but a backup will be created).")
            
            col_cancel, col_confirm = st.columns(2)
            
            with col_cancel:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()
            
            with col_confirm:
                if st.button("🗑️ Yes, Clear Everything", use_container_width=True, type="primary"):
                    processor.buffer.clear(create_backup=True)
                    # Also clear session state history
                    if 'history' in st.session_state:
                        st.session_state.history = []
                    if 'last_result' in st.session_state:
                        st.session_state.last_result = None
                    st.session_state.confirm_clear = False
                    st.success("Buffer cleared! A backup was created.")
                    st.rerun()


if __name__ == "__main__":
    main()
