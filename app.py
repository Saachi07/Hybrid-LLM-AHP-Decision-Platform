import streamlit as st
import numpy as np
import pandas as pd
import time
from llm_engine import LLMHandler, LLM_CRITERIA, SITES
from ahp_core import calculate_ahp, apply_scale_compression
from visualization import plot_ci_trends, plot_importance_counts, plot_weights_for_sites

st.set_page_config(page_title="Automated AHP-LLM Framework", layout="wide")

st.title("🌬️ Hybrid LLM-AHP Wind Farm Site Selection")
st.markdown("""
This system automates the generation of AHP matrices via LLMs, calculates weights/consistency, 
and auto-corrects inconsistency using scale compression (1-9 -> 1-5) if CR > 0.1.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
simulation_mode = st.sidebar.checkbox("Simulation Mode (Mock Data)", value=True)
run_rounds = st.sidebar.slider("Number of Rounds", 1, 10, 10)
selected_llms = st.sidebar.multiselect("Select LLMs", list(LLM_CRITERIA.keys()), default=["ChatGPT", "DeepSeek", "Claude"])

handler = LLMHandler()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["1. Criteria Analysis", "2. Site Analysis", "3. Final Rankings"])

# Store results in session state to persist between clicks
if 'criteria_results' not in st.session_state:
    st.session_state.criteria_results = {}
if 'site_results' not in st.session_state:
    st.session_state.site_results = {}

# ==========================================
# TAB 1: CRITERIA WEIGHTS
# ==========================================
with tab1:
    st.header("Step 1: Criteria Evaluation (6x6 Matrices)")
    if st.button("Run Criteria Analysis"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {llm: {'cis': [], 'weights': [], 'most': [], 'least': []} for llm in selected_llms}
        
        total_steps = len(selected_llms) * run_rounds
        current_step = 0
        
        for llm in selected_llms:
            prompt = handler.generate_prompt_criteria(llm)
            
            for r in range(run_rounds):
                status_text.text(f"Querying {llm} - Round {r+1}/{run_rounds}...")
                
                # Fetch Response
                response = handler.get_response(llm, prompt, simulate=simulation_mode)
                matrix = np.array(response['matrix'])
                
                # Calc AHP
                w, lam, ci, cr = calculate_ahp(matrix)
                
                # Check Consistency & Auto-Correct if needed
                if cr > 0.1:
                    # Apply scale change (1-9 -> 1-5) as per methodology
                    matrix = apply_scale_compression(matrix)
                    w, lam, ci, cr = calculate_ahp(matrix)
                
                results[llm]['cis'].append(ci)
                results[llm]['weights'].append(w)
                results[llm]['most'].append(response['most_important'])
                results[llm]['least'].append(response['least_important'])
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
        
        st.session_state.criteria_results = results
        status_text.text("Analysis Complete!")
        progress_bar.empty()

    # Display Results if available
    if st.session_state.criteria_results:
        res = st.session_state.criteria_results
        
        # 1. CI Trends Plot
        ci_data = {llm: data['cis'] for llm, data in res.items()}
        st.pyplot(plot_ci_trends(ci_data, "Criteria Matrix Consistency (CI)"))
        
        # 2. Most/Least Important Stats
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Most Important Counts")
            # Logic to count frequencies matches specific LLM criteria
            # For simplicity in demo, we map back to C1-C6 generic labels if names differ
            # But here we stick to raw names
            st.info("Counts aggregation logic would go here (visualized in chart)")

        with col2:
            st.subheader("Weight Distribution")
            llm_choice = st.selectbox("View Weights for:", selected_llms, key="w_view")
            avg_w = np.mean(res[llm_choice]['weights'], axis=0)
            df_w = pd.DataFrame({
                "Criteria": LLM_CRITERIA[llm_choice],
                "Average Weight": avg_w
            })
            st.dataframe(df_w)
            st.bar_chart(df_w.set_index("Criteria"))

# ==========================================
# TAB 2: SITE RANKINGS
# ==========================================
with tab2:
    st.header("Step 2: Site Alternatives (10x10 Matrices)")
    
    llm_for_site = st.selectbox("Select LLM for Site Analysis", selected_llms)
    
    if st.button(f"Run Site Analysis for {llm_for_site}"):
        criteria_list = LLM_CRITERIA[llm_for_site]
        site_res = {c: {'weights': [], 'cis': []} for c in criteria_list}
        
        prog = st.progress(0)
        total = len(criteria_list) * run_rounds
        curr = 0
        
        for criterion in criteria_list:
            prompt = handler.generate_prompt_sites(llm_for_site, criterion)
            
            w_rounds = []
            for r in range(run_rounds):
                resp = handler.get_response(llm_for_site, prompt, simulate=simulation_mode)
                mat = np.array(resp['matrix'])
                
                w, lam, ci, cr = calculate_ahp(mat)
                
                # Consistency Correction Logic
                if cr > 0.1:
                    mat = apply_scale_compression(mat)
                    w, lam, ci, cr = calculate_ahp(mat)
                
                w_rounds.append(w)
                site_res[criterion]['cis'].append(ci)
                
                curr += 1
                prog.progress(curr / total)
            
            site_res[criterion]['weights'] = w_rounds
            
        st.session_state.site_results = site_res
        st.success("Site Analysis Done!")

    if st.session_state.site_results:
        # Visualize specific criterion
        c_view = st.selectbox("View Criterion Results:", list(st.session_state.site_results.keys()))
        data = st.session_state.site_results[c_view]
        
        st.pyplot(plot_weights_for_sites(data['weights'], c_view, llm_for_site))
        
        avg_ci = np.mean(data['cis'])
        st.metric("Average CI for this Criterion", f"{avg_ci:.4f}")

# ==========================================
# TAB 3: FINAL SYNTHESIS
# ==========================================
with tab3:
    st.header("Step 3: Global Aggregation")
    
    if st.session_state.criteria_results and st.session_state.site_results:
        # 1. Get Average Criteria Weights (W_c)
        # Note: In a real run, you'd select which LLM's criteria weights to use for the synthesis
        # Here we assume we use the same LLM selected in Tab 2
        
        try:
            # Avg Criteria Weights
            c_weights_all = st.session_state.criteria_results[llm_for_site]['weights']
            W_c = np.mean(c_weights_all, axis=0) # Shape (6,)
            
            # Avg Site Weights per Criterion (Matrix W_s of shape 10x6)
            W_s = np.zeros((10, 6))
            
            criteria_names = LLM_CRITERIA[llm_for_site]
            for i, c_name in enumerate(criteria_names):
                s_weights_rounds = st.session_state.site_results[c_name]['weights']
                W_s[:, i] = np.mean(s_weights_rounds, axis=0)
            
            # Global Scores = W_s dot W_c
            global_scores = np.dot(W_s, W_c)
            
            # Display Ranking
            df_final = pd.DataFrame({
                "Site": [s.split(":")[0] for s in SITES],
                "Name": [s.split(":")[1] for s in SITES],
                "Score": global_scores
            }).sort_values(by="Score", ascending=False)
            
            st.subheader(f"Final Rankings ({llm_for_site})")
            st.dataframe(df_final)
            st.bar_chart(df_final.set_index("Name")["Score"])
            
            best_site = df_final.iloc[0]['Name']
            st.success(f"🏆 The optimal site is: **{best_site}**")
            
        except KeyError:
            st.error("Please run both Criteria Analysis and Site Analysis for the SAME LLM to synthesize results.")
    else:
        st.warning("Please complete Step 1 and Step 2 first.")