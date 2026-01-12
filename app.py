import streamlit as st
import numpy as np
import pandas as pd
import io
from llm_engine import LLMHandler, LLM_CRITERIA, SITES
from ahp_core import calculate_ahp, apply_scale_compression, parse_manual_matrix
from visualization import plot_ci_trends, plot_importance_counts, plot_weights_for_sites

st.set_page_config(page_title="Transparent AHP-LLM Framework", layout="wide")

st.title("🌬️ Hybrid LLM-AHP Wind Farm Site Selection")
st.markdown("Automated generation, manual input, transparent calculations, and downloadable reports.")

# Sidebar
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Data Source", ["LLM Simulation (Mock Data)", "LLM API (Real)", "Manual Input (User Paste)"])
if "LLM" in data_source:
    run_rounds = st.sidebar.slider("Number of Rounds", 1, 10, 10)
    selected_llms = st.sidebar.multiselect("Select LLMs", list(LLM_CRITERIA.keys()), default=["ChatGPT"])
else:
    run_rounds = 1 # Manual is usually 1 round
    selected_llms = st.sidebar.multiselect("Select Target LLM (for Criteria Labels)", list(LLM_CRITERIA.keys()), default=["ChatGPT"], max_selections=1)

handler = LLMHandler()

# Download
def display_matrix_process(matrix, title, round_num):
    """
    Renders the matrix, calculations, and auto-correction transparently.
    """
    with st.expander(f"🔍 View Calculations: {title} (Round {round_num})", expanded=False):
        col1, col2 = st.columns(2)
        
        
        with col1:
            st.write("**Original Input Matrix:**")
            df_orig = pd.DataFrame(matrix)
            st.dataframe(df_orig.style.format("{:.2f}"))
            
            csv_orig = df_orig.to_csv().encode('utf-8')
            st.download_button(f"⬇️ Download Matrix (CSV)", csv_orig, f"matrix_{title}_r{round_num}.csv", "text/csv")

        w, lam, ci, cr = calculate_ahp(matrix)
        
        with col2:
            st.write("**Consistency Check:**")
            st.write(f"λ_max: `{lam:.4f}`")
            st.write(f"CI: `{ci:.4f}`")
           
            if cr < 0.1:
                st.success(f"✅ Consistent (CR = {cr:.4f})")
                final_w = w
            else:
                st.error(f"❌ Inconsistent (CR = {cr:.4f}) -> Applying Scale Correction (1-9 to 1-5)")
                
                corrected_matrix = apply_scale_compression(matrix)
                w_new, lam_new, ci_new, cr_new = calculate_ahp(corrected_matrix)
                
                st.write("**Corrected Metrics:**")
                st.write(f"New CR: `{cr_new:.4f}`")
                final_w = w_new
                
                st.write("**Corrected Matrix:**")
                st.dataframe(pd.DataFrame(corrected_matrix).style.format("{:.2f}"))

        st.write("**Calculated Weights:**")
        st.bar_chart(pd.Series(final_w))
        
    return final_w, ci, cr

# Tabs
tab1, tab2, tab3 = st.tabs(["1. Criteria Analysis", "2. Site Analysis", "3. Final Rankings"])

if 'criteria_results' not in st.session_state: st.session_state.criteria_results = {}
if 'site_results' not in st.session_state: st.session_state.site_results = {}


with tab1:
    st.header("Step 1: Criteria Evaluation (6x6 Matrices)")
    
    # MANUAL INPUT LOGIC
    if data_source == "Manual Input (User Paste)":
        target_llm = selected_llms[0]
        st.info(f"Using Criteria Definitions for: **{target_llm}**")
        st.write("Paste your 6x6 Matrix below (space, tab, or comma separated rows):")
        manual_text = st.text_area("Matrix Input", height=150, value="1 2 3 4 5 6\n0.5 1 2 3 4 5\n...")
        
        if st.button("Calculate Manual Criteria"):
            try:
                mat = parse_manual_matrix(manual_text, 6)
                w, ci, cr = display_matrix_process(mat, "Manual_Criteria", 1)
                
                
                st.session_state.criteria_results = {
                    target_llm: {'cis': [ci], 'weights': [w], 'most': [], 'least': []}
                }
            except Exception as e:
                st.error(f"Error parsing matrix: {e}")

    # LLM GENERATION LOGIC
    else: 
        if st.button("Run LLM Criteria Analysis"):
            results = {llm: {'cis': [], 'weights': [], 'most': [], 'least': []} for llm in selected_llms}
            
            for llm in selected_llms:
                prompt = handler.generate_prompt_criteria(llm)
                st.subheader(f"Analysis for {llm}")
                
                for r in range(run_rounds):
                    resp = handler.get_response(llm, prompt, simulate=("Simulation" in data_source))
                    mat = np.array(resp['matrix'])
                    
                    
                    w, ci, cr = display_matrix_process(mat, f"{llm}_Criteria", r+1)
                    
                    results[llm]['cis'].append(ci)
                    results[llm]['weights'].append(w)
                    results[llm]['most'].append(resp['most_important'])
                    results[llm]['least'].append(resp['least_important'])
            
            st.session_state.criteria_results = results

    # VISUALIZATION
    if st.session_state.criteria_results:
        st.markdown("---")
        st.subheader("Aggregate Results")
        res = st.session_state.criteria_results
        
    
        fig_ci = plot_ci_trends({llm: data['cis'] for llm, data in res.items()})
        st.pyplot(fig_ci)
        buf = io.BytesIO()
        fig_ci.savefig(buf, format="png")
        st.download_button("⬇️ Download CI Graph (PNG)", buf.getvalue(), "ci_trends.png", "image/png")


with tab2:
    st.header("Step 2: Site Alternatives (10x10 Matrices)")
    target_llm_site = selected_llms[0] if selected_llms else "ChatGPT"
    
    # MANUAL INPUT LOGIC
    if data_source == "Manual Input (User Paste)":
        st.info(f"Comparing 10 Sites using Criteria from: **{target_llm_site}**")
        criteria_list = LLM_CRITERIA[target_llm_site]
        
        if 'manual_site_matrices' not in st.session_state:
            st.session_state.manual_site_matrices = {c: "" for c in criteria_list}
            
        selected_crit = st.selectbox("Select Criterion to Input Matrix", criteria_list)
        st.session_state.manual_site_matrices[selected_crit] = st.text_area(
            f"Paste 10x10 Matrix for '{selected_crit}'", 
            value=st.session_state.manual_site_matrices[selected_crit],
            height=200
        )
        
        if st.button("Process All Manual Site Matrices"):
            site_res = {c: {'weights': [], 'cis': []} for c in criteria_list}
            try:
                for c in criteria_list:
                    txt = st.session_state.manual_site_matrices[c]
                    if not txt.strip():
                        st.warning(f"Skipping empty matrix for {c}")
                        continue
                        
                    mat = parse_manual_matrix(txt, 10)
                    w, ci, cr = display_matrix_process(mat, f"Site_{c}", 1)
                    site_res[c]['weights'].append(w)
                    site_res[c]['cis'].append(ci)
                
                st.session_state.site_results = site_res
                st.success("Manual Site Data Processed!")
            except Exception as e:
                st.error(f"Error: {e}")

    # LLM GENERATION LOGIC
    else:
        if st.button(f"Run Site Analysis ({target_llm_site})"):
            criteria_list = LLM_CRITERIA[target_llm_site]
            site_res = {c: {'weights': [], 'cis': []} for c in criteria_list}
            
            progress_bar = st.progress(0)
            step = 0
            total_steps = len(criteria_list) * run_rounds
            
            for c in criteria_list:
                prompt = handler.generate_prompt_sites(target_llm_site, c)
                st.markdown(f"**Processing Criterion:** {c}")
                
                for r in range(run_rounds):
                    resp = handler.get_response(target_llm_site, prompt, simulate=("Simulation" in data_source))
                    mat = np.array(resp['matrix'])
                    
                    w, ci, cr = display_matrix_process(mat, f"Site_{c[:10]}", r+1)
                    
                    site_res[c]['weights'].append(w)
                    site_res[c]['cis'].append(ci)
                    
                    step += 1
                    progress_bar.progress(step / total_steps)
            
            st.session_state.site_results = site_res

    # VISUALIZATION
    if st.session_state.site_results:
        st.markdown("---")
        st.subheader("Site Weights Distribution")
        c_view = st.selectbox("Select Criterion to View Graph:", list(st.session_state.site_results.keys()))
        data = st.session_state.site_results.get(c_view)
        
        if data and data['weights']:
            fig_site = plot_weights_for_sites(data['weights'], c_view, target_llm_site)
            st.pyplot(fig_site)
            
            buf2 = io.BytesIO()
            fig_site.savefig(buf2, format="png")
            st.download_button(f"⬇️ Download {c_view} Graph", buf2.getvalue(), f"site_weights_{c_view}.png", "image/png")

with tab3:
    st.header("Step 3: Global Aggregation")
    
    if st.session_state.criteria_results and st.session_state.site_results:
        target_llm = list(st.session_state.criteria_results.keys())[0] # Pick the first available
        
        c_weights_all = st.session_state.criteria_results[target_llm]['weights']
        W_c = np.mean(c_weights_all, axis=0)
        
        st.subheader("A. Averaged Criteria Weights")
        df_cw = pd.DataFrame({"Criterion": LLM_CRITERIA[target_llm], "Weight": W_c})
        st.dataframe(df_cw.style.format({"Weight": "{:.4f}"}))
        
        st.subheader("B. Averaged Site Weights (Matrix)")
        W_s = np.zeros((10, 6))
        for i, c_name in enumerate(LLM_CRITERIA[target_llm]):
            if c_name in st.session_state.site_results:
                s_weights_all = st.session_state.site_results[c_name]['weights']
                if s_weights_all:
                    W_s[:, i] = np.mean(s_weights_all, axis=0)
        
        df_sw = pd.DataFrame(W_s, columns=[f"C{i+1}" for i in range(6)], index=[s.split(":")[0] for s in SITES])
        st.dataframe(df_sw.style.format("{:.4f}"))

        global_scores = np.dot(W_s, W_c)
        
        st.subheader("C. Final Site Rankings")
        df_final = pd.DataFrame({
            "Site Code": [s.split(":")[0] for s in SITES],
            "Location": [s.split(":")[1].strip() for s in SITES],
            "Final Score": global_scores
        }).sort_values(by="Final Score", ascending=False)
        
        st.dataframe(df_final.style.format({"Final Score": "{:.4f}"}))
        
        # Download Final Report
        csv_final = df_final.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Final Rankings (CSV)", csv_final, "final_rankings.csv", "text/csv")
        
        best = df_final.iloc[0]
        st.success(f" Optimal Site: **{best['Location']}** (Score: {best['Final Score']:.4f})")
        
    else:
        st.warning("Please complete Steps 1 and 2 to see final rankings.")