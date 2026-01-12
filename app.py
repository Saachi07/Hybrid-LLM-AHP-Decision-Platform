import streamlit as st
import numpy as np
import pandas as pd
import io
from llm_engine import LLMHandler, LLM_CRITERIA, SITES
from ahp_core import calculate_ahp, apply_scale_compression, parse_manual_matrix
from visualization import plot_ci_trends, plot_importance_counts, plot_weights_for_sites

st.set_page_config(page_title="Transparent AHP-LLM Framework", layout="wide")

st.title("Hybrid LLM-AHP Wind Farm Site Selection")
st.markdown("Automated generation, manual input, transparent calculations, and downloadable reports.")

st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Data Source", ["LLM Simulation (Mock Data)", "LLM API (Real)", "Manual Input (User Paste)"])

if "LLM" in data_source:
    run_rounds = st.sidebar.slider("Number of Rounds", 1, 10, 10)
    default_llms = ["ChatGPT"]
else:
    run_rounds = 1 
    
    default_llms = ["ChatGPT"]

selected_llms = st.sidebar.multiselect("Select LLMs (for Criteria Definitions)", list(LLM_CRITERIA.keys()), default=default_llms)

handler = LLMHandler()

def display_matrix_process(matrix, title, round_num):
    """
    Renders the matrix, calculations, and auto-correction transparently.
    """
    with st.expander(f"View Calculations: {title} (Round {round_num})", expanded=False):
        col1, col2 = st.columns(2)
     
        with col1:
            st.write("**Original Input Matrix:**")
            df_orig = pd.DataFrame(matrix)
            st.dataframe(df_orig.style.format("{:.2f}"))
            
            csv_orig = df_orig.to_csv().encode('utf-8')
            st.download_button(f"Download Matrix (CSV)", csv_orig, f"matrix_{title}_r{round_num}.csv", "text/csv", key=f"dl_{title}_{round_num}")

        w, lam, ci, cr = calculate_ahp(matrix)
        
        with col2:
            st.write("**Consistency Check:**")
            st.write(f"λ_max: `{lam:.4f}`")
            st.write(f"CI: `{ci:.4f}`")
            
            # Consistency status
            if cr < 0.1:
                st.success(f" Consistent (CR = {cr:.4f})")
                final_w = w
            else:
                st.error(f" Inconsistent (CR = {cr:.4f}) -> Applying Scale Correction (1-9 to 1-5)")
             
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
if 'manual_inputs' not in st.session_state: st.session_state.manual_inputs = {}

with tab1:
    st.header("Step 1: Criteria Evaluation (6x6 Matrices)")
    
    # MANUAL INPUT LOGIC
    if data_source == "Manual Input (User Paste)":
        if not selected_llms:
            st.warning("Please select at least one LLM in the sidebar.")
        else:
            st.write("Enter matrices for each selected LLM below:")
            
            for llm in selected_llms:
                with st.expander(f"📝 Input for: {llm}", expanded=True):
               
                    default_text = "1 2 3 4 5 6\n0.5 1 2 3 4 5\n0.33 0.5 1 2 3 4\n0.25 0.33 0.5 1 2 3\n0.2 0.25 0.33 0.5 1 2\n0.16 0.2 0.25 0.33 0.5 1"
                    
                    key_name = f"manual_crit_text_{llm}"
                    if key_name not in st.session_state.manual_inputs:
                        st.session_state.manual_inputs[key_name] = default_text
                        
                    manual_text = st.text_area(f"6x6 Matrix for {llm}", key=key_name, height=150)
            
            if st.button("Calculate Manual Criteria"):
                for llm in selected_llms:
                    try:
                        txt = st.session_state.manual_inputs.get(f"manual_crit_text_{llm}", "")
                        mat = parse_manual_matrix(txt, 6)
                        
                        st.markdown(f"### Results for {llm}")
                        w, ci, cr = display_matrix_process(mat, f"Manual_{llm}", 1)
                        
                        st.session_state.criteria_results[llm] = {
                            'cis': [ci], 'weights': [w], 'most': [], 'least': []
                        }
                    except Exception as e:
                        st.error(f"Error in {llm}: {e}")

    # LLM 
    else: 
        if st.button("Run LLM Criteria Analysis"):
            for llm in selected_llms:
                st.session_state.criteria_results[llm] = {'cis': [], 'weights': [], 'most': [], 'least': []}
            
            for llm in selected_llms:
                prompt = handler.generate_prompt_criteria(llm)
                st.subheader(f"Analysis for {llm}")
                
                for r in range(run_rounds):
                    resp = handler.get_response(llm, prompt, simulate=("Simulation" in data_source))
                    mat = np.array(resp['matrix'])
                    
                    w, ci, cr = display_matrix_process(mat, f"{llm}_Criteria", r+1)
                    
                    st.session_state.criteria_results[llm]['cis'].append(ci)
                    st.session_state.criteria_results[llm]['weights'].append(w)
                    st.session_state.criteria_results[llm]['most'].append(resp['most_important'])
                    st.session_state.criteria_results[llm]['least'].append(resp['least_important'])
            
    # VISUALIZATION
    if st.session_state.criteria_results:
        st.markdown("---")
        st.subheader("Aggregate Results")
        
        valid_llms = [llm for llm in selected_llms if llm in st.session_state.criteria_results]
        
        if valid_llms:
            ci_data = {llm: st.session_state.criteria_results[llm]['cis'] for llm in valid_llms}
            fig_ci = plot_ci_trends(ci_data)
            st.pyplot(fig_ci)
            
            buf = io.BytesIO()
            fig_ci.savefig(buf, format="png")
            st.download_button("⬇️ Download CI Graph (PNG)", buf.getvalue(), "ci_trends.png", "image/png")

with tab2:
    st.header("Step 2: Site Alternatives (10x10 Matrices)")
    
    if not selected_llms:
        st.warning("Please select LLMs in the sidebar first.")
    else:
        target_llm_site = st.selectbox("Select LLM Framework to Analyze:", selected_llms)
        criteria_list = LLM_CRITERIA[target_llm_site]
        
        if target_llm_site not in st.session_state.site_results:
            st.session_state.site_results[target_llm_site] = {}

        # MANUAL INPUT LOGIC
        if data_source == "Manual Input (User Paste)":
            st.info(f"Inputting 10x10 Matrices for: **{target_llm_site}**")
            
            selected_crit = st.selectbox("Select Criterion for Input:", criteria_list)
            
            input_key = f"manual_site_{target_llm_site}_{selected_crit}"
            if input_key not in st.session_state.manual_inputs:
                st.session_state.manual_inputs[input_key] = ""
                
            st.session_state.manual_inputs[input_key] = st.text_area(
                f"Paste 10x10 Matrix for '{selected_crit}'", 
                value=st.session_state.manual_inputs[input_key],
                height=200,
                key=f"textarea_{input_key}" 
            )
            
            if st.button(f"Process Manual Data for {target_llm_site}"):
            
                current_results = {c: {'weights': [], 'cis': []} for c in criteria_list}
                
                try:
                    for c in criteria_list:
                        key = f"manual_site_{target_llm_site}_{c}"
                        txt = st.session_state.manual_inputs.get(key, "")
                        
                        if not txt.strip():
                            st.warning(f"Skipping empty matrix for criterion: {c}")
                            continue
                            
                        mat = parse_manual_matrix(txt, 10)
                        w, ci, cr = display_matrix_process(mat, f"{target_llm_site}_{c[:5]}", 1)
                        current_results[c]['weights'].append(w)
                        current_results[c]['cis'].append(ci)
                    
                    
                    st.session_state.site_results[target_llm_site] = current_results
                    st.success(f"Saved manual site data for {target_llm_site}!")
                    
                except Exception as e:
                    st.error(f"Error parsing manual input: {e}")

        # LLM GENERATION LOGIC
        else:
            if st.button(f"Run Site Analysis for {target_llm_site}"):
            
                current_results = {c: {'weights': [], 'cis': []} for c in criteria_list}
                
                progress_bar = st.progress(0)
                step = 0
                total_steps = len(criteria_list) * run_rounds
                
                for c in criteria_list:
                    prompt = handler.generate_prompt_sites(target_llm_site, c)
                    st.markdown(f"**Processing:** {c}")
                    
                    for r in range(run_rounds):
                        resp = handler.get_response(target_llm_site, prompt, simulate=("Simulation" in data_source))
                        mat = np.array(resp['matrix'])
                        
                       
                        w, ci, cr = display_matrix_process(mat, f"Site_{c[:10]}", r+1)
                        
                        current_results[c]['weights'].append(w)
                        current_results[c]['cis'].append(ci)
                        
                        step += 1
                        progress_bar.progress(step / total_steps)
                
                st.session_state.site_results[target_llm_site] = current_results

        
        if target_llm_site in st.session_state.site_results and st.session_state.site_results[target_llm_site]:
            st.markdown("---")
            st.subheader(f"Results for {target_llm_site}")
            
            res_data = st.session_state.site_results[target_llm_site]
            valid_criteria = [k for k in res_data.keys() if res_data[k]['weights']]
            
            if valid_criteria:
                c_view = st.selectbox("Select Criterion to View Graph:", valid_criteria)
                data_c = res_data.get(c_view)
                
                if data_c and data_c['weights']:
                    fig_site = plot_weights_for_sites(data_c['weights'], c_view, target_llm_site)
                    st.pyplot(fig_site)
                    
                    buf2 = io.BytesIO()
                    fig_site.savefig(buf2, format="png")
                    st.download_button(f"⬇️ Download Graph", buf2.getvalue(), f"site_weights_{target_llm_site}_{c_view[:5]}.png", "image/png")


with tab3:
    st.header("Step 3: Global Aggregation")
    
    has_crit = bool(st.session_state.criteria_results)
    has_site = bool(st.session_state.site_results)
    
    if has_crit and has_site:
        available_llms = [llm for llm in selected_llms if llm in st.session_state.criteria_results and llm in st.session_state.site_results]
        
        if not available_llms:
            st.warning("Data incomplete. Please ensure you have run BOTH Criteria and Site analysis for the same LLM.")
        else:
            target_llm = st.selectbox("Select LLM for Final Report:", available_llms)
            
            c_res = st.session_state.criteria_results[target_llm]
            s_res = st.session_state.site_results[target_llm]
            
            if not c_res['weights']:
                st.error("Missing weights for criteria.")
            else:
                W_c = np.mean(c_res['weights'], axis=0)
                
                st.subheader("A. Averaged Criteria Weights")
                df_cw = pd.DataFrame({"Criterion": LLM_CRITERIA[target_llm], "Weight": W_c})
                st.dataframe(df_cw.style.format({"Weight": "{:.4f}"}))
                
                st.subheader("B. Averaged Site Weights (Matrix)")
                
                W_s = np.zeros((10, 6))
                valid_cols = []
                
                for i, c_name in enumerate(LLM_CRITERIA[target_llm]):
                    if c_name in s_res and s_res[c_name]['weights']:
                        W_s[:, i] = np.mean(s_res[c_name]['weights'], axis=0)
                        valid_cols.append(c_name)
                    else:
                        st.warning(f"No site data found for criterion: {c_name}. Treating as zeros.")
                
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
                st.download_button("⬇️ Download Final Rankings (CSV)", csv_final, f"final_rankings_{target_llm}.csv", "text/csv")
                
                best = df_final.iloc[0]
                st.success(f"🏆 Optimal Site ({target_llm}): **{best['Location']}** (Score: {best['Final Score']:.4f})")
    else:
        st.warning("Please complete Steps 1 and 2 to see final rankings.")