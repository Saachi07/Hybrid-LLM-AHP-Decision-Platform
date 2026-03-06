import streamlit as st
import numpy as np
import pandas as pd
import io
import plotly.express as px
from llm_engine import LLMHandler, LLM_CRITERIA, SITES
from ahp_core import calculate_ahp, apply_scale_compression, parse_manual_matrix, parse_file_to_matrix
from visualization import plot_ci_trends, plot_importance_counts, plot_weights_for_sites

st.set_page_config(page_title="Transparent AHP-LLM Framework", layout="wide")

st.title("Hybrid LLM-AHP Wind Farm Site Selection")
st.markdown("Automated generation, manual input, transparent calculations, and downloadable reports.")

# Sidebar Configuration
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Data Source", ["LLM Simulation (Mock Data)", "LLM API (Real)", "Manual Input (User Paste)", "File Upload (Excel/Word)"])

# Dynamic Round Configuration
if "LLM" in data_source:
    run_rounds = st.sidebar.slider("Number of Rounds", 1, 10, 10)
    default_llms = ["ChatGPT"]
else:
    # Ask user for number of inputs in Manual Mode
    run_rounds = st.sidebar.number_input("Number of Manual Rounds (Inputs)", min_value=1, value=1, step=1)
    default_llms = ["ChatGPT"]

selected_llms = st.sidebar.multiselect("Select LLMs (for Criteria Definitions)", list(LLM_CRITERIA.keys()), default=default_llms)

handler = LLMHandler()

# GPS Coordinates for the 10 Alberta Sites
SITE_COORDS = {
    "Canmore": {"lat": 51.0899, "lon": -115.3441},
    "Lethbridge": {"lat": 49.6956, "lon": -112.8396},
    "Hanna": {"lat": 51.6389, "lon": -111.9304},
    "Fort McMurray": {"lat": 56.7265, "lon": -111.3803},
    "Medicine Hat": {"lat": 50.0416, "lon": -110.6775},
    "Cold Lake": {"lat": 54.4642, "lon": -110.1825},
    "Drumheller": {"lat": 51.4650, "lon": -112.7106},
    "Grande Prairie": {"lat": 55.1708, "lon": -118.7947},
    "Fort Chipewyan": {"lat": 58.7145, "lon": -111.1528},
    "High Level": {"lat": 58.5165, "lon": -117.1365}
}


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
            st.write(f"Enter **{run_rounds}** matrices for each selected LLM below:")
            
            for llm in selected_llms:
                with st.expander(f" Input for: {llm}", expanded=True):
               
                    for r in range(run_rounds):
                        st.markdown(f"**Round {r+1}**")
                        default_text = "1 2 3 4 5 6\n0.5 1 2 3 4 5\n0.33 0.5 1 2 3 4\n0.25 0.33 0.5 1 2 3\n0.2 0.25 0.33 0.5 1 2\n0.16 0.2 0.25 0.33 0.5 1"
                        
                        key_name = f"manual_crit_text_{llm}_r{r}"
                        if key_name not in st.session_state.manual_inputs:
                            st.session_state.manual_inputs[key_name] = default_text
                            
                        st.session_state.manual_inputs[key_name] = st.text_area(f"Matrix (Round {r+1})", key=f"ta_{key_name}", value=st.session_state.manual_inputs[key_name], height=150)
            
            if st.button("Calculate Manual Criteria"):
                for llm in selected_llms:
                    # Initialize result storage for this LLM
                    st.session_state.criteria_results[llm] = {'cis': [], 'weights': [], 'most': [], 'least': []}
                    
                    try:
                        st.markdown(f"### Results for {llm}")
                        for r in range(run_rounds):
                            key_name = f"manual_crit_text_{llm}_r{r}"
                            txt = st.session_state.manual_inputs.get(key_name, "")
                            mat = parse_manual_matrix(txt, 6)
                            
                            w, ci, cr = display_matrix_process(mat, f"Manual_{llm}", r+1)
                            
                            # Store results
                            st.session_state.criteria_results[llm]['cis'].append(ci)
                            st.session_state.criteria_results[llm]['weights'].append(w)
                            
                    except Exception as e:
                        st.error(f"Error in {llm} (Round {r+1}): {e}")

    # LLM GENERATION LOGIC
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
            # Handle plotting for variable number of rounds
            ci_data = {llm: st.session_state.criteria_results[llm]['cis'] for llm in valid_llms}
            fig_ci = plot_ci_trends(ci_data)
            st.pyplot(fig_ci)
            
            buf = io.BytesIO()
            fig_ci.savefig(buf, format="png")
            st.download_button("Download CI Graph (PNG)", buf.getvalue(), "ci_trends.png", "image/png")

    # File upload
    elif data_source == "File Upload (Excel/Word)":
        if not selected_llms:
            st.warning("Please select at least one LLM in the sidebar.")
        else:
            st.write(f"Upload **{run_rounds}** files for each selected LLM:")
            for llm in selected_llms:
                with st.expander(f"Upload for: {llm}", expanded=True):
                    uploaded_files = st.file_uploader(f"Choose files for {llm}", type=['xlsx', 'docx', 'csv'], accept_multiple_files=True, key=f"file_crit_{llm}")
                    
                    if st.button(f"Process Files for {llm}"):
                        if len(uploaded_files) != run_rounds:
                            st.error(f"Please upload exactly {run_rounds} files.")
                        else:
                            st.session_state.criteria_results[llm] = {'cis': [], 'weights': [], 'most': [], 'least': []}
                            for i, file in enumerate(uploaded_files):
                                # Use the parsing function from ahp_core.py
                                mat = parse_file_to_matrix(file, 6) 
                                w, ci, cr = display_matrix_process(mat, f"File_{llm}", i+1)
                                st.session_state.criteria_results[llm]['cis'].append(ci)
                                st.session_state.criteria_results[llm]['weights'].append(w)

with tab2:
    st.header("Step 2: Site Alternatives (10x10 Matrices)")
    
    if not selected_llms:
        st.warning("Please select LLMs in the sidebar first.")
    else:
        target_llm_site = st.selectbox("Select LLM Framework to Analyze:", selected_llms)
        criteria_list = LLM_CRITERIA[target_llm_site]
        
        # Ensure storage exists
        if target_llm_site not in st.session_state.site_results:
            st.session_state.site_results[target_llm_site] = {}

        # MANUAL INPUT LOGIC
        if data_source == "Manual Input (User Paste)":
            st.info(f"Inputting 10x10 Matrices for: **{target_llm_site}**")
            st.write("Select a criterion, enter data for all rounds, then click Process.")
            
            selected_crit = st.selectbox("Select Criterion for Input:", criteria_list)
            
            # Generate input boxes for the number of rounds requested
            with st.form(key=f"form_{target_llm_site}_{selected_crit}"):
                for r in range(run_rounds):
                    st.markdown(f"**Round {r+1}**")
                    input_key = f"manual_site_{target_llm_site}_{selected_crit}_r{r}"
                    
                    if input_key not in st.session_state.manual_inputs:
                        st.session_state.manual_inputs[input_key] = ""
                        
                    st.session_state.manual_inputs[input_key] = st.text_area(
                        f"Paste Matrix (Round {r+1})", 
                        value=st.session_state.manual_inputs[input_key],
                        height=150,
                        key=f"textarea_{input_key}" 
                    )
                
                submit_button = st.form_submit_button(label=f"Process Rounds for '{selected_crit}'")
            
            if submit_button:
                # Initialize specific list for this criterion
                if selected_crit not in st.session_state.site_results[target_llm_site]:
                     st.session_state.site_results[target_llm_site][selected_crit] = {'weights': [], 'cis': []}
                
                # Reset lists to avoid appending duplicates if button pressed twice
                current_weights = []
                current_cis = []

                try:
                    st.markdown(f"### Processing {selected_crit}")
                    for r in range(run_rounds):
                        key = f"manual_site_{target_llm_site}_{selected_crit}_r{r}"
                        txt = st.session_state.manual_inputs.get(key, "")
                        
                        if not txt.strip():
                            st.warning(f"Skipping empty matrix for Round {r+1}")
                            continue
                            
                        mat = parse_manual_matrix(txt, 10)
                        w, ci, cr = display_matrix_process(mat, f"{target_llm_site}_{selected_crit[:5]}", r+1)
                        current_weights.append(w)
                        current_cis.append(ci)
                    
                    # Update Session State
                    st.session_state.site_results[target_llm_site][selected_crit]['weights'] = current_weights
                    st.session_state.site_results[target_llm_site][selected_crit]['cis'] = current_cis
                    st.success(f"Saved {len(current_weights)} rounds for {selected_crit}!")
                    
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

        # VISUALIZATION
        if target_llm_site in st.session_state.site_results and st.session_state.site_results[target_llm_site]:
            st.markdown("---")
            st.subheader(f"Results for {target_llm_site}")
            
            res_data = st.session_state.site_results[target_llm_site]
            # Filter criteria that actually have data
            valid_criteria = [k for k in res_data.keys() if res_data[k]['weights']]
            
            if valid_criteria:
                c_view = st.selectbox("Select Criterion to View Graph:", valid_criteria)
                data_c = res_data.get(c_view)
                
                if data_c and data_c['weights']:
                    fig_site = plot_weights_for_sites(data_c['weights'], c_view, target_llm_site)
                    st.pyplot(fig_site)
                    
                    buf2 = io.BytesIO()
                    fig_site.savefig(buf2, format="png")
                    st.download_button(f"Download Graph", buf2.getvalue(), f"site_weights_{target_llm_site}_{c_view[:5]}.png", "image/png")
        
    # NEW FILE UPLOAD LOGIC
    if data_source == "File Upload (Excel/Word)":
        st.info(f"Uploading 10x10 Matrices for: **{target_llm_site}**")
        selected_crit = st.selectbox("Select Criterion for Upload:", criteria_list)
        
        uploaded_site_files = st.file_uploader(f"Upload {run_rounds} files for {selected_crit}", type=['xlsx', 'docx', 'csv'], accept_multiple_files=True)
        
        if st.button(f"Process Site Files for {selected_crit}"):
            if len(uploaded_site_files) != run_rounds:
                st.error(f"Please upload exactly {run_rounds} files.")
            else:
                if selected_crit not in st.session_state.site_results[target_llm_site]:
                    st.session_state.site_results[target_llm_site][selected_crit] = {'weights': [], 'cis': []}
                
                current_weights = []
                current_cis = []
                for i, file in enumerate(uploaded_site_files):
                    mat = parse_file_to_matrix(file, 10) # 10x10 for sites
                    w, ci, cr = display_matrix_process(mat, f"File_{selected_crit[:5]}", i+1)
                    current_weights.append(w)
                    current_cis.append(ci)
                
                st.session_state.site_results[target_llm_site][selected_crit]['weights'] = current_weights
                st.session_state.site_results[target_llm_site][selected_crit]['cis'] = current_cis
                st.success(f"Processed {len(current_weights)} files for {selected_crit}!")

with tab3:
    st.header("Step 3: Global Aggregation & Spatial Heatmap")
    
    has_crit = bool(st.session_state.criteria_results)
    has_site = bool(st.session_state.site_results)
    
    if has_crit and has_site:
        # Find LLMs that have data in both steps
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
                # 1. Averaged Criteria Weights
                W_c = np.mean(c_res['weights'], axis=0)
                
                st.subheader("A. Averaged Criteria Weights")
                df_cw = pd.DataFrame({"Criterion": LLM_CRITERIA[target_llm], "Weight": W_c})
                st.dataframe(df_cw.style.format({"Weight": "{:.4f}"}))
                
                # 2. Averaged Site Weights
                st.subheader("B. Averaged Site Weights (Matrix)")
                
                W_s = np.zeros((10, 6))
                valid_cols = []
                
                # Fill W_s matrix
                for i, c_name in enumerate(LLM_CRITERIA[target_llm]):
                    if c_name in s_res and s_res[c_name]['weights']:
                        W_s[:, i] = np.mean(s_res[c_name]['weights'], axis=0)
                        valid_cols.append(c_name)
                    else:
                        st.warning(f"No site data found for criterion: {c_name}. Treating as zeros.")
                
                df_sw = pd.DataFrame(W_s, columns=[f"C{i+1}" for i in range(6)], index=[s.split(":")[0] for s in SITES])
                st.dataframe(df_sw.style.format("{:.4f}"))

                # 3. Global Synthesis
                global_scores = np.dot(W_s, W_c)
                
                st.subheader("C. Final Site Rankings")
                
                # Build final dataframe with coordinates
                locations = [s.split(":")[1].strip() for s in SITES]
                lats = [SITE_COORDS[loc]["lat"] for loc in locations]
                lons = [SITE_COORDS[loc]["lon"] for loc in locations]
                
                df_final = pd.DataFrame({
                    "Site Code": [s.split(":")[0] for s in SITES],
                    "Location": locations,
                    "Final Score": global_scores,
                    "Latitude": lats,
                    "Longitude": lons
                }).sort_values(by="Final Score", ascending=False)
                
                # Display table without the lat/lon cluttering it up
                st.dataframe(df_final[["Site Code", "Location", "Final Score"]].style.format({"Final Score": "{:.4f}"}))
                
                # --- NEW MAPPING SECTION ---
                st.subheader(" Geographical Site Scoring ")
                
                # Create interactive map
                fig_map = px.scatter_mapbox(
                    df_final, 
                    lat="Latitude", 
                    lon="Longitude", 
                    hover_name="Location", 
                    hover_data={"Final Score": ':.4f', "Latitude": False, "Longitude": False},
                    color="Final Score",
                    size="Final Score",
                    color_continuous_scale=px.colors.sequential.YlOrRd, # Yellow to Red heatmap colors
                    size_max=25,
                    zoom=4.5, 
                    center={"lat": 54.5, "lon": -115.0}, # Centered roughly on Alberta
                    mapbox_style="carto-positron",
                    title=f"Optimal Wind Farm Locations based on {target_llm} AHP Framework"
                )
                
                fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
                # ---------------------------
                
                # Download Final Report
                csv_final = df_final.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Final Rankings (CSV)", csv_final, f"final_rankings_{target_llm}.csv", "text/csv")
                
                best = df_final.iloc[0]
                st.success(f"Optimal Site ({target_llm}): **{best['Location']}** (Score: {best['Final Score']:.4f})")
    else:
        st.warning("Please complete Steps 1 and 2 to see final rankings.")
    