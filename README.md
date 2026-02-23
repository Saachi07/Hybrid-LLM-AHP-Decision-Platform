## Research
# A Hybrid LLM-AHP Decision Making Framework for Wind Farm Site Selection and Criteria Evaluation

# Prompt for criteria matrices
Construct a 6x6 matrix using the 1–9 scale to evaluate offshore renewable energy sites. Ensure that the matrix reflects meaningful and realistic differences in importance among the following six high-level criteria, based on domain-specific considerations. Use a wide range of values from the full scale (1–9) and their reciprocals to show clear contrasts in priority where justified. Force use all 1-9 scale. Use the following mentioned criteria 
* C1: Wind Resource Quality
* C2: Grid Connection and Infrastructure
* C3: Land Availability and Terrain
* C4: Environmental and Ecological Impact
* C5: Community and Land Use Considerations
* C6: Economic Viability


# Prompt for site matrices

Create six 10×10 matrices (one per criterion) using the 1–9 scale and reciprocals (fractions only), where the sites are S1: Canmore | S2: Lethbridge | S3: Hanna | S4: Fort McMurray | S5: Medicine Hat | S6: Cold Lake | S7: Drumheller | S8: Grande Prairie | S9: Fort Chipewyan | S10: High Level. 
The main criteria are 1. Wind Resource Quality 2. Grid Connection and Infrastructure 3. Environmental and Ecological Impact 4. Land Availability and Terrain 5. Community and Land Use Considerations 6. Economic Viability 
Instructions for the matrix generation: 
* Matrices are 10×10 with S1–S10 labels; diagonals=1 
* Use the scale (1–9 + 1/2 … 1/9) with diverse, realistic values (avoid clustering around 1) 
* Each matrix should reflect site differences under its criterion. 
Output: A 10x10 symmetric matrix showing pairwise comparisons with appropriate values and reciprocals


# How to run
Run the command below in terminal
python -m streamlit run app.py