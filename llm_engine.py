import numpy as np
import random
import json

LLM_CRITERIA = {
    "ChatGPT": ["Wind Resource Availability", "Proximity to transmission", "Environmental Impact", "Regulatory and zoning compliance", "Land Use and topography", "Socioeconomic and community considerations"],
    "DeepSeek": ["Wind resource", "Grid connection", "Environental impact", "Site accessibility", "Social Acceptance and land use", "Economic Impact"],
    "Claude": ["Wind resource Quality", "Grid connection and infrastructure", "Environmental and ecological impact", "Land availability and terrain", "Community and Land value", "Ecoomic viability"],
    "Grok": ["Wind resource Quality", "Grid infrastructure proximity", "Environmental impact", "Topography and terrain", "Land availability and cost", "Socio economic factors"],
    "Perplexity": ["Wind resource availibility", "Proximity to electrical grid", "Environmental and social consideration", "Accessibility (infrastructure)", "Land use and topography", "Regulatory and economic factors"],
    "Gemini": ["Wind resource availability", "Proximity to elctrical grid", "Environment and wildlife impact", "Site access and logistics", "Land use and topography", "Regulatory and social acceptance"]
}

SITES = [
    "S1: Canmore", "S2: Lethbridge", "S3: Hanna", "S4: Fort McMurray", 
    "S5: Medicine Hat", "S6: Cold Lake", "S7: Drumheller", 
    "S8: Grande Prairie", "S9: Fort Chipewyan", "S10: High Level"
]

class LLMHandler:
    def __init__(self, api_keys=None):
        self.api_keys = api_keys or {}

    def generate_prompt_criteria(self, llm_name):
        criteria_list = LLM_CRITERIA.get(llm_name, [])
        criteria_text = ", ".join([f"{i+1}) {c}" for i, c in enumerate(criteria_list)])
        
        prompt = f"""
        Construct a 6x6 AHP pairwise comparison matrix using the full Saaty 1-9 scale to evaluate offshore renewable energy sites. Ensure that the matrix reflects meaningful and realistic differences in importance among the following six high-level criteria, based on domain-specific considerations. Use a wide range of values from the full scale (1-9) and their reciprocals to show clear contrasts in priority where justified.
        The six criteria (each with sub-criteria for context) are: {criteria_text}
        
        Which criteria do you think is the Most Important and Least Important?
        
        Output strictly in JSON format:
        {{
            "most_important": "Criteria Name",
            "least_important": "Criteria Name",
            "matrix": [[1, ...], ..., [..., 1]]
        }}
        """
        return prompt

    def generate_prompt_sites(self, llm_name, criterion_name):
        criteria_list = LLM_CRITERIA.get(llm_name, [])
        criteria_text = "; ".join([f"{i+1}) {c}" for i, c in enumerate(criteria_list)])
        sites_text = " | ".join(SITES)
        
        prompt = f"""
        Create six 10x10 Analytic Hierarchy Process (AHP) pairwise comparison matrices (one per criterion) using the Saaty 1-9 scale and reciprocals (fractions only), where the sites are {sites_text}. The main criteria are {criteria_text}.
        
        Instructions for the matrix generation:
        Matrices are 10x10 with S1-S10 labels; diagonals = 1
        Use the full Saaty scale (1-9 + 1/2 ... 1/9) with diverse, realistic values (avoid clustering around 1)
        Each matrix should reflect site differences under its criterion.
        
        For this specific request, return ONLY the matrix for criterion: "{criterion_name}".
        
        Output strictly in JSON format with key "matrix".
        """
        return prompt

    def get_response(self, llm_name, prompt, simulate=True):
        if simulate:
            return self._simulate_response(llm_name, prompt)
        else:
            # Placeholder: Connect your API calls here (OpenAI, Anthropic, etc.)
            return self._simulate_response(llm_name, prompt)

    def _simulate_response(self, llm_name, prompt):
        """Generates dummy data for testing."""
        is_site_matrix = "10x10" in prompt
        n = 10 if is_site_matrix else 6
        
        # Random consistent matrix simulation
        vec = np.random.rand(n)
        vec = vec / np.sum(vec)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j: matrix[i, j] = 1
                else: 
                    ratio = vec[i] / vec[j]
                    noise = random.uniform(0.5, 1.5) # Add noise to trigger inconsistency checks
                    matrix[i, j] = ratio * noise
        
        criteria = LLM_CRITERIA.get(llm_name, [])
        most_imp = criteria[np.argmax(vec)] if not is_site_matrix else ""
        least_imp = criteria[np.argmin(vec)] if not is_site_matrix else ""

        return {
            "most_important": most_imp,
            "least_important": least_imp,
            "matrix": matrix.tolist()
        }