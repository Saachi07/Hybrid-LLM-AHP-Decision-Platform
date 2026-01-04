import numpy as np
import random
import re
import json
from ahp_core import parse_fraction

# Configuration for specific LLM Criteria
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
        criteria = LLM_CRITERIA.get(llm_name, [])
        criteria_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(criteria)])
        
        prompt = f"""
        Construct a 6x6 AHP pairwise comparison matrix using the full Saaty 1–9 scale.
        Criteria:
        {criteria_text}
        
        Which criteria do you think is the Most Important and Least Important?
        
        Output strictly in JSON format:
        {{
            "most_important": "Criteria Name",
            "least_important": "Criteria Name",
            "matrix": [[1, 2, ...], [0.5, 1, ...], ...]
        }}
        """
        return prompt

    def generate_prompt_sites(self, llm_name, criterion_name):
        sites_text = " | ".join(SITES)
        prompt = f"""
        Construct a 10x10 AHP pairwise comparison matrix for the following sites based on the criterion: '{criterion_name}'.
        Sites: {sites_text}
        
        Output strictly in JSON format with key "matrix".
        """
        return prompt

    def get_response(self, llm_name, prompt, simulate=True):
        """
        If simulate=True, generates a random consistent-ish matrix.
        If simulate=False, would call actual APIs (OpenAI/Anthropic placeholders included).
        """
        if simulate:
            return self._simulate_response(llm_name, prompt)
        else:
            # Placeholder for actual API integration
            # You would perform actual API calls here based on llm_name
            return self._simulate_response(llm_name, prompt)

    def _simulate_response(self, llm_name, prompt):
        """Generates dummy data for testing the pipeline."""
        is_site_matrix = "10x10" in prompt
        n = 10 if is_site_matrix else 6
        
        # Generate a semi-consistent random matrix
        vec = np.random.rand(n)
        vec = vec / np.sum(vec)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j: matrix[i, j] = 1
                else: 
                    ratio = vec[i] / vec[j]
                    # Add noise
                    noise = random.uniform(0.8, 1.2)
                    matrix[i, j] = ratio * noise
        
        # Determine most/least based on the vector
        criteria = LLM_CRITERIA.get(llm_name, [])
        most_imp = criteria[np.argmax(vec)] if not is_site_matrix else ""
        least_imp = criteria[np.argmin(vec)] if not is_site_matrix else ""

        return {
            "most_important": most_imp,
            "least_important": least_imp,
            "matrix": matrix.tolist()
        }