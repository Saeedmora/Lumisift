import os
import random
from typing import Dict

class AxisEvaluator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            # from openai import OpenAI
            # self.client = OpenAI(api_key=self.api_key)
            print("AxisEvaluator: OpenAI API key detected (Mocking fully for MVP stability unless requested).")
        else:
            print("AxisEvaluator: No API key found. Using heuristic/random mode.")

    def evaluate(self, text: str) -> Dict[str, float]:
        """
        Analyzes the text and returns values for standard axes:
        - Relevance (0.0 - 1.0)
        - Risk (0.0 - 1.0)
        - Time (0.0 - 1.0) (e.g., urgency or temporal reference)
        """
        if self.api_key:
            # TODO: Implement actual LLM call here for production
            # response = self.client.chat.completions.create(...)
            return self._heuristic_evaluate(text)
        else:
            return self._heuristic_evaluate(text)

    def _heuristic_evaluate(self, text: str) -> Dict[str, float]:
        """
        A simple heuristic evaluator for the MVP.
        """
        text_lower = text.lower()
        
        # Heuristic for 'Risk'
        risk_keywords = ['error', 'fail', 'critical', 'warning', 'danger', 'risk', 'security']
        risk_score = sum(1 for w in risk_keywords if w in text_lower) / len(risk_keywords)
        risk_score = min(1.0, risk_score + random.uniform(0.0, 0.1)) # Add a little jitter

        # Heuristic for 'Time'
        time_keywords = ['now', 'urgent', 'immediately', 'schedule', 'deadline', 'date', 'time']
        time_score = sum(1 for w in time_keywords if w in text_lower) / len(time_keywords)
        time_score = min(1.0, time_score + random.uniform(0.0, 0.1))

        # Heuristic for 'Relevance' (mocking context relevance as generic importance)
        relevance_score = random.uniform(0.5, 1.0) # Assume input text is generally relevant

        return {
            "risk": round(risk_score, 2),
            "time": round(time_score, 2),
            "relevance": round(relevance_score, 2)
        }
