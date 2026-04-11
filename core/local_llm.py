"""Local LLM Integration for Logical Rooms MVP."""

import os
import json
import re
import random
from typing import Dict

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False


class LocalLLMEvaluator:
    """Evaluates text using TinyLlama or heuristic fallback."""
    
    DEFAULT_MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    def __init__(self, model_path=None, n_ctx=512, n_gpu_layers=0):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.model = None
        self.is_ready = False
        
        if LLAMA_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
                self.is_ready = True
                print("TinyLlama loaded successfully!")
            except Exception as e:
                print("Failed to load model:", e)
        else:
            print("LocalLLMEvaluator: Using heuristic fallback mode.")
    
    def evaluate(self, text: str) -> Dict[str, float]:
        """Evaluate text and return axis values."""
        if self.is_ready and self.model:
            return self._llm_evaluate(text)
        return self._heuristic_evaluate(text)
    
    def _llm_evaluate(self, text: str) -> Dict[str, float]:
        """Use TinyLlama to evaluate text."""
        system_msg = "Return JSON with risk, time, relevance scores (0.0-1.0)"
        user_msg = "Evaluate: " + text[:200]
        
        try:
            output = self.model(
                system_msg + "\nText: " + user_msg,
                max_tokens=100,
                temperature=0.1,
                stop=["```", "\n\n"]
            )
            
            response = output["choices"][0]["text"]
            return self._parse_json_response(response)
        except Exception as e:
            print("LLM evaluation failed:", e)
            return self._heuristic_evaluate(text)
    
    def _parse_json_response(self, response: str) -> Dict[str, float]:
        """Parse JSON from LLM response."""
        try:
            match = re.search(r'\{[^}]+\}', response)
            if match:
                data = json.loads(match.group())
                return {
                    "risk": max(0.0, min(1.0, float(data.get("risk", 0.5)))),
                    "time": max(0.0, min(1.0, float(data.get("time", 0.5)))),
                    "relevance": max(0.0, min(1.0, float(data.get("relevance", 0.5))))
                }
        except:
            pass
        return self._heuristic_evaluate("")
    
    def _heuristic_evaluate(self, text: str) -> Dict[str, float]:
        """Heuristic fallback evaluation."""
        text_lower = text.lower()
        
        risk_keywords = ['error', 'fail', 'critical', 'warning', 'danger', 'risk', 'security', 'breach', 'attack']
        risk_score = sum(1 for w in risk_keywords if w in text_lower) / len(risk_keywords)
        risk_score = min(1.0, risk_score + random.uniform(0.0, 0.1))
        
        time_keywords = ['now', 'urgent', 'immediately', 'asap', 'deadline', 'critical']
        time_score = sum(1 for w in time_keywords if w in text_lower) / len(time_keywords)
        time_score = min(1.0, time_score + random.uniform(0.0, 0.1))
        
        relevance_score = random.uniform(0.5, 1.0)
        
        return {
            "risk": round(risk_score, 2),
            "time": round(time_score, 2),
            "relevance": round(relevance_score, 2)
        }
    
    def generate_response(self, user_message: str, context: str = "") -> str:
        """Generate a chat response."""
        if not self.is_ready or not self.model:
            return self._heuristic_response(user_message, context)
        
        try:
            prompt = "Context: " + context[:300] + "\nUser: " + user_message + "\nAssistant:"
            output = self.model(prompt, max_tokens=256, temperature=0.7)
            return output["choices"][0]["text"].strip()
        except Exception as e:
            return self._heuristic_response(user_message, context)
    
    def _heuristic_response(self, user_message: str, context: str) -> str:
        """Generate a simple response without LLM."""
        if context:
            return "Based on the available context, here is what I found: " + context[:200] + "..."
        return "I understand you are asking about: " + user_message[:100] + ". Please provide more context for a detailed answer."
