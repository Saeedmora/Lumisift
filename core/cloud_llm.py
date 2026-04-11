"""
Cloud LLM Integration - Gemini, OpenAI, Claude
================================================
Ermöglicht die Nutzung von Cloud-APIs mit Logical Rooms Kompression.
"""

import os
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded .env from {env_path}")
except ImportError:
    pass  # dotenv not installed, use system env vars

# Try to import API clients
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class APIConfig:
    """Configuration for cloud LLM APIs."""
    gemini_key: Optional[str] = None
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None
    default_provider: str = "gemini"
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Load API keys from environment variables."""
        return cls(
            gemini_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
            openai_key=os.getenv("OPENAI_API_KEY"),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    
    @classmethod
    def from_file(cls, path: str = "api_config.json") -> 'APIConfig':
        """Load API keys from config file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()
    
    def save(self, path: str = "api_config.json"):
        """Save API keys to config file."""
        data = {
            "gemini_key": self.gemini_key,
            "openai_key": self.openai_key,
            "anthropic_key": self.anthropic_key,
            "default_provider": self.default_provider
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_configured(self, provider: str = None) -> bool:
        """Check if a provider is configured."""
        provider = provider or self.default_provider
        if provider == "gemini":
            return bool(self.gemini_key)
        elif provider == "openai":
            return bool(self.openai_key)
        elif provider == "anthropic":
            return bool(self.anthropic_key)
        return False


class CloudLLM:
    """
    Cloud LLM Integration with Logical Rooms compression.
    
    Features:
    - Token-optimized prompts via Logical Rooms preprocessing
    - Multi-provider support (Gemini, OpenAI, Anthropic)
    - Cost tracking
    """
    
    def __init__(self, config: APIConfig = None):
        self.config = config or APIConfig.from_env()
        self.token_stats = {
            "total_input": 0,
            "total_output": 0,
            "total_saved": 0,
            "calls": 0
        }
        self._init_providers()
    
    def _init_providers(self):
        """Initialize available providers."""
        self.providers = {}
        
        # Gemini
        if GEMINI_AVAILABLE and self.config.gemini_key:
            genai.configure(api_key=self.config.gemini_key)
            self.providers["gemini"] = genai.GenerativeModel("gemini-1.5-flash")
            print("✅ Gemini API configured")
        
        # OpenAI
        if OPENAI_AVAILABLE and self.config.openai_key:
            openai.api_key = self.config.openai_key
            self.providers["openai"] = openai
            print("✅ OpenAI API configured")
    
    def generate(self, 
                 prompt: str, 
                 provider: str = None,
                 compressed_context: str = None,
                 axes_metadata: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate response using cloud LLM.
        
        Args:
            prompt: The user's question/request
            provider: Which API to use (gemini, openai)
            compressed_context: Pre-compressed context from Logical Rooms
            axes_metadata: 7-axes scores for context prioritization
        
        Returns:
            Dict with response, tokens used, and cost estimate
        """
        provider = provider or self.config.default_provider
        
        if provider not in self.providers:
            return {"error": f"Provider {provider} not configured", "response": None}
        
        # Build optimized prompt with metadata
        system_context = self._build_context(compressed_context, axes_metadata)
        full_prompt = f"{system_context}\n\nQuestion: {prompt}"
        
        # Estimate original vs compressed tokens
        original_tokens = len(prompt.split()) + (len(compressed_context.split()) if compressed_context else 0)
        compressed_tokens = len(full_prompt.split())
        
        try:
            if provider == "gemini":
                response = self._call_gemini(full_prompt)
            elif provider == "openai":
                response = self._call_openai(full_prompt)
            else:
                return {"error": f"Unknown provider: {provider}", "response": None}
            
            # Update stats
            self.token_stats["total_input"] += compressed_tokens
            self.token_stats["total_saved"] += max(0, original_tokens - compressed_tokens)
            self.token_stats["calls"] += 1
            
            return {
                "response": response,
                "provider": provider,
                "tokens_used": compressed_tokens,
                "tokens_saved": original_tokens - compressed_tokens,
                "cost_estimate": self._estimate_cost(compressed_tokens, provider)
            }
            
        except Exception as e:
            return {"error": str(e), "response": None}
    
    def _build_context(self, context: str, axes: Dict[str, float]) -> str:
        """Build optimized context with metadata."""
        if not context:
            return ""
        
        parts = ["Context (compressed by Logical Rooms):"]
        
        if axes:
            # Add priority hints based on axes
            if axes.get("risk", 0) > 0.7:
                parts.append("[HIGH PRIORITY - Security Critical]")
            if axes.get("temporal", 0) > 0.5:
                parts.append("[TIME SENSITIVE]")
            if axes.get("trust", 0) < 0.3:
                parts.append("[UNVERIFIED SOURCE]")
        
        parts.append(context)
        return "\n".join(parts)
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        model = self.providers["gemini"]
        response = model.generate_content(prompt)
        return response.text
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        client = self.providers["openai"]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def _estimate_cost(self, tokens: int, provider: str) -> float:
        """Estimate cost in USD."""
        # Prices per 1K tokens (approximate)
        prices = {
            "gemini": 0.000125,  # Gemini 1.5 Flash
            "openai": 0.00015,   # GPT-4o-mini
        }
        return tokens * prices.get(provider, 0.001) / 1000
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            **self.token_stats,
            "savings_pct": (self.token_stats["total_saved"] / 
                          max(1, self.token_stats["total_input"] + self.token_stats["total_saved"])) * 100
        }


# ============================================================================
# CLI for API key configuration
# ============================================================================

def configure_api_keys():
    """Interactive CLI to configure API keys."""
    print("\n" + "=" * 50)
    print("🔑 Logical Rooms - API Key Configuration")
    print("=" * 50)
    
    config = APIConfig.from_file()
    
    print("\nCurrent configuration:")
    print(f"  Gemini: {'✅ Configured' if config.gemini_key else '❌ Not set'}")
    print(f"  OpenAI: {'✅ Configured' if config.openai_key else '❌ Not set'}")
    print(f"  Default: {config.default_provider}")
    
    print("\n1. Set Gemini API Key")
    print("2. Set OpenAI API Key")
    print("3. Set Default Provider")
    print("4. Save & Exit")
    print("5. Exit without saving")
    
    while True:
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            key = input("Gemini API Key: ").strip()
            if key:
                config.gemini_key = key
                print("✅ Gemini key set")
        
        elif choice == "2":
            key = input("OpenAI API Key: ").strip()
            if key:
                config.openai_key = key
                print("✅ OpenAI key set")
        
        elif choice == "3":
            provider = input("Default provider (gemini/openai): ").strip().lower()
            if provider in ["gemini", "openai"]:
                config.default_provider = provider
                print(f"✅ Default set to {provider}")
        
        elif choice == "4":
            config.save()
            print("✅ Configuration saved to api_config.json")
            break
        
        elif choice == "5":
            print("❌ Exiting without saving")
            break


if __name__ == "__main__":
    configure_api_keys()
