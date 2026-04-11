"""
7-Axes Evaluator for Logical Rooms
====================================
Evaluates text on 7 semantic dimensions using TinyLlama or
deterministic heuristics. Supports batch processing and
calibration from the fine-tuning engine.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional

from core.atom import OntologyCategory, AXIS_NAMES

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

try:
    from core.nf4_loader import NF4Evaluator
    NF4_AVAILABLE = True
except ImportError:
    NF4_AVAILABLE = False


# ─── Keyword Dictionaries ──────────────────────────────────────────────────


CATEGORY_KEYWORDS = {
    OntologyCategory.HUMAN: [
        # General
        "user", "employee", "person", "team", "customer", "human", "staff",
        "patient", "volunteer", "subject", "participant", "individual",
        "researcher", "scientist", "clinician", "physician", "investigator",
        "operator", "analyst", "engineer", "developer", "administrator",
        "population", "cohort", "group", "sample size", "demographics",
        "organism", "host", "donor", "recipient",
    ],
    OntologyCategory.PROCESS: [
        # General
        "workflow", "process", "procedure", "step", "method", "operation",
        "protocol", "pipeline", "assay", "screening", "selection",
        # Biotech / Lab
        "mutagenesis", "evolution", "directed evolution", "incubation",
        "purification", "extraction", "amplification", "cloning", "ligation",
        "transformation", "transfection", "fermentation", "cultivation",
        "centrifugation", "filtration", "dialysis", "elution", "titration",
        "crystallization", "lyophilization", "homogenization",
        # Computational
        "optimization", "iteration", "simulation", "modeling", "docking",
        "alignment", "clustering", "classification", "regression", "training",
        "inference", "validation", "calibration", "normalization",
        # Clinical
        "treatment", "therapy", "administration", "dosing", "monitoring",
        "diagnosis", "prognosis", "intervention", "trial", "study",
    ],
    OntologyCategory.TECHNOLOGY: [
        # General IT
        "server", "api", "database", "software", "system", "code", "algorithm",
        "platform", "framework", "tool", "device", "instrument", "sensor",
        # Biotech / Lab tech
        "crispr", "cas9", "cas12", "pcr", "qpcr", "rt-pcr",
        "sequencing", "ngs", "rna-seq", "chip-seq", "mass spectrometry",
        "chromatography", "hplc", "electrophoresis", "gel", "western blot",
        "elisa", "facs", "flow cytometry", "microscopy", "spectroscopy",
        "nmr", "x-ray", "cryo-em", "crystallography",
        "plasmid", "vector", "construct", "reporter", "biosensor",
        "microarray", "bioreactor", "fermenter",
        # Computational tools
        "machine learning", "deep learning", "neural network",
        "transformer", "language model", "embedding", "encoder", "decoder",
        "bayesian", "gaussian", "random forest", "svm",
        "alphafold", "rosetta", "pymol", "blast",
    ],
    OntologyCategory.INFORMATION: [
        # General
        "data", "report", "document", "file", "record", "information",
        "dataset", "database", "repository", "archive", "library",
        # Scientific data
        "sequence", "structure", "genome", "proteome", "transcriptome",
        "metabolome", "phenotype", "genotype", "mutation", "variant",
        "residue", "amino acid", "nucleotide", "codon", "motif",
        "domain", "fold", "conformation", "topology",
        # Measurements / results
        "spectrum", "chromatogram", "electropherogram", "thermogram",
        "measurement", "observation", "finding", "result", "outcome",
        "parameter", "variable", "coefficient", "constant", "value",
        "score", "metric", "index", "ratio", "rate",
        "table", "figure", "chart", "graph", "plot", "image",
        "publication", "paper", "article", "abstract", "citation",
    ],
    OntologyCategory.STRATEGY: [
        # General
        "goal", "plan", "strategy", "objective", "mission", "vision",
        "approach", "design", "framework", "paradigm", "concept",
        # Scientific strategy
        "hypothesis", "rationale", "aim", "purpose", "motivation",
        "application", "implication", "significance", "innovation",
        "advantage", "limitation", "challenge", "bottleneck", "trade-off",
        "improvement", "enhancement", "advancement", "breakthrough",
        "novel", "pioneering", "state-of-the-art", "next-generation",
        "future", "prospect", "potential", "promising", "emerging",
        "commercialization", "translation", "deployment", "scalability",
    ],
}

# Specificity detection — quantitative data patterns
import re as _re
_SPECIFICITY_PATTERNS = [
    _re.compile(r'\d+\.\d+'),                          # decimals: 3.14, 0.005
    _re.compile(r'\d+\s*[%‰]'),                        # percentages: 53%, 0.5%
    _re.compile(r'\d+\s*-?fold'),                      # fold changes: 4.8-fold
    _re.compile(r'\d+\s*×\s*10'),                      # scientific notation: 7.2×10⁻⁵
    _re.compile(r'\d+\s*(?:kDa|Da|bp|kb|nm|µ[MmLl]|mM|µg|mg|ng|ml|µl|nM)\b', _re.IGNORECASE),  # units
    _re.compile(r'\b(?:IC50|EC50|Kd|Km|Ki|Vmax|kcat|ΔG|ΔH|Tm)\s*[=:]?\s*\d', _re.IGNORECASE),  # bioscience constants
    _re.compile(r'\d+\s*(?:°C|K|°F)'),                 # temperatures
    _re.compile(r'\bp\s*[<>=]\s*0\.\d+'),               # p-values: p<0.05
    _re.compile(r'\b\d{4,}\b'),                        # large numbers (e.g. iteration counts)
    _re.compile(r'\d+/\d+'),                           # ratios/fractions: 3/5
]
_SPECIFICITY_KEYWORDS = [
    "mutation rate", "rate of", "yield", "efficiency", "conversion",
    "concentration", "activity", "half-life", "affinity", "selectivity",
    "turnover", "throughput", "accuracy", "precision", "resolution",
    "improvement", "increase", "decrease", "reduction", "enhancement",
]

# Per-axis keyword sets (used by heuristic evaluator)
AXIS_KEYWORDS = {
    "temporal": {
        "positive": ["will", "future", "plan", "upcoming", "next", "2025", "2026", "2027"],
        "negative": ["was", "legacy", "deprecated", "old", "previous", "2020", "2019"],
    },
    "relevance": {
        "positive": ["critical", "important", "key", "essential", "major", "significant"],
    },
    "risk": {
        "positive": ["error", "fail", "risk", "danger", "threat", "vulnerability", "attack", "breach"],
        "negative": ["secure", "stable", "protected", "safe", "reliable"],
    },
    "causality": {
        "positive": ["result", "therefore", "consequently", "leads to", "impact", "effect"],
        "negative": ["because", "due to", "caused", "reason", "source", "origin"],
    },
    "visibility": {
        "positive": ["public", "external", "customer", "announcement", "release", "open"],
        "negative": ["internal", "private", "confidential", "secret", "restricted"],
    },
    "trust": {
        "positive": ["verified", "confirmed", "official", "authentic", "reliable", "proven"],
        "negative": ["unverified", "rumor", "alleged", "uncertain", "suspicious", "unknown"],
    },
}


class SevenAxesEvaluator:
    """
    Evaluates text on 7+1 semantic axes (7 base + specificity boost).

    Modes (in priority order):
      1. GGUF Q4 mode: Uses TinyLlama GGUF (Q4_K_M) via llama-cpp-python.
      2. NF4 mode: Uses HuggingFace model with bitsandbytes NF4 quantization.
      3. Heuristic mode: Deterministic keyword-based scoring (no LLM needed).

    Calibration: If an AxisCalibration is provided (from AxisFineTuner),
    it is applied as a post-processing step.
    """

    DEFAULT_MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_llm: bool = True,
        use_nf4: bool = True,
        calibration: Optional[object] = None,  # AxisCalibration
    ):
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.model = None
        self.nf4_evaluator = None
        self.is_ready = False
        self.use_llm = use_llm
        self.calibration = calibration
        self.mode = "heuristic"

        # Priority 1: GGUF Q4_K_M (fastest, CPU-only)
        if use_llm and LLAMA_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=512,
                    n_gpu_layers=0,
                    verbose=False,
                )
                self.is_ready = True
                self.mode = "gguf_q4"
                print("SevenAxesEvaluator: TinyLlama GGUF Q4 loaded.")
            except Exception as e:
                print(f"SevenAxesEvaluator: GGUF load failed: {e}")

        # Priority 2: NF4 via bitsandbytes (requires CUDA)
        if not self.is_ready and use_nf4 and NF4_AVAILABLE:
            try:
                self.nf4_evaluator = NF4Evaluator()
                if self.nf4_evaluator.is_ready:
                    self.is_ready = True
                    self.mode = "nf4"
                    print("SevenAxesEvaluator: NF4 mode active.")
            except Exception as e:
                print(f"SevenAxesEvaluator: NF4 load failed: {e}")

        # Priority 3: Heuristic (always available)
        if not self.is_ready:
            print("SevenAxesEvaluator: Using heuristic mode (with specificity boost).")

    # ─── Public API ─────────────────────────────────────────────────────

    def evaluate(self, text: str) -> Tuple[Dict[str, float], OntologyCategory]:
        """
        Evaluate a single text.

        Returns:
            (axes_dict, ontology_category)
        """
        if self.is_ready and self.model:
            axes, category = self._llm_evaluate(text)
        else:
            axes, category = self._heuristic_evaluate(text)

        # Apply calibration if available
        if self.calibration is not None:
            axes = self.calibration.apply(axes)

        return axes, category

    def evaluate_batch(self, texts: List[str]) -> List[Tuple[Dict[str, float], OntologyCategory]]:
        """
        Evaluate multiple texts.

        Returns:
            List of (axes_dict, ontology_category) tuples.
        """
        return [self.evaluate(text) for text in texts]

    # ─── LLM Evaluation ────────────────────────────────────────────────

    def _llm_evaluate(self, text: str) -> Tuple[Dict[str, float], OntologyCategory]:
        prompt = """Analyze this text and return JSON scores:
- temporal: -1 (outdated) to +1 (future-relevant)
- relevance: 0 to 1 (importance)
- risk: -1 to +1 (threat level)
- causality: -1 (cause) to +1 (effect)
- visibility: 0 (internal) to 1 (public)
- trust: 0 to 1 (reliability)
- category: human/process/technology/information/strategy

Text: """ + text[:200] + """

JSON:"""

        try:
            output = self.model(prompt, max_tokens=150, temperature=0.1)
            response = output["choices"][0]["text"]
            return self._parse_llm_response(response, text)
        except Exception:
            return self._heuristic_evaluate(text)

    def _parse_llm_response(self, response: str, text: str) -> Tuple[Dict[str, float], OntologyCategory]:
        try:
            match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                axes = {
                    "temporal": self._clamp(data.get("temporal", 0), -1, 1),
                    "relevance": self._clamp(data.get("relevance", 0.5), 0, 1),
                    "risk": self._clamp(data.get("risk", 0), -1, 1),
                    "ontology": 0.0,
                    "causality": self._clamp(data.get("causality", 0), -1, 1),
                    "visibility": self._clamp(data.get("visibility", 0.5), 0, 1),
                    "trust": self._clamp(data.get("trust", 0.5), 0, 1),
                }
                cat_str = data.get("category", "unknown").lower()
                category = OntologyCategory.from_string(cat_str)
                axes["ontology"] = category.numeric
                return axes, category
        except Exception:
            pass
        return self._heuristic_evaluate(text)

    # ─── Deterministic Heuristic Evaluation ─────────────────────────────

    def _heuristic_evaluate(self, text: str) -> Tuple[Dict[str, float], OntologyCategory]:
        """
        Deterministic keyword-based evaluation. No randomness.
        Now includes specificity detection to boost data-rich chunks.
        """
        text_lower = text.lower()

        # A1: Temporal
        temporal = self._keyword_score(
            text_lower,
            AXIS_KEYWORDS["temporal"]["positive"],
            AXIS_KEYWORDS["temporal"].get("negative", []),
            divisor=5, center=0.0,
        )
        temporal = self._clamp(temporal, -1, 1)

        # A2: Relevance
        relevance_kw = AXIS_KEYWORDS["relevance"]["positive"]
        relevance = min(1.0, sum(w in text_lower for w in relevance_kw) / 3 + 0.3)

        # A3: Risk
        risk = self._keyword_score(
            text_lower,
            AXIS_KEYWORDS["risk"]["positive"],
            AXIS_KEYWORDS["risk"].get("negative", []),
            divisor=4, center=0.0,
        )
        risk = self._clamp(risk, -1, 1)

        # A4: Ontology
        category = self._detect_category(text_lower)

        # A5: Causality
        causality = self._keyword_score(
            text_lower,
            AXIS_KEYWORDS["causality"]["positive"],
            AXIS_KEYWORDS["causality"].get("negative", []),
            divisor=3, center=0.0,
        )
        causality = self._clamp(causality, -1, 1)

        # A6: Visibility
        visibility = self._keyword_score(
            text_lower,
            AXIS_KEYWORDS["visibility"]["positive"],
            AXIS_KEYWORDS["visibility"].get("negative", []),
            divisor=4, center=0.5,
        )
        visibility = self._clamp(visibility, 0, 1)

        # A7: Trust
        trust = self._keyword_score(
            text_lower,
            AXIS_KEYWORDS["trust"]["positive"],
            AXIS_KEYWORDS["trust"].get("negative", []),
            divisor=4, center=0.5,
        )
        trust = self._clamp(trust, 0, 1)

        # A8: Specificity (NEW — quantitative data detection)
        specificity = self._compute_specificity(text)

        axes = {
            "temporal": round(temporal, 3),
            "relevance": round(relevance, 3),
            "risk": round(risk, 3),
            "ontology": category.numeric,
            "causality": round(causality, 3),
            "visibility": round(visibility, 3),
            "trust": round(trust, 3),
            "specificity": round(specificity, 3),
        }

        return axes, category

    # ─── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _keyword_score(
        text: str,
        positive_kw: List[str],
        negative_kw: List[str],
        divisor: int = 4,
        center: float = 0.0,
    ) -> float:
        """Score based on positive – negative keyword hits."""
        pos = sum(1 for w in positive_kw if w in text)
        neg = sum(1 for w in negative_kw if w in text)
        return center + (pos - neg) / divisor

    @staticmethod
    def _detect_category(text: str) -> OntologyCategory:
        scores = {}
        for cat, keywords in CATEGORY_KEYWORDS.items():
            scores[cat] = sum(1 for k in keywords if k in text)
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return OntologyCategory.UNKNOWN

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        try:
            return max(min_val, min(max_val, float(value)))
        except (TypeError, ValueError):
            return (min_val + max_val) / 2.0

    @staticmethod
    def _compute_specificity(text: str) -> float:
        """
        Compute specificity score (0-1) based on quantitative data density.

        Detects: numbers, percentages, scientific units, fold-changes,
        p-values, bioscience constants (Km, Kd, IC50, etc.),
        and quantitative keywords.

        Returns:
            Float 0-1. Higher = more quantitative/specific data.
        """
        # Count regex pattern matches
        pattern_hits = 0
        for pattern in _SPECIFICITY_PATTERNS:
            pattern_hits += len(pattern.findall(text))

        # Count keyword matches
        text_lower = text.lower()
        keyword_hits = sum(1 for kw in _SPECIFICITY_KEYWORDS if kw in text_lower)

        # Combined score: saturates at ~1.0 with 6+ quantitative elements
        raw = (pattern_hits * 0.15) + (keyword_hits * 0.1)
        return min(1.0, raw)
