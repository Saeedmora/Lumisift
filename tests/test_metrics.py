"""
Professional Benchmark Suite for Logical Rooms MVP
===================================================
Tests Token Reduction, Context Accuracy, Self-Optimization, and Efficiency
across multiple scenarios as defined in the mHC architecture spec.
"""

import sys
import os
import time
import random
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.atom import Atom
from core.models import LogicalRoom
from core.embeddings import EmbeddingService
from core.llm_interface import AxisEvaluator
from core.vector_store import VectorStore
from core.graph import GraphManager

# ============================================================================
# TEST DATA
# ============================================================================

# Domain 1: Cybersecurity
SECURITY_TEXTS = [
    "The security protocol failed due to a buffer overflow error in the legacy authentication module.",
    "Critical vulnerability CVE-2024-1234 allows remote code execution through crafted HTTP headers.",
    "Firewall logs indicate unusual traffic patterns originating from IP range 192.168.1.0/24.",
    "Two-factor authentication bypass discovered in password reset flow; patch deployed.",
    "SQL injection vulnerability in user search endpoint allows extraction of hashed passwords.",
    "Malware sample analyzed: trojan horse with C2 communication on port 443.",
    "Intrusion detection system flagged 1500 failed login attempts in the last hour.",
    "Zero-day exploit targeting unpatched Windows SMB service discovered in the wild.",
]

# Domain 2: Financial Reports
FINANCIAL_TEXTS = [
    "Quarterly revenue increased by 23% year-over-year, driven by cloud services growth.",
    "Operating expenses reduced by 8% through automation initiatives in Q3 2025.",
    "Net profit margin improved to 15.2%, exceeding analyst expectations of 13.8%.",
    "Cash reserves now stand at $4.2 billion, providing strong liquidity position.",
    "Customer acquisition cost decreased to $45 per user from $62 last quarter.",
    "Annual recurring revenue reached $850 million milestone in December 2025.",
    "Gross margin expanded 200 basis points due to improved supply chain efficiency.",
    "Stock buyback program completed: 2 million shares repurchased at average $145.",
]

# Domain 3: Technical Documentation
TECHNICAL_TEXTS = [
    "The API endpoint /v2/users accepts POST requests with JSON payload containing user metadata.",
    "Database migration script converts legacy VARCHAR fields to standardized UUID format.",
    "Microservice architecture employs event-driven communication via Apache Kafka topics.",
    "Load balancer distributes traffic across 12 application servers using round-robin algorithm.",
    "Redis cache layer reduces database query latency from 250ms to 15ms average.",
    "Container orchestration managed by Kubernetes with automatic horizontal pod scaling.",
    "CI/CD pipeline includes unit tests, integration tests, and security scanning stages.",
    "GraphQL schema defines 45 types with 200+ fields for comprehensive data access layer.",
]

# Domain 4: News / Current Events
NEWS_TEXTS = [
    "Climate summit concludes with 150 nations committing to net-zero emissions by 2050.",
    "Central bank raises interest rates by 25 basis points amid persistent inflation concerns.",
    "Tech giant announces layoffs affecting 10,000 employees across global operations.",
    "New space telescope captures unprecedented images of distant galaxy formation.",
    "Electric vehicle sales surpass gasoline cars for first time in European market.",
    "Artificial intelligence regulation bill passes Senate with bipartisan support.",
    "Major earthquake measuring 7.2 strikes coastal region; tsunami warning issued.",
    "Breakthrough in fusion energy achieves net positive output for first time.",
]

# Long Document Simulation (concatenated paragraphs)
LONG_DOCUMENT = """
The comprehensive annual review of enterprise security posture reveals significant 
improvements across all measured dimensions. Network segmentation initiatives have 
reduced lateral movement risk by 67% compared to baseline measurements from 2024.

Endpoint detection and response (EDR) coverage now extends to 98.5% of corporate 
devices, with mean time to detection improving from 4.2 hours to 23 minutes. 
Security awareness training completion rates reached 94% across all departments.

Vulnerability management metrics show 89% of critical vulnerabilities patched within 
7 days of disclosure, up from 62% in the previous year. The security operations 
center processed 2.3 million alerts, with false positive rates reduced to 12%.

Third-party risk assessments completed for all 450 vendors with access to sensitive 
data. Zero significant breaches attributed to supply chain compromise during the 
review period. Cloud security posture management tools deployed across AWS, Azure, 
and GCP environments.

Identity and access management modernization project completed on schedule. 
Privileged access management solution now covers 100% of administrative accounts. 
Password-less authentication rolled out to 15,000 users with positive adoption metrics.

Incident response capabilities validated through four tabletop exercises and one 
full-scale simulation. Mean time to containment improved to 2.1 hours. Forensic 
investigation capacity expanded with additional three certified examiners.

Regulatory compliance maintained across SOC 2 Type II, ISO 27001, GDPR, and 
HIPAA frameworks. Zero material findings in external audits. Internal audit 
completed 48 control testing procedures with 97% pass rate.

Budget utilization for security initiatives reached 94% of allocated funds. 
Return on security investment calculated at 340% based on prevented breach costs. 
Strategic roadmap for 2026 includes AI-powered threat detection and zero trust 
architecture expansion.
"""

# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def calculate_token_reduction(raw_texts: list, rooms: list) -> dict:
    """Calculate token reduction metrics."""
    raw_word_count = sum(len(t.split()) for t in raw_texts)
    
    compressed_word_count = 0
    for r in rooms:
        compressed_word_count += len(r.name.split())
        compressed_word_count += len(str(r.axes_summary).split())
    
    reduction = 1.0 - (compressed_word_count / raw_word_count) if raw_word_count > 0 else 0.0
    
    return {
        "raw_tokens": raw_word_count,
        "compressed_tokens": compressed_word_count,
        "reduction_pct": max(0.0, reduction) * 100
    }

def run_scenario(name: str, texts: list, embedder: EmbeddingService, evaluator: AxisEvaluator) -> dict:
    """Run a single benchmark scenario."""
    start_time = time.time()
    
    rooms = []
    vector_store = VectorStore()
    graph_manager = GraphManager()
    
    # Process texts into rooms (cluster every 2 for compression demo)
    import numpy as np
    
    for i in range(0, len(texts), 2):
        chunk = texts[i:min(i+2, len(texts))]
        room = LogicalRoom(name=f"{name}-Room-{i//2}")
        
        for text in chunk:
            embedding = embedder.embed(text)
            axes = evaluator.evaluate(text)
            obj = Atom(text=text, embedding=embedding, axes=axes)
            room.add_object(obj)
        
        rooms.append(room)
        
        # Index room centroid
        if room.centroid is not None:
            vector_store.add(np.array([room.centroid]), [room.id])
        
        graph_manager.add_room(room)
    
    # Build relationships
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if rooms[i].centroid is not None and rooms[j].centroid is not None:
                sim = np.dot(rooms[i].centroid, rooms[j].centroid) / (
                    np.linalg.norm(rooms[i].centroid) * np.linalg.norm(rooms[j].centroid) + 1e-9
                )
                if sim > 0.3:
                    graph_manager.add_relationship(rooms[i].id, rooms[j].id, weight=float(sim))
    
    end_time = time.time()
    
    # Calculate metrics
    token_metrics = calculate_token_reduction(texts, rooms)
    
    return {
        "scenario": name,
        "input_texts": len(texts),
        "rooms_created": len(rooms),
        "edges_created": len(graph_manager.get_graph_data().edges()),
        "processing_time_ms": (end_time - start_time) * 1000,
        "token_reduction_pct": token_metrics["reduction_pct"],
        "raw_tokens": token_metrics["raw_tokens"],
        "compressed_tokens": token_metrics["compressed_tokens"]
    }

def run_self_optimization_test(embedder: EmbeddingService, evaluator: AxisEvaluator) -> dict:
    """Test self-optimization capability."""
    # Create initial room
    room = LogicalRoom(name="Self-Opt-Test")
    
    initial_axes = {"risk": 0.3, "time": 0.2, "relevance": 0.5}
    room.axes_summary = initial_axes.copy()
    
    # Simulate new high-risk data arriving
    new_texts = [
        "CRITICAL: System breach detected, immediate action required.",
        "URGENT: Data exfiltration in progress, containment needed.",
    ]
    
    for text in new_texts:
        embedding = embedder.embed(text)
        axes = evaluator.evaluate(text)
        obj = Atom(text=text, embedding=embedding, axes=axes)
        room.add_object(obj)
    
    # Check if axes adapted
    risk_change = room.axes_summary.get("risk", 0) - initial_axes["risk"]
    time_change = room.axes_summary.get("time", 0) - initial_axes["time"]
    
    adaptation_detected = risk_change > 0.1 or time_change > 0.1
    
    return {
        "initial_risk": initial_axes["risk"],
        "final_risk": room.axes_summary.get("risk", 0),
        "risk_delta": risk_change,
        "adaptation_success": adaptation_detected
    }

def run_search_accuracy_test(embedder: EmbeddingService, evaluator: AxisEvaluator) -> dict:
    """Test search accuracy using known relevant queries."""
    # Create rooms from security texts
    rooms = []
    vector_store = VectorStore()
    import numpy as np
    
    for i, text in enumerate(SECURITY_TEXTS):
        room = LogicalRoom(name=f"Sec-{i}")
        embedding = embedder.embed(text)
        axes = evaluator.evaluate(text)
        obj = Atom(text=text, embedding=embedding, axes=axes)
        room.add_object(obj)
        rooms.append(room)
        vector_store.add(np.array([room.centroid]), [room.id])
    
    # Query with known relevant term
    query = "buffer overflow vulnerability exploit"
    query_vec = embedder.embed(query)
    
    _, results = vector_store.search(np.array([query_vec]), k=3)
    
    # Check if relevant rooms are in top results
    # Rooms 0, 1, 4, 5, 7 contain security-related terms
    relevant_ids = {rooms[i].id for i in [0, 1, 4, 5, 7]}
    hits = sum(1 for r in results if r in relevant_ids)
    
    precision = hits / len(results) if results else 0
    
    return {
        "query": query,
        "top_k": 3,
        "relevant_in_top_k": hits,
        "precision": precision
    }

# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_full_benchmark():
    """Execute the complete professional benchmark suite."""
    print("\n" + "="*70)
    print("  LOGICAL ROOMS MVP - PROFESSIONAL BENCHMARK SUITE")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70 + "\n")
    
    # Initialize services
    print("Initializing services...")
    embedder = EmbeddingService()
    evaluator = AxisEvaluator()
    print()
    
    results = []
    
    # -------------------------------------------------------------------------
    # SCENARIO 1: Short Queries (Single Domain)
    # -------------------------------------------------------------------------
    print("Running Scenario 1: Short Queries (Security Domain)...")
    r1 = run_scenario("Security", SECURITY_TEXTS, embedder, evaluator)
    results.append(r1)
    
    # -------------------------------------------------------------------------
    # SCENARIO 2: Financial Domain
    # -------------------------------------------------------------------------
    print("Running Scenario 2: Financial Reports...")
    r2 = run_scenario("Financial", FINANCIAL_TEXTS, embedder, evaluator)
    results.append(r2)
    
    # -------------------------------------------------------------------------
    # SCENARIO 3: Technical Documentation
    # -------------------------------------------------------------------------
    print("Running Scenario 3: Technical Documentation...")
    r3 = run_scenario("Technical", TECHNICAL_TEXTS, embedder, evaluator)
    results.append(r3)
    
    # -------------------------------------------------------------------------
    # SCENARIO 4: News/Events
    # -------------------------------------------------------------------------
    print("Running Scenario 4: News & Current Events...")
    r4 = run_scenario("News", NEWS_TEXTS, embedder, evaluator)
    results.append(r4)
    
    # -------------------------------------------------------------------------
    # SCENARIO 5: Long Document Analysis
    # -------------------------------------------------------------------------
    print("Running Scenario 5: Long Document Analysis...")
    long_doc_sentences = [s.strip() for s in LONG_DOCUMENT.split('.') if s.strip()]
    r5 = run_scenario("LongDoc", long_doc_sentences, embedder, evaluator)
    results.append(r5)
    
    # -------------------------------------------------------------------------
    # SCENARIO 6: Multi-Domain Combined
    # -------------------------------------------------------------------------
    print("Running Scenario 6: Multi-Domain Combined...")
    combined = SECURITY_TEXTS + FINANCIAL_TEXTS + TECHNICAL_TEXTS + NEWS_TEXTS
    r6 = run_scenario("MultiDomain", combined, embedder, evaluator)
    results.append(r6)
    
    # -------------------------------------------------------------------------
    # SELF-OPTIMIZATION TEST
    # -------------------------------------------------------------------------
    print("Running Self-Optimization Test...")
    opt_result = run_self_optimization_test(embedder, evaluator)
    
    # -------------------------------------------------------------------------
    # SEARCH ACCURACY TEST
    # -------------------------------------------------------------------------
    print("Running Search Accuracy Test...")
    search_result = run_search_accuracy_test(embedder, evaluator)
    
    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    
    print("\n" + "="*70)
    print("  BENCHMARK RESULTS")
    print("="*70 + "\n")
    
    # Table Header
    print(f"{'Scenario':<15} {'Inputs':>7} {'Rooms':>6} {'Edges':>6} {'Time(ms)':>10} {'Reduction':>10}")
    print("-"*70)
    
    total_reduction = 0
    for r in results:
        print(f"{r['scenario']:<15} {r['input_texts']:>7} {r['rooms_created']:>6} {r['edges_created']:>6} {r['processing_time_ms']:>10.2f} {r['token_reduction_pct']:>9.1f}%")
        total_reduction += r['token_reduction_pct']
    
    avg_reduction = total_reduction / len(results)
    
    print("-"*70)
    print(f"{'AVERAGE':<15} {'':<7} {'':<6} {'':<6} {'':<10} {avg_reduction:>9.1f}%")
    print()
    
    # Self-Optimization Results
    print("Self-Optimization Test:")
    print(f"  Initial Risk Score: {opt_result['initial_risk']:.2f}")
    print(f"  Final Risk Score:   {opt_result['final_risk']:.2f}")
    print(f"  Risk Delta:         {opt_result['risk_delta']:.2f}")
    print(f"  Adaptation:         {'SUCCESS' if opt_result['adaptation_success'] else 'NEEDS TUNING'}")
    print()
    
    # Search Accuracy Results
    print("Search Accuracy Test:")
    print(f"  Query:              '{search_result['query']}'")
    print(f"  Top-K Retrieved:    {search_result['top_k']}")
    print(f"  Relevant Hits:      {search_result['relevant_in_top_k']}")
    print(f"  Precision@K:        {search_result['precision']*100:.1f}%")
    print()
    
    # Summary
    print("="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"  Average Token Reduction:     {avg_reduction:.1f}%")
    print(f"  Token Reduction Target:      >= 50%")
    print(f"  Status:                      {'PASS' if avg_reduction >= 50 else 'FAIL'}")
    print()
    print(f"  Self-Optimization:           {'PASS' if opt_result['adaptation_success'] else 'FAIL'}")
    print(f"  Search Precision:            {search_result['precision']*100:.1f}%")
    print("="*70 + "\n")
    
    # =========================================================================
    # GENERATE VISUALIZATION WITH FULL DATASET
    # =========================================================================
    print("Generating Full Graph Visualization...")
    
    import numpy as np
    
    # Build combined graph with all domains
    all_texts = SECURITY_TEXTS + FINANCIAL_TEXTS + TECHNICAL_TEXTS + NEWS_TEXTS
    long_doc_sentences = [s.strip() for s in LONG_DOCUMENT.split('.') if s.strip()]
    all_texts += long_doc_sentences
    
    all_rooms = []
    full_graph = GraphManager()
    
    domain_labels = (
        ["Security"] * len(SECURITY_TEXTS) +
        ["Financial"] * len(FINANCIAL_TEXTS) +
        ["Technical"] * len(TECHNICAL_TEXTS) +
        ["News"] * len(NEWS_TEXTS) +
        ["LongDoc"] * len(long_doc_sentences)
    )
    
    # Create one room per text for detailed visualization
    for i, text in enumerate(all_texts):
        domain = domain_labels[i]
        room = LogicalRoom(name=f"{domain}-{i}")
        embedding = embedder.embed(text)
        axes = evaluator.evaluate(text)
        obj = Atom(text=text, embedding=embedding, axes=axes)
        room.add_object(obj)
        all_rooms.append(room)
        full_graph.add_room(room)
    
    # Build all edges
    edge_count = 0
    for i in range(len(all_rooms)):
        for j in range(i + 1, len(all_rooms)):
            if all_rooms[i].centroid is not None and all_rooms[j].centroid is not None:
                sim = np.dot(all_rooms[i].centroid, all_rooms[j].centroid) / (
                    np.linalg.norm(all_rooms[i].centroid) * np.linalg.norm(all_rooms[j].centroid) + 1e-9
                )
                if sim > 0.35:  # Slightly higher threshold for cleaner graph
                    full_graph.add_relationship(all_rooms[i].id, all_rooms[j].id, weight=float(sim))
                    edge_count += 1
    
    print(f"  Total Nodes: {len(all_rooms)}")
    print(f"  Total Edges: {edge_count}")
    
    # Generate HTML
    from visualization.simple_visualizer import SimpleVisualizer
    viz = SimpleVisualizer(full_graph)
    viz.generate_html("logical_rooms_graph.html")
    
    print("\n>>> Open 'logical_rooms_graph.html' to view the full visualization <<<\n")
    
    return results

if __name__ == "__main__":
    run_full_benchmark()

