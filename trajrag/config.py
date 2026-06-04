"""
Canonical hyperparameters for TrajRAG (ICDM ADS 2026, paper Sec. IV-G).

Controlled-benchmark eval only (Table I and related). Import in notebooks:

    from trajrag.config import LLM_MODEL, EMBED_MODEL, K_NEIGHBORS, ALPHA
"""

# LLM / embeddings
LLM_MODEL = "gpt-5.1"
EMBED_MODEL = "text-embedding-3-large"
SUMMARY_TEMPERATURE = 0.3
GENERATION_TEMPERATURE = 0.0

# Retrieval
K_NEIGHBORS = 5

# Kinematic ring validator (paper Eq. 1)
ALPHA = 1.2
DELTA_T_S = 120

# Data splits (Paris–Rome primary evaluation)
TRAIN_FLIGHTS = 831
TEST_FLIGHTS = 100
CHRONOLOGICAL_TRAIN_FRAC = 0.83
