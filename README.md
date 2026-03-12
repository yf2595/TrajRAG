

# Repository Structure

trajrag-pipeline/

data/  
sample data illustrating the expected input format

methods/RAG.ipynb  
experiments and pipeline execution notebooks

flight_radar_api.ipynb  
load flights data

requirements.txt  
python dependencies

---

# Installation

Create a Python environment and install the dependencies:

pip install -r requirements.txt

---

# API Configuration

The pipeline requires API access to external services used to collect flight and meteorological data.

Create a `.env` file in the project root containing your credentials:

OPENAI_API_KEY=your_openai_key  
FLIGHTRADAR_API_KEY=your_flightradar_key  

The code will automatically load these variables using `python-dotenv`.

---

# Data

Due to licensing restrictions associated with the commercial APIs used to collect the data (e.g., Flightradar24 and Meteomatics), the full dataset used in the experiments cannot be publicly released.

Instead, this repository provides a **small sample dataset** illustrating the expected data format.

Researchers wishing to reproduce the full experiments should obtain their own API credentials and collect the data using the provided scripts.

---

# Running the Pipeline

The main experiments can be executed through the provided notebooks.

Typical workflow:

1. Load trajectory data.
2. Generate textual trajectory representations.
3. Retrieve similar flight segments using embeddings.
4. Generate the next aircraft state using the LLM.
5. Apply kinematic validation.

Each notebook reproduces a specific stage of the pipeline.
