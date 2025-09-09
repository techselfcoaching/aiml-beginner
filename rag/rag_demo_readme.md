# RAG Employee Demo

This project demonstrates **Retrieval-Augmented Generation (RAG)** using employee records as private data. It includes two versions:

1. **In-Memory Version (`simple_rag_employee.py`)**
   - Stores embeddings in a Python dictionary.
   - Lightweight, no external dependencies beyond `sentence-transformers`.
   - Good for simple demonstrations.

2. **Weaviate Version (`rag_employee_weaviate.py`)**
   - Stores embeddings in a **Weaviate vector database**.
   - Demonstrates a production-like setup.
   - Requires running a Weaviate instance locally (via Docker).

Both versions load employee data from `employee_records.json` and compare **responses without RAG** (generic) vs **responses with RAG** (context-aware).

---

## Setup Instructions

### 1. Clone and Install Dependencies
```bash
pip install sentence-transformers numpy weaviate-client
```

### 2. In-Memory Version
Run the basic demo:
```bash
python simple_rag_employee.py
```

This will:
- Load employee data from `employee_records.json`
- Create embeddings in memory
- Show plain vs RAG answers for sample queries

### 3. Weaviate Version
#### Step 1: Run Weaviate Locally with Docker
```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  semitechnologies/weaviate:latest \
  --host 0.0.0.0 \
  --port 8080 \
  --scheme http
```

#### Step 2: Run the Weaviate Demo
```bash
python rag_employee_weaviate.py
```

This will:
- Connect to Weaviate at `http://localhost:8080`
- Create an `Employee` class schema
- Insert employee records with embeddings
- Perform semantic search with RAG-style responses

### 4. Compare Outputs
- **Without RAG**: Generic “contact HR” responses
- **With RAG**: Specific employee details retrieved from vector search

---

## Example Questions
- Who has Python and machine learning skills?
- Find someone with sales experience
- Who works in HR?
- Do we have any finance analysts?

---

## Benefits of RAG in This Demo
- **In-Memory**: Simple, easy to understand
- **Weaviate**: Production-ready, scalable, supports advanced search

---

## Cost & Resource Notes
- **In-Memory Version**: Free to run. Only uses your local CPU/RAM.
- **Weaviate Local (Docker)**: Free to run locally. Consumes disk space (~1–2 GB for Docker image) and memory while active.
- **No Cloud Costs**: These demos make no external API calls.
- **Potential Costs**: Only if you use **Weaviate Cloud (WCS)** or deploy on cloud providers (AWS, GCP, Azure). In that case, costs depend on provider pricing.

👉 For this demo, running locally is **completely free** beyond your machine’s resources.

---

## Notes
- Ensure Docker is installed and running if you want to use Weaviate.
- The schema in `rag_employee_weaviate.py` resets on each run (for demo clarity).
- Adapt schema and queries for more advanced real-world scenarios.