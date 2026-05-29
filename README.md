# S7code - FAISS-Powered Agent with Document Indexing

## Overview

S7code is an AI agent system that uses **FAISS (Facebook AI Similarity Search)** for vector-based document retrieval. It can index large collections of documents (like 50 PDF files) and answer questions by semantically searching through the indexed content.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent Loop                            │
│  (agent7.py - orchestrates the 4-layer architecture)        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐         ┌──────────┐         ┌──────────┐
   │ Memory  │◄────────│Perception│────────►│ Decision │
   │ (FAISS) │         │          │         │          │
   └─────────┘         └──────────┘         └──────────┘
        │                                          │
        │                                          ▼
        │                                    ┌──────────┐
        └────────────────────────────────────│  Action  │
                                             │  (MCP)   │
                                             └──────────┘
```

### Core Components

1. **Memory** (`memory.py`) - Vector storage with FAISS indexing
2. **Perception** (`perception.py`) - Goal decomposition and orchestration
3. **Decision** (`decision.py`) - LLM-based tool selection
4. **Action** (`action.py`) - MCP tool execution
5. **Vector Index** (`vector_index.py`) - FAISS wrapper
6. **MCP Server** (`mcp_server.py`) - 11 tools including document indexing

## How 50 Files Are Indexed in FAISS

### Step 1: Document Discovery

```python
# Agent calls list_dir tool via MCP
list_dir("research_pprs/")
# Returns: 51 files (50 PDFs + 1 .DS_Store)
```

**Key Fix Applied**: `ARTIFACT_THRESHOLD_BYTES` was lowered from 4KB to 1KB in `action.py`, so the directory listing (6.5KB) becomes an **artifact** that Perception can attach to goals.

### Step 2: Chunking Strategy

Each PDF is processed through the `index_document` tool:

```python
# mcp_server.py - index_document function
def _chunk_text(text: str, size: int = 400, overlap: int = 80):
    """Sliding-window chunking by word count"""
    words = text.split()
    chunks = []
    stride = max(1, size - overlap)
    
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += stride
    return chunks
```

**Parameters**:
- **Chunk size**: 400 words per chunk
- **Overlap**: 80 words between consecutive chunks
- **Stride**: 320 words (400 - 80)

**Example**: A 10-page PDF (~5,000 words) creates approximately **16 chunks**.

### Step 3: Embedding Generation

For each chunk, the system:

1. Calls the **LLM Gateway** (`gateway.py`) embed endpoint
2. Uses the configured embedding model (e.g., `text-embedding-3-small`)
3. Generates a **1536-dimensional vector** (model-dependent)

```python
# memory.py - _try_embed function
def _try_embed(text: str, task_type: str) -> list[float] | None:
    resp = _gateway_embed(text, task_type="retrieval_document")
    return list(resp["embedding"])  # Returns 1536-dim vector
```

### Step 4: FAISS Index Storage

Each embedded chunk is added to the FAISS index:

```python
# vector_index.py - VectorIndex.add
def add(self, item_id: str, embedding: list[float]) -> None:
    vec = _l2_normalize(np.array(embedding, dtype=np.float32))
    
    if self._index is None:
        self._dim = vec.shape[0]  # e.g., 1536
        self._index = faiss.IndexFlatIP(self._dim)  # Inner Product index
    
    self._index.add(vec.reshape(1, -1))
    self._ids.append(item_id)  # Parallel ID tracking
```

**FAISS Index Type**: `IndexFlatIP` (Inner Product on L2-normalized vectors = Cosine Similarity)

### Step 5: Persistence

Two files are created in `state/`:

```
state/
├── index.faiss        # Binary FAISS index (vectors)
├── index_ids.json     # Parallel list of MemoryItem IDs
└── memory.json        # Full metadata for each chunk
```

**memory.json structure** for each chunk:
```json
{
  "id": "mem:abc123",
  "kind": "fact",
  "descriptor": "[sandbox:research_pprs/paper.pdf chunk 5/16] This paper presents...",
  "value": {
    "chunk": "full 400-word chunk text...",
    "chunk_index": 4,
    "total_chunks": 16,
    "source": "sandbox:research_pprs/paper.pdf"
  },
  "embedding": [0.123, -0.456, ...],  // 1536 floats
  "source": "sandbox:research_pprs/paper.pdf",
  "run_id": "index-20260529201830"
}
```

## How Queries Work: Reading from 50 Files

### Query Flow

```
User Query: "What is attention mechanism?"
     │
     ▼
┌─────────────────────────────────────────┐
│ 1. MEMORY READ (memory.py)              │
│    - Embed query → 1536-dim vector      │
│    - FAISS search → Top 8 similar chunks│
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 2. PERCEPTION (perception.py)           │
│    - Decompose into goals                │
│    - Attach relevant artifacts           │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 3. DECISION (decision.py)               │
│    - Sees top 8 chunks from all 50 PDFs │
│    - Decides: answer or search_knowledge│
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ 4. ACTION (action.py)                   │
│    - Execute search_knowledge if needed │
│    - Or produce final answer            │
└─────────────────────────────────────────┘
```

### Vector Search Implementation

```python
# memory.py - _vector_search
def _vector_search(query: str, *, kinds: list[str] | None, top_k: int = 8):
    # 1. Embed the query
    qvec = _try_embed(query, task_type="retrieval_query")
    
    # 2. Load FAISS index
    idx = _index()  # Loads from state/index.faiss
    
    # 3. Search for similar vectors
    hits = idx.search(qvec, k=top_k * 2)  # Get 16, filter to 8
    
    # 4. Map FAISS indices back to MemoryItems
    by_id = {item.id: item for item in _load()}
    out = []
    for item_id, score in hits:
        item = by_id.get(item_id)
        if kinds and item.kind not in kinds:
            continue
        out.append(item)
        if len(out) >= top_k:
            break
    
    return out  # Returns top 8 most relevant chunks
```

### FAISS Search Details

```python
# vector_index.py - VectorIndex.search
def search(self, query_embedding: list[float], k: int = 5):
    vec = _l2_normalize(np.array(query_embedding, dtype=np.float32))
    
    # FAISS inner product search (cosine similarity after normalization)
    scores, idxs = self._index.search(vec.reshape(1, -1), k)
    
    # Map integer indices back to string IDs
    out = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx >= 0:
            out.append((self._ids[idx], float(score)))
    
    return out  # [(item_id, similarity_score), ...]
```

## Example: Indexing 50 PDFs

### Command
```bash
uv run agent7.py "Index every .pdf file under research pprs/. Confirm how many chunks were indexed in total."
```

### What Happens

1. **Discovery** (Iteration 1)
   - Calls `list_dir("research_pprs/")`
   - Result: 51 files (stored as artifact due to 1KB threshold)
   - Perception creates 50 individual goals: "Make paper1.pdf searchable", "Make paper2.pdf searchable", ...

2. **Indexing** (Iterations 2-51)
   - For each PDF:
     - Calls `index_document("research_pprs/paper.pdf")`
     - Extracts text using PyMuPDF (`fitz`)
     - Chunks into ~400-word segments with 80-word overlap
     - Embeds each chunk via gateway
     - Adds to FAISS index
     - Persists to `state/`

3. **Final Report** (Iteration 52)
   - Counts total chunks indexed
   - Example output: "Indexed 50 PDFs with 847 total chunks"

### Storage Breakdown

For 50 PDFs (~10 pages each, ~5,000 words):
- **Chunks per PDF**: ~16 chunks
- **Total chunks**: 50 × 16 = **800 chunks**
- **FAISS index size**: 800 × 1536 × 4 bytes = **~4.9 MB**
- **memory.json size**: ~800 KB (metadata only)
- **Total storage**: **~5.7 MB**

## Example: Querying Indexed Documents

### Command
```bash
uv run agent7.py "What papers discuss attention mechanisms? Summarize the key findings."
```

### What Happens

1. **Memory Read** (Vector Search)
   ```python
   hits = memory.read("attention mechanisms", top_k=8)
   # Returns 8 most similar chunks from across all 50 PDFs
   # Example hits:
   # - [sandbox:research_pprs/1706.03762v7.pdf chunk 3/45] "Attention mechanisms allow..."
   # - [sandbox:research_pprs/2203.02155v1.pdf chunk 12/23] "Self-attention computes..."
   # - [sandbox:research_pprs/2310.12036v2.pdf chunk 7/19] "Multi-head attention..."
   ```

2. **Perception**
   - Goal 1: "Query knowledge base about attention mechanisms"
   - Goal 2: "Summarize findings from retrieved papers"

3. **Decision + Action**
   - Calls `search_knowledge("attention mechanisms", k=5)`
   - Gets 5 most relevant chunks with provenance
   - Synthesizes answer from chunk content

4. **Result**
   ```
   Based on indexed papers:
   
   1. "Attention Is All You Need" (1706.03762v7.pdf):
      - Introduced Transformer architecture
      - Self-attention mechanism replaces recurrence
   
   2. "Efficient Attention" (2203.02155v1.pdf):
      - Linear complexity attention variants
      - Reduces O(n²) to O(n)
   
   [... synthesized from 5 chunks across 3 different PDFs ...]
   ```

## Key Features

### 1. **Cross-Document Retrieval**
- Single query searches across all 50 PDFs simultaneously
- FAISS returns most relevant chunks regardless of source file
- Provenance tracked: each chunk knows its source PDF and position

### 2. **Semantic Search**
- Not keyword matching - understands meaning
- Query "neural networks" finds chunks about "deep learning", "transformers", "MLPs"
- Cosine similarity threshold ensures relevance

### 3. **Efficient Storage**
- Chunks stored once, searchable forever
- No need to re-read PDFs for subsequent queries
- FAISS index enables sub-millisecond search over 800+ chunks

### 4. **Fallback Mechanism**
```python
# memory.py - read function
def read(query: str, history: list[dict] | None = None, *, top_k: int = 8):
    # Try vector search first
    vec_hits = _vector_search(query, kinds=kinds, top_k=top_k)
    if vec_hits:
        return vec_hits
    
    # Fall back to keyword search if vector fails
    return _keyword_search(query, history, kinds=kinds, top_k=top_k)
```

## Performance Characteristics

### Indexing Time
- **Per PDF**: ~2-5 seconds (depends on size)
- **50 PDFs**: ~2-4 minutes total
- **Bottleneck**: Embedding API calls (1 call per chunk)

### Query Time
- **Vector search**: <10ms (FAISS is extremely fast)
- **Total query**: ~1-2 seconds (includes LLM synthesis)
- **Scales**: O(log n) with FAISS approximate search (not used in S7)

### Memory Usage
- **Runtime**: ~50 MB (FAISS index loaded in memory)
- **Disk**: ~6 MB for 50 PDFs (800 chunks)
- **Scales linearly**: 100 PDFs ≈ 12 MB

## Tools Available

### Document Indexing
- `index_document(path)` - Chunk and index a file
- `search_knowledge(query, k)` - Vector search over indexed facts

### File Operations
- `read_file(path)` - Read text file
- `list_dir(path)` - List directory contents
- `create_file(path, content)` - Create new file
- `update_file(path, content)` - Overwrite file
- `edit_file(path, find, replace)` - Find-and-replace

### Web & Utilities
- `web_search(query)` - Tavily/DuckDuckGo search
- `fetch_url(url)` - Crawl4ai markdown extraction
- `get_time(timezone)` - Current time
- `currency_convert(amount, from, to)` - Currency conversion

## Configuration

### Embedding Model
Set in `llm_gatewayV7/.env`:
```bash
EMBED_MODEL=text-embedding-3-small  # 1536 dimensions
# or
EMBED_MODEL=text-embedding-3-large  # 3072 dimensions
```

**Important**: Changing the model invalidates existing FAISS indices. Run `memory.clear()` first.

### Chunk Parameters
In `mcp_server.py`:
```python
@mcp.tool()
def index_document(path: str, chunk_size: int = 400, overlap: int = 80):
    # Adjust chunk_size and overlap as needed
    chunks = _chunk_text(text, size=chunk_size, overlap=overlap)
```

### Artifact Threshold
In `action.py`:
```python
ARTIFACT_THRESHOLD_BYTES = 1024  # 1KB - directory listings become artifacts
```

## Troubleshooting

### Only 4 Files Indexed Instead of 50

**Cause**: `ARTIFACT_THRESHOLD_BYTES` was too high (4KB), so directory listings didn't become artifacts.

**Fix**: Lower to 1KB in `action.py`:
```python
ARTIFACT_THRESHOLD_BYTES = 1024
```

### FAISS Index Dimension Mismatch

**Error**: `Embedding dim 3072 does not match index dim 1536`

**Cause**: Changed embedding model after building index.

**Fix**:
```python
from memory import clear
clear()  # Wipes memory.json and FAISS index
# Then re-index documents
```

### Out of Memory

**Cause**: Too many chunks loaded at once.

**Fix**: Reduce `top_k` in memory reads:
```python
hits = memory.read(query, top_k=5)  # Instead of 8
```

## File Structure

```
S7code/
├── agent7.py              # Main orchestrator
├── memory.py              # FAISS-backed memory service
├── vector_index.py        # FAISS wrapper
├── perception.py          # Goal decomposition
├── decision.py            # Tool selection
├── action.py              # MCP execution
├── mcp_server.py          # 11 MCP tools
├── gateway.py             # LLM gateway client
├── schemas.py             # Pydantic models
├── artifacts.py           # Binary blob storage
├── state/
│   ├── memory.json        # All memory items with embeddings
│   ├── index.faiss        # FAISS binary index
│   ├── index_ids.json     # ID mapping for FAISS
│   └── artifacts/         # Large tool results
└── sandbox/
    └── research_pprs/     # 50 PDF files
```

## Running the Agent

### Index Documents
```bash
uv run agent7.py "Index all PDFs in research_pprs/"
```

### Query Indexed Content
```bash
uv run agent7.py "What papers discuss transformers?"
uv run agent7.py "Summarize findings on attention mechanisms"
uv run agent7.py "Compare LoRA and full fine-tuning approaches"
```

### Clear and Re-index
```python
from memory import clear
clear()
# Then run indexing command again
```

## Technical Details

### Why FAISS?
- **Fast**: Billion-scale vector search in milliseconds
- **Memory-efficient**: Optimized C++ implementation
- **Flexible**: Supports exact and approximate search
- **Production-ready**: Used by Facebook, Spotify, etc.

### Why Cosine Similarity?
- **Scale-invariant**: Measures angle, not magnitude
- **Semantic**: Similar meanings → similar vectors → high cosine
- **Normalized**: Scores in [-1, 1], easy to threshold

### Why Chunking?
- **Context window limits**: LLMs can't process entire PDFs
- **Precision**: Smaller chunks = more precise retrieval
- **Overlap**: Ensures context isn't lost at boundaries

## Future Enhancements (Session 8+)

- **Hybrid retrieval**: Combine vector + keyword with RRF
- **Semantic chunking**: Use LLM to find natural boundaries
- **Re-ranking**: Two-stage retrieval for better precision
- **Metadata filtering**: Filter by date, author, document type
- **Approximate search**: FAISS IVF for billion-scale indices

---

**Version**: Session 7 (S7)  
**Last Updated**: 2026-05-29  
**Author**: EAGV3 Architecture
