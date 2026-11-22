# MemoRAG: Memory-Augmented Retrieval over a PDF

*Documentation for `MemoRAG.ipynb`*

## 1. Overview

This notebook implements a simplified version of **MemoRAG**, a retrieval-augmented generation (RAG) approach that adds a *global memory* step in front of normal retrieval.

In standard RAG, the user's question is embedded and used directly to search the document. This works poorly for vague or broad questions such as "Summarize the article". MemoRAG addresses this by first consulting a compact, LLM-built memory of the whole document. That memory is used to write better search queries, and those queries are then run against the original document chunks.

The demo document is a 33-page PDF, `Understanding_Climate_Change.pdf`. The notebook builds the memory store and the chunk index, then answers three test questions about it.

### Pipeline at a glance

1. **Index the document.** Load the PDF, split it into 1000-character chunks with 200 overlap, embed them, and store them in a FAISS index (the *document vector store*).
2. **Build the memory.** Send the full document text to an LLM, which returns topic/details pairs. Embed these pairs and store them in a second FAISS index (the *memory store*).
3. **Answer a question.**
   - Search the memory store for the 10 pairs closest to the question.
   - Ask the LLM to turn those pairs into *clues*: text spans and surrogate questions.
   - Run each clue (plus the original question) against the document vector store, taking the top 3 chunks per clue.
   - Remove duplicate chunks.
   - Ask the LLM to answer using only those chunks.

The code comments map these steps to the MemoRAG paper's notation: `y = Θ_mem(q, D)` for memory-based clue generation, `c = Γ(y, D)` for retrieval, and `y = Θ(q, c)` for generation.

## 2. Technology stack

| Component | Choice |
| --- | --- |
| LLM (extraction, clue generation, answering) | Cohere `command-a-03-2025` via `cohere.ClientV2` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `langchain_huggingface.HuggingFaceEmbeddings` |
| Vector store | FAISS via `langchain_community.vectorstores` |
| PDF loading | `PyPDFLoader` (`langchain_community`) |
| Chunking | `RecursiveCharacterTextSplitter` (`langchain_text_splitters`) |
| Output validation | `pydantic` models |
| Config | `python-dotenv` (reads `COHERE_API_KEY` from `.env`) |

## 3. Setup

### Requirements

Install the dependencies (package names inferred from the notebook's imports):

```bash
pip install cohere python-dotenv pydantic faiss-cpu \
    langchain-community langchain-huggingface langchain-text-splitters \
    sentence-transformers pypdf
```

Cell 1 also imports `pandas` and `langchain_groq`, but neither is used. Both can be removed (see Known issues).

### Environment

Create a `.env` file next to the notebook:

```
COHERE_API_KEY=your_key_here
```

### Data layout

```
project/
├── MemoRAG.ipynb
├── .env
└── data/
    └── Understanding_Climate_Change.pdf     # input document
```

The memory store is written to `data/Understanding_Climate_Change_Memory_Store.faiss` on first run. Despite the extension, FAISS's `save_local` creates a *directory* at that path.

### Running

Run the cells **top to bottom**. Later cells depend on names defined earlier: the prompts and Pydantic models from Cell 1, the `co` client from Cell 2, and the `embeddings` object from Cell 3.

## 4. Notebook structure

### Cell 1: Schemas and prompts

Defines the data models and prompt templates used everywhere else.

**Pydantic models**

- `KeyValuePair`: one memory entry with `topic: str` and `details: str`.
- `MemoryResponse`: wraps `pairs: List[KeyValuePair]`. It is used to validate the LLM's JSON output.

**Prompts**

| Variable | Purpose |
| --- | --- |
| `memory_store_system_prompt` | System prompt for memory creation: extract key-value pairs without redundancy. |
| `memory_store_kv_cache_prompt` | User prompt template. It takes `{document}` and demands a strict JSON object `{"pairs": [{"topic", "details"}, ...]}`. It describes the task as mimicking KV-cache memory creation. |
| `system_prompt_for_queries_creation` | System prompt for the clue-generation calls. |
| `memorag_span_prompt_for_queries_creation` | Takes `{question}` and `{relevant_info}` and asks for **text-span clues**, one per line. |
| `memorag_sur_prompt_for_queries_creation` | Takes the same inputs and asks for **surrogate questions**, one per line. |

### Cell 2: Cohere connectivity test

Loads `.env`, creates the global Cohere client `co`, and sends a test prompt ("Explain RAG in simple terms."). If this prints an explanation, your API key and model access work. Every later cell uses `co`.

### Cell 3: The `MemoryStore` class

The memory module. It also instantiates the global `embeddings` model, which is reused by the rest of the notebook.

**State**

- `store`: the FAISS index of memories, or `None` until the first call to `memorize`.
- `processed_count`: number of documents processed.
- `json_parse_failures`: count of times the fallback text parser was needed. A warning prints every 5 failures.

**Methods**

| Method | Description |
| --- | --- |
| `_get_response_text(response)` | Returns `response.message.content[0].text` from a Cohere v2 response. Raises a descriptive `ValueError` if the shape is unexpected. |
| `memorize(document)` | Calls the LLM in JSON mode to extract topic/details pairs, parses them, formats each as `"Topic: …\nDetails: …"`, and adds them to FAISS. The first call creates the index. Later calls skip any pair whose topic is already stored. |
| `_parse_into_pairs(content)` | Parses the JSON and validates it with `MemoryResponse`. On failure it sets a fallback flag and calls the text parser. |
| `_parse_text_into_pairs(text)` | Fallback parser. It reads lines beginning with `Topic:` and `Details:` into `(topic, details)` tuples. |
| `create_retrieval_queries(query)` | The core MemoRAG step; see below. Returns `None` if the store is empty. |
| `save_store(path)` | Persists the index with `FAISS.save_local`. |
| `load_store(path)` | Loads it with `FAISS.load_local(..., allow_dangerous_deserialization=True)`. |

#### How `create_retrieval_queries` works

1. Run `similarity_search_with_score` over the memory store for the top `k = min(10, ntotal)` entries.
2. Format the hits into a `relevant_info` list.
3. **Call 1 (spans):** send the span prompt and receive text-span clues.
4. **Call 2 (surrogates):** send the surrogate prompt at `temperature=0.5` and receive clue questions.
5. Split both responses on newlines and keep only lines longer than 3 words. This drops fragments and blanks.
6. Append the original query so it is always searched.
7. Return the combined list.

### Cell 4: Retrieval and answer pipeline

| Function | Description |
| --- | --- |
| `encode_pdf(path, chunk_size=1000, chunk_overlap=200)` | Loads the PDF, splits it, embeds the chunks, and returns a FAISS vector store. Prints page and chunk counts. |
| `retrieve_context(retrieval_query, vectorstore, k=3)` | Returns the text of the top-`k` chunks for one query. |
| `generate_answer(query, contexts, temperature=0)` | Builds a grounded prompt ("Based ONLY on the provided context…") and returns the Cohere answer. If `contexts` is empty it returns a fixed "information not available" message without calling the LLM. |
| `process_query(query, memory_store, vectorstore)` | Runs the full flow: generate clues, retrieve for each clue, deduplicate with `dict.fromkeys` (order-preserving), generate the answer, and return `(unique_contexts, final_answer)`. If no clues are generated it falls back to searching with the raw query. |

The lower half of the cell is the driver script:

1. Build the document vector store from the PDF.
2. Create an empty `MemoryStore`.
3. If the memory path does not exist, read the whole PDF into one string, call `memorize`, and call `save_store`.
4. Call `load_store` to load the memory from disk.
5. Run three test questions:
   - "What are the impacts of climate change on biodiversity?"
   - "Please summarize the climate change article"
   - "Describe the social and economic influence of climate change."

## 5. Example run

The saved output shows the following:

- 33 PDF pages were split into 97 chunks.
- The memory build ran as a single LLM call and yielded 42 topic/details entries, such as *Climate Change Definition*, *Historical Context*, and *Modern Observations*.
- For the biodiversity question, the system generated 8 retrieval queries. These included clues taken from memory (extreme weather, forests as carbon sinks) and surrogate questions such as "How do greenhouse gases impact biodiversity and ecosystem health?". Together they retrieved 24 chunks, which reduced to 11 unique chunks before answering.

## 6. Caching and reuse

The memory store is built only if the path does not already exist, which avoids repeating the most expensive LLM call. To rebuild it, for example after changing the prompts or the PDF, **delete the saved memory directory**. The document vector store is rebuilt on every run because `encode_pdf` is called unconditionally.

To use a different document, change `path` and `memory_store_path`. Always give each document its own memory path, otherwise you will load the wrong memory.

## 7. Tunable parameters

| Parameter | Location | Default | Effect |
| --- | --- | --- | --- |
| `chunk_size` / `chunk_overlap` | `encode_pdf` call | 1000 / 200 | Size and overlap of document chunks. |
| `k` (memory) | `create_retrieval_queries` | 10 | Number of memory entries used to write clues. |
| `k` (documents) | `retrieve_context` | 3 | Chunks fetched per clue. |
| Clue length filter | `create_retrieval_queries` | more than 3 words | Drops short or noisy clue lines. |
| `temperature` | surrogate call / `generate_answer` | 0.5 / 0 | Diversity of clues versus determinism of the answer. |
| Model | all `co.chat` calls | `command-a-03-2025` | Hardcoded in several places. |

## 8. Known issues and suggested improvements

These are observations from reading the code and its saved output.

1. **The whole document is memorized in one call.** The driver joins all pages into a single string and sends it in one request, so the log reports only "chunk 1". The prompt text says the article is supplied "chunk by chunk", but the code does not do this. Long documents may exceed the model's context window or be summarized shallowly. Splitting the text into large segments and calling `memorize` per segment would match the intended design.
2. **Raw memory entries leak into queries.** The retrieval queries in the example output include lines like `'Topic: Extreme Weather Events'` and `'Details: …'`. The span prompt receives `relevant_info` as a Python list, so the model sometimes echoes its `Topic:` and `Details:` labels. Join the entries with newlines instead, and strip the `Topic:` and `Details:` prefixes from clues.
3. **Duplicate detection is topic-only and inefficient.** `memorize` fetches every stored document with `similarity_search("", k=ntotal)` and compares topic strings. Two different topics with similar content both survive, and one topic name reused for different content is dropped. Tracking a set of topics in the class, or comparing embeddings, would be cheaper and more accurate.
4. **Unsafe deserialization flag.** `allow_dangerous_deserialization=True` is required by FAISS (it uses pickle) but is unsafe if the saved index can be tampered with. Load only files you created yourself.
5. **Stale imports.** `ChatGroq`, `pandas`, and `sys` are unused. `load_dotenv` is imported in Cell 1 but only called in Cell 2.
6. **Hardcoded model name and no error handling around API calls.** Rate limits and network errors will raise exceptions mid-pipeline. Move the model name into a constant and add retries.
7. **Fallback parser is basic.** It only handles `Topic:` and `Details:` lines. Markdown-formatted or bulleted output will produce no pairs.
8. **Naming.** `memorag_span_prompt_for_queries_creation` and the `kv_cache` prompt wording describe KV-cache memory, but this implementation uses a text-based vector memory rather than a real KV cache as in the original paper.
9. **No evaluation.** The notebook prints answers but does not score them. Comparing against plain RAG on the same questions would show what the memory step adds.

## 9. Glossary

- **RAG**: Retrieval-Augmented Generation. The LLM answers using passages fetched from an external source.
- **MemoRAG**: a RAG variant that uses a global memory of the document to generate retrieval clues.
- **Memory store**: here, a FAISS index of LLM-extracted topic/details pairs.
- **Document vector store**: a FAISS index of raw PDF chunks.
- **Clue**: a generated text span or surrogate question used as a search query.
- **Surrogate query**: a rephrased or decomposed question that makes retrieval more precise.
