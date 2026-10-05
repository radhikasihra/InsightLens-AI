# Multimodal PDF RAG System

An AI-powered **Multimodal Retrieval-Augmented Generation (RAG)** system designed to understand and retrieve information from both **text and images inside PDF documents**.

The project combines **CLIP multimodal embeddings, FAISS vector search, LangChain, PyMuPDF, and OpenAI GPT-4.1 integration** to build a document intelligence pipeline capable of retrieving semantically relevant textual and visual information.

---

## Project Overview

Traditional RAG systems primarily work with text. Real-world documents, however, often contain important information in **charts, figures, diagrams, and images**.

This project creates a multimodal document-processing and retrieval pipeline that:

- Extracts textual and visual content from PDFs.
- Splits document text into contextual chunks.
- Generates embeddings for both text and images using **CLIP**.
- Stores multimodal representations in a **FAISS vector database**.
- Converts natural-language queries into CLIP embeddings.
- Performs semantic similarity search across document content.
- Retrieves relevant text and images together.
- Constructs multimodal context that can be supplied to an LLM.

---

## System Architecture

```text
                         PDF Document
                              │
                              ▼
                       ┌─────────────┐
                       │   PyMuPDF   │
                       └──────┬──────┘
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
             Text Extraction        Image Extraction
                  │                       │
                  ▼                       │
             Text Chunking               │
                  │                       │
                  └───────────┬───────────┘
                              ▼
                   CLIP Multimodal Model
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Text Embeddings      Image Embeddings
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       FAISS Vector Store
                              │
                         User Query
                              │
                              ▼
                    CLIP Query Embedding
                              │
                              ▼
                    Similarity Retrieval
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
             Relevant Text          Relevant Images
                  │                       │
                  └───────────┬───────────┘
                              ▼
                    Multimodal Context
                              │
                              ▼
                     Answer Pipeline
```

---

## Key Features

### Multimodal Document Understanding
Processes both textual and visual information contained within PDF documents instead of limiting retrieval to text.

### CLIP-Based Multimodal Embeddings
Uses **`openai/clip-vit-base-patch32`** to generate representations for text and images within a compatible embedding space.

### Semantic Retrieval
User queries are embedded with CLIP and matched against indexed document content based on semantic similarity.

### FAISS Vector Search
Uses **FAISS** for efficient storage and retrieval of multimodal embeddings.

### Intelligent Text Chunking
Uses LangChain's `RecursiveCharacterTextSplitter` with overlapping chunks to preserve contextual information.

### PDF Image Extraction
Extracts embedded visual content from PDF pages using **PyMuPDF**.

### Multimodal Context Construction
Combines retrieved text and Base64-encoded images into a multimodal message structure suitable for downstream multimodal LLM processing.

### Metadata-Aware Retrieval
Maintains metadata including page numbers, content type, and image identifiers to preserve document context.

---

## Tech Stack

| Technology | Application |
|---|---|
| **Python** | Core development |
| **LangChain** | Document processing and multimodal message construction |
| **OpenAI GPT-4.1** | Multimodal LLM integration initialized in the project |
| **CLIP** | Multimodal text and image embeddings |
| **Hugging Face Transformers** | CLIP model and processor |
| **FAISS** | Vector storage and similarity retrieval |
| **PyMuPDF** | PDF text and image extraction |
| **PyTorch** | Model inference |
| **Pillow** | Image processing |
| **NumPy** | Embedding manipulation |
| **python-dotenv** | Environment configuration |

---

## Multimodal RAG Workflow

```text
PDF
 │
 ├──► Text ──► Chunking ──► CLIP Text Embeddings ──┐
 │                                                  │
 └──► Images ─────────────► CLIP Image Embeddings ─┤
                                                    │
                                                    ▼
                                             FAISS Vector DB
                                                    │
User Question ──► CLIP Query Embedding ────────────┤
                                                    │
                                                    ▼
                                         Similarity Retrieval
                                                    │
                                         ┌──────────┴──────────┐
                                         ▼                     ▼
                                  Retrieved Text        Retrieved Images
                                         │                     │
                                         └──────────┬──────────┘
                                                    ▼
                                          Multimodal Context
```

---

## Core Components

### PDF Processing

The document-processing layer uses **PyMuPDF** to extract both textual and visual information from PDF pages.

Text is converted into LangChain document objects while maintaining page-level metadata.

---

### Text Chunking

Extracted text is divided into smaller overlapping sections using:

- **Chunk size:** 500
- **Chunk overlap:** 100

This helps retain contextual continuity between neighboring chunks.

---

### Multimodal Embedding Generation

The project uses:

**`openai/clip-vit-base-patch32`**

CLIP creates normalized vector representations for:

- PDF text chunks
- PDF images
- User queries

This enables cross-modal retrieval within a shared representation space.

---

### Vector Database

The generated multimodal embeddings are indexed using **FAISS**.

FAISS enables efficient similarity search across the embedded document content and returns the most relevant results for a query.

---

### Multimodal Retrieval

The retrieval component:

1. Converts a user query into a CLIP text embedding.
2. Searches the FAISS vector store.
3. Retrieves the most semantically relevant document elements.
4. Identifies whether each result contains text or visual information.

The notebook currently retrieves the **top 5** relevant results.

---

### Multimodal Context Construction

Retrieved information is separated into:

- Relevant text
- Relevant images

The project constructs a LangChain `HumanMessage` containing the original question, retrieved textual context, page information, and relevant images encoded for multimodal processing.

---

## Core Functions

### `embed_text()`

Creates a normalized CLIP embedding for textual information.

### `embed_image()`

Creates a normalized CLIP embedding for visual information.

### `retrieve_multimodal()`

Performs similarity retrieval from the FAISS vector store using the embedded user query.

### `create_multimodal_message()`

Combines retrieved text and images into a multimodal LangChain message.

### `multimodal_pdf_rag_pipeline()`

Coordinates query retrieval and the project's current answer-generation demonstration.

---

## Current Implementation

The project initializes an **OpenAI GPT-4.1** chat model and implements multimodal message construction for retrieved text and images.

The current notebook's final demonstration pipeline uses predefined responses for selected example query categories and retrieved text for other questions.

The implemented architecture therefore demonstrates the complete:

**PDF Processing → Multimodal Embedding → Vector Indexing → Semantic Retrieval → Multimodal Context Construction**

pipeline and provides a foundation for connecting retrieved multimodal context directly to the initialized LLM for fully generated responses.

---

## Project Highlights

- Built an end-to-end **Multimodal RAG architecture** for PDF document intelligence.
- Implemented joint semantic retrieval across **text and visual content**.
- Generated multimodal vector representations using **CLIP**.
- Built efficient similarity retrieval using **FAISS**.
- Preserved page-level metadata across document processing.
- Implemented automatic extraction of embedded PDF images.
- Constructed multimodal prompts containing retrieved text and images.
- Integrated components from **LangChain, Hugging Face, OpenAI, and PyTorch**.

---

## Applications

The architecture can serve as a foundation for:

- Financial report analysis
- Research-paper assistants
- Business document intelligence
- Chart and figure retrieval
- Technical document Q&A
- Educational document assistants
- Multimodal enterprise knowledge systems
- PDF-based AI research assistants

---

## Future Enhancements

- Directly connect retrieved multimodal messages to GPT-4.1 for generated responses.
- Add grounded source and page citations.
- Support multiple PDF documents.
- Add persistent vector indexes.
- Implement hybrid retrieval and reranking.
- Add retrieval and response evaluation.
- Support conversational follow-up questions.
- Build an interactive multimodal document interface.

---

## Skills Demonstrated

`Python` • `Multimodal AI` • `Retrieval-Augmented Generation (RAG)` • `CLIP` • `FAISS` • `LangChain` • `OpenAI` • `Transformers` • `PyTorch` • `Vector Databases` • `Semantic Search` • `PDF Processing` • `Multimodal Retrieval`

---

## Author

**Radhika Sihra**

Data Science • Machine Learning • Deep Learning • Generative AI
