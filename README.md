# Moolig.ai – Generative AI Chatbot for Ayurvedic Remedies

A Retrieval-Augmented Generation (RAG) system for Ayurvedic knowledge, combining traditional wisdom with modern AI capabilities to provide accurate, contextual responses to health and wellness queries.

## Overview

Moolig.ai is an advanced question-answering system that leverages multiple knowledge sources to deliver reliable Ayurvedic information. The system integrates vector databases, web search capabilities, and large language models to provide comprehensive answers grounded in classical Ayurvedic texts and contemporary research.

## Key Features

- **Multi-Source Knowledge Retrieval**: Combines Pinecone vector database, PDF document processing, and real-time web search
- **Context-Aware Responses**: Utilizes semantic search to retrieve relevant information from Ayurvedic texts
- **Response Validation**: Cross-validates generated answers against web sources for accuracy
- **PDF Knowledge Base**: Supports uploading and processing custom Ayurvedic PDF documents
- **Evaluation Framework**: Includes automated evaluation system to assess response quality against ground truth data

## Architecture

### Core Components

1. **Vector Database (Pinecone)**: Stores embeddings of Ayurvedic texts for efficient semantic search
2. **Embedding Model**: Uses `all-MiniLM-L6-v2` for text vectorization
3. **Language Model**: Leverages Meta's Llama 3.3 70B Instruct Turbo via Together AI
4. **Web Search Integration**: DuckDuckGo search API for supplementary information
5. **Document Processing**: PDF extraction and vectorization using FAISS

### System Workflow

1. Query processing and embedding generation
2. Parallel retrieval from Pinecone vector database and uploaded PDFs
3. Context validation and gap identification
4. Optional web search for additional information
5. Response generation with source attribution
6. Cross-validation against web results

## Technical Stack

- **Framework**: Streamlit
- **Vector Databases**: Pinecone, FAISS
- **Embeddings**: HuggingFace Sentence Transformers
- **LLM Provider**: Together AI
- **Document Processing**: PyPDF2, LangChain
- **Search Integration**: DuckDuckGo Search API
- **Evaluation**: LangChain with Groq API (DeepSeek-R1)

## Installation

```bash
pip install -r requirements.txt
```

### Required Dependencies

- streamlit
- together
- pinecone-client
- langchain-community
- sentence-transformers
- faiss-cpu
- pypdf2
- duckduckgo-search
- python-dotenv

## Configuration

The system requires API keys for:
- Together AI (LLM inference)
- Pinecone (vector database)
- Groq (evaluation framework)

Configure these in your environment or directly in the application files.

## Usage

### Basic Application

```bash
streamlit run app.py
```

### Advanced CRAG System

```bash
streamlit run newcrag.py
```

The advanced system includes:
- Context validation
- Automated web search triggering
- Response cross-validation
- Multi-source integration

## Evaluation

The system includes a comprehensive evaluation framework that:
- Compares generated responses against ground truth answers
- Assesses accuracy using LLM-based evaluation
- Provides detailed performance metrics

Run evaluation:

```bash
python eval.py
```

Current system accuracy: **88.67%** (133/150 correct responses)

## File Structure

- `app.py`: Basic Streamlit application
- `newcrag.py`: Advanced CRAG implementation with validation
- `crag.py`: Corrective RAG system with context validation
- `eval.py`: Automated evaluation framework
- `pine.py`: Pinecone integration utilities
- `runs.py`: Batch processing and evaluation scripts
- `requirements.txt`: Python dependencies

## Data Sources

The system draws from:
- Curated Ayurvedic text corpus stored in Pinecone
- Classical texts including Charaka Samhita, Sushruta Samhita, and Ashtanga Hridaya
- User-uploaded PDF documents
- Real-time web search results

## Performance

- Response time: 2-5 seconds per query
- Retrieval accuracy: 88.67%
- Context window: Up to 512 tokens
- Supported concurrent sessions: Multiple via Streamlit

## Limitations

- Response length limited to 512 tokens
- Web search restricted to DuckDuckGo results
- PDF processing limited to text-based documents
- Evaluation dependent on ground truth dataset quality

## Future Enhancements

- Integration with additional Ayurvedic text databases
- Multi-lingual support for Sanskrit and regional languages
- Enhanced citation and source tracking
- Real-time collaborative features
- Mobile application development

## Ethical Considerations

This system provides informational content based on Ayurvedic principles. It is not a substitute for professional medical advice, diagnosis, or treatment. Users should consult qualified healthcare practitioners for personalized medical guidance.

## License

This project is intended for educational and research purposes.

## Contact

For questions or collaboration inquiries, please refer to the repository issues section.
