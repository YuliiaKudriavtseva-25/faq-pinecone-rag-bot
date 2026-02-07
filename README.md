# FAQ Bot (RAG) — Pinecone + OpenAI

A simple FAQ assistant that answers user questions using a vector knowledge base (Pinecone) and an LLM response generator (OpenAI).

Pipeline: question → embedding → Pinecone search → context → LLM response.

## Features
- Multi-turn CLI chat (exit/quit supported)
- Retrieval with TOP_K and similarity threshold
- Context building from multiple relevant FAQ matches
- LLM fallback model strategy
- Error handling and safe "I don't know" behavior

## Setup
1) Install dependencies:
```bash
pip install -r requirements.txt

