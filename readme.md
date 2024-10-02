
# MindVault
Vector search database layer built on top of `sqlite` with python + RAG search and LLM interactions. A "light-weight" fully **local** vector database for personal files.

Features:
- Includes automatic change tracking and file system indexing
- Quick fts5 trigram search
- Semantic cosine similarity search across vectors or other custom bi-encoder
- Supports cross-encoder reranking
- Simple RAG toolchain to summarize database retreival results
- Containerized deployment environment with docker
- REST API interface to interact with database layer


TODO:
- [] Complete docker container setup
- [] Complete rag chain integration
- [] Refine user facing API 