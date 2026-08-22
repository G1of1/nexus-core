from nexus.config import NexusSettings
import asyncio
from nexus.models.document import Document
from nexus.pipeline.ingestion import IngestionPipeline
from nexus.pipeline.rag import RAGEngine, RAGResponse
from nexus.processing.loaders import DocumentLoader, LoadedDocument
from nexus.providers.embeddings.gemini import GeminiEmbeddingProvider
from nexus.providers.embeddings.mock import MockEmbeddingProvider
from nexus.providers.llm.gemini import GeminiLLMProvider
from nexus.providers.llm.mock import MockLLMProvider
from nexus.providers.vectorstore.memory import InMemoryVectorStore
from nexus.providers.vectorstore.qdrant import QdrantVectorStore
settings = NexusSettings()
engine = RAGEngine(vector_store=QdrantVectorStore(settings=settings), embedding_provider=GeminiEmbeddingProvider(settings=settings), llm_provider=GeminiLLMProvider(settings=settings))


def load_document(path: str) -> LoadedDocument:
    loader = DocumentLoader()
    return loader.load(path)


async def ingest(text: str, document_id: str, metadata: dict[str, str]):
    await engine.ingestion.ingest_text(text=text, document_id=document_id, metadata=metadata)

async def query(question: str):
    response = await engine.query(question=question)
    for metadata in response:
        for data in metadata:
            print(f"{data}\n")
    
    return response


async def test_nexus_engine():
    doc = load_document("tests/data/github-pp.pdf")
    await ingest(doc.text, doc.content_hash, doc.metadata)
    await query("What does GitHub do with my data?")
    #print(response.answer)


async def main():
    await test_nexus_engine()

if __name__ == "__main__":
    asyncio.run(main())
