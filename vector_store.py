
from typing import List, Dict, Any
from neo4j import GraphDatabase


class VectorStore:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "graphragdb",
    ):
        self.database = database
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
        )
        self._setup_schema()

    def _setup_schema(self):
        with self.driver.session(database=self.database) as session:
            session.run("""
                CREATE CONSTRAINT chunk_id IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.id IS UNIQUE
            """)

            session.run("""
                CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 3072,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)

    def store_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ):
        """Store document chunks and their embeddings in Neo4j."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        params = []

        for chunk, embedding in zip(chunks, embeddings):
            if len(embedding) != 3072:
                raise ValueError(
                    f"Expected 3072 embedding dimensions, "
                    f"received {len(embedding)}."
                )

            params.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "embedding": embedding,
                "metadata": str(chunk.get("metadata", {})),
            })

        with self.driver.session(database=self.database) as session:
            session.run("""
                UNWIND $params AS p
                MERGE (c:Chunk {id: p.id})
                SET c.text = p.text,
                    c.source = p.source,
                    c.chunk_index = p.chunk_index,
                    c.embedding = p.embedding,
                    c.metadata = p.metadata
            """, {"params": params}).consume()

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find the most semantically similar chunks using the vector index."""

        if len(query_embedding) != 3072:
            raise ValueError(
                f"Expected 3072 query dimensions, "
                f"received {len(query_embedding)}."
            )

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Chunk)
                SEARCH c IN (
                    VECTOR INDEX chunk_embedding
                    FOR $qe
                    LIMIT $top_k
                ) SCORE AS similarity
                RETURN c.id AS id,
                       c.text AS text,
                       c.source AS source,
                       c.chunk_index AS chunk_index,
                       similarity
                ORDER BY similarity DESC
            """, {
                "qe": query_embedding,
                "top_k": top_k,
            })

            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "source": record["source"],
                    "chunk_index": record["chunk_index"],
                    "similarity": float(record["similarity"]),
                }
                for record in result
            ]

    def get_all_sources(self) -> List[str]:
        """Return the distinct document sources stored in Neo4j."""

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.source IS NOT NULL
                RETURN DISTINCT c.source AS source
                ORDER BY source
            """)

            return [record["source"] for record in result]

    def get_chunk_count(self) -> int:
        """Return the number of document chunks in Neo4j."""

        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (c:Chunk)
                RETURN count(c) AS count
            """)

            return result.single()["count"]

    def delete_by_source(self, source: str):
        """Delete all chunks belonging to a particular source."""

        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (c:Chunk {source: $source})
                DELETE c
            """, {"source": source}).consume()

    def close(self):
        """Close the Neo4j driver."""

        self.driver.close()
