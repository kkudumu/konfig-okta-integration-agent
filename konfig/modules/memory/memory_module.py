"""
Memory Module for Konfig.

This module provides the persistence layer for the agent's state, knowledge,
and experiences. It implements three types of memory:
- Episodic Memory: Specific events and execution traces
- Semantic Memory: Factual knowledge from documentation
- Procedural Memory: Learned skills and recovery strategies
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import and_, desc, func, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from konfig.database.connection import get_async_session
from konfig.database.models import (
    ExecutionTrace,
    IntegrationJob,
    KnowledgeBaseVector,
    ProceduralMemoryPattern,
)
from konfig.utils.logging import LoggingMixin


class MemoryModule(LoggingMixin):
    """
    Memory Module implementing episodic, semantic, and procedural memory.
    
    This class provides the persistence layer for the agent's state, knowledge,
    and experiences, enabling state management, learning, and recovery.
    """
    
    def __init__(self):
        """Initialize the Memory Module."""
        super().__init__()
        self.setup_logging("memory")
        
        # Initialize embedding model for semantic memory
        self._embedding_model = None
        
        self.logger.info("Memory Module initialized")
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Get or create the sentence transformer model for embeddings."""
        if self._embedding_model is None:
            self.logger.info("Loading sentence transformer model")
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedding_model
    
    # ========== JOB STATE MANAGEMENT ==========
    
    async def create_job(
        self,
        job_id: uuid.UUID,
        input_url: str,
        vendor_name: Optional[str] = None,
        okta_domain: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IntegrationJob:
        """
        Create a new integration job record.
        
        Args:
            job_id: Unique identifier for the job
            input_url: URL to vendor documentation
            vendor_name: Name of the vendor application
            okta_domain: Okta domain for the integration
            initial_state: Initial agent state
            metadata: Additional job metadata
            
        Returns:
            Created IntegrationJob instance
        """
        async with get_async_session() as session:
            job = IntegrationJob(
                job_id=job_id,
                input_url=input_url,
                vendor_name=vendor_name,
                okta_domain=okta_domain,
                status="pending",
                state_snapshot=initial_state or {},
                job_metadata=metadata or {}
            )
            
            session.add(job)
            await session.commit()
            await session.refresh(job)
            
            self.logger.info("Job created", job_id=str(job_id), vendor=vendor_name)
            return job
    
    async def get_job(self, job_id: uuid.UUID) -> Optional[IntegrationJob]:
        """
        Get a job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            IntegrationJob instance or None
        """
        async with get_async_session() as session:
            result = await session.get(IntegrationJob, job_id)
            return result
    
    async def update_job_status(
        self,
        job_id: uuid.UUID,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update job status.
        
        Args:
            job_id: Job identifier
            status: New status
            error_message: Optional error message
            
        Returns:
            True if updated successfully
        """
        async with get_async_session() as session:
            job = await session.get(IntegrationJob, job_id)
            if job:
                job.status = status
                job.updated_at = datetime.utcnow()
                if error_message:
                    job.error_message = error_message
                await session.commit()
                
                self.logger.info("Job status updated", job_id=str(job_id), status=status)
                return True
            return False
    
    async def store_job_state(
        self,
        job_id: uuid.UUID,
        state: Dict[str, Any]
    ) -> bool:
        """
        Store/update the complete agent state for a job.
        
        Args:
            job_id: Job identifier
            state: Complete agent state dictionary
            
        Returns:
            True if stored successfully
        """
        async with get_async_session() as session:
            job = await session.get(IntegrationJob, job_id)
            if job:
                job.state_snapshot = state
                job.updated_at = datetime.utcnow()
                await session.commit()
                
                self.logger.debug("Job state stored", job_id=str(job_id), state_keys=list(state.keys()))
                return True
            return False
    
    async def load_job_state(self, job_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """
        Load the complete agent state for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Agent state dictionary or None
        """
        async with get_async_session() as session:
            job = await session.get(IntegrationJob, job_id)
            if job and job.state_snapshot:
                self.logger.debug("Job state loaded", job_id=str(job_id))
                return job.state_snapshot
            return None
    
    async def list_jobs(
        self,
        status: Optional[str] = None,
        vendor_name: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[IntegrationJob]:
        """
        List integration jobs with optional filtering.
        
        Args:
            status: Filter by status
            vendor_name: Filter by vendor name
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            
        Returns:
            List of IntegrationJob instances
        """
        async with get_async_session() as session:
            from sqlalchemy import select
            
            query = select(IntegrationJob)
            
            if status:
                query = query.where(IntegrationJob.status == status)
            if vendor_name:
                query = query.where(IntegrationJob.vendor_name == vendor_name)
            
            query = query.order_by(desc(IntegrationJob.created_at))
            query = query.limit(limit).offset(offset)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    # ========== EPISODIC MEMORY (EXECUTION TRACES) ==========
    
    async def store_trace(
        self,
        job_id: uuid.UUID,
        trace_type: str,
        content: Dict[str, Any],
        status: str = "success",
        parent_trace_id: Optional[uuid.UUID] = None,
        duration_ms: Optional[int] = None,
        error_details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """
        Store an execution trace.
        
        Args:
            job_id: Job identifier
            trace_type: Type of trace (thought, tool_call, observation, hitl_request)
            content: Trace content
            status: Trace status (success, failure)
            parent_trace_id: Parent trace for hierarchy
            duration_ms: Duration in milliseconds
            error_details: Error details if status is failure
            context: Additional context
            
        Returns:
            Created ExecutionTrace instance
        """
        async with get_async_session() as session:
            trace = ExecutionTrace(
                job_id=job_id,
                parent_trace_id=parent_trace_id,
                trace_type=trace_type,
                content=content,
                status=status,
                duration_ms=duration_ms,
                error_details=error_details,
                context=context or {}
            )
            
            session.add(trace)
            await session.commit()
            await session.refresh(trace)
            
            self.logger.debug(
                "Trace stored",
                job_id=str(job_id),
                trace_type=trace_type,
                status=status
            )
            return trace
    
    async def get_job_traces(
        self,
        job_id: uuid.UUID,
        trace_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ExecutionTrace]:
        """
        Get execution traces for a job.
        
        Args:
            job_id: Job identifier
            trace_type: Filter by trace type
            limit: Maximum number of traces
            
        Returns:
            List of ExecutionTrace instances
        """
        async with get_async_session() as session:
            from sqlalchemy import select
            
            query = select(ExecutionTrace).where(ExecutionTrace.job_id == job_id)
            
            if trace_type:
                query = query.where(ExecutionTrace.trace_type == trace_type)
            
            query = query.order_by(ExecutionTrace.timestamp).limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_trace_hierarchy(self, trace_id: uuid.UUID) -> List[ExecutionTrace]:
        """
        Get a trace and all its child traces.
        
        Args:
            trace_id: Root trace identifier
            
        Returns:
            List of ExecutionTrace instances in hierarchical order
        """
        async with get_async_session() as session:
            # Recursive CTE to get trace hierarchy
            cte = text("""
                WITH RECURSIVE trace_hierarchy AS (
                    SELECT * FROM execution_traces WHERE trace_id = :trace_id
                    UNION ALL
                    SELECT et.* FROM execution_traces et
                    JOIN trace_hierarchy th ON et.parent_trace_id = th.trace_id
                )
                SELECT * FROM trace_hierarchy ORDER BY timestamp
            """)
            
            result = await session.execute(cte, {"trace_id": trace_id})
            rows = result.fetchall()
            
            # Convert rows to ExecutionTrace objects
            traces = []
            for row in rows:
                trace = ExecutionTrace()
                for column in ExecutionTrace.__table__.columns:
                    setattr(trace, column.name, getattr(row, column.name))
                traces.append(trace)
            
            return traces
    
    # ========== SEMANTIC MEMORY (KNOWLEDGE BASE) ==========
    
    async def store_knowledge_chunks(
        self,
        job_id: uuid.UUID,
        chunks: List[Dict[str, Any]]
    ) -> List[KnowledgeBaseVector]:
        """
        Store knowledge chunks with embeddings.
        
        Args:
            job_id: Job identifier
            chunks: List of chunk dictionaries with text and metadata
            
        Returns:
            List of created KnowledgeBaseVector instances
        """
        # Generate embeddings for all chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        async with get_async_session() as session:
            vectors = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector = KnowledgeBaseVector(
                    job_id=job_id,
                    chunk_text=chunk["text"],
                    embedding=embedding.tolist(),  # Convert numpy array to list
                    chunk_metadata=chunk.get("metadata", {}),
                    content_type=chunk.get("content_type", "documentation"),
                    chunk_index=chunk.get("chunk_index", i),
                    relevance_score=chunk.get("relevance_score")
                )
                vectors.append(vector)
                session.add(vector)
            
            await session.commit()
            
            for vector in vectors:
                await session.refresh(vector)
            
            self.logger.info(
                "Knowledge chunks stored",
                job_id=str(job_id),
                num_chunks=len(chunks)
            )
            return vectors
    
    async def search_knowledge(
        self,
        job_id: uuid.UUID,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[KnowledgeBaseVector, float]]:
        """
        Search knowledge base using semantic similarity.
        
        Args:
            job_id: Job identifier
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (KnowledgeBaseVector, similarity_score) tuples
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query])[0]
        
        async with get_async_session() as session:
            # Use pgvector's L2 distance for similarity search
            similarity_query = text("""
                SELECT *, (1 - (embedding <-> :query_embedding)) as similarity
                FROM knowledge_base_vectors
                WHERE job_id = :job_id
                AND (1 - (embedding <-> :query_embedding)) >= :threshold
                ORDER BY embedding <-> :query_embedding
                LIMIT :limit
            """)
            
            result = await session.execute(similarity_query, {
                "query_embedding": query_embedding.tolist(),
                "job_id": job_id,
                "threshold": similarity_threshold,
                "limit": limit
            })
            
            results = []
            for row in result:
                # Create KnowledgeBaseVector object from row
                vector = KnowledgeBaseVector()
                for column in KnowledgeBaseVector.__table__.columns:
                    if column.name != "similarity":  # Skip our computed column
                        setattr(vector, column.name, getattr(row, column.name))
                
                similarity_score = row.similarity
                results.append((vector, similarity_score))
            
            self.logger.debug(
                "Knowledge search completed",
                job_id=str(job_id),
                query_length=len(query),
                num_results=len(results)
            )
            return results
    
    # ========== PROCEDURAL MEMORY (LEARNED PATTERNS) ==========
    
    async def store_pattern(
        self,
        vendor_domain: str,
        context_hash: str,
        successful_action: Dict[str, Any],
        failed_action: Optional[Dict[str, Any]] = None,
        pattern_type: str = "error_recovery",
        description: Optional[str] = None,
        confidence_score: float = 1.0,
        tags: Optional[List[str]] = None
    ) -> ProceduralMemoryPattern:
        """
        Store a learned pattern.
        
        Args:
            vendor_domain: Vendor domain this pattern applies to
            context_hash: Hash of the task context
            successful_action: The action that succeeded
            failed_action: The action that previously failed
            pattern_type: Type of pattern
            description: Human-readable description
            confidence_score: Initial confidence score
            tags: Tags for categorization
            
        Returns:
            Created ProceduralMemoryPattern instance
        """
        async with get_async_session() as session:
            pattern = ProceduralMemoryPattern(
                vendor_domain=vendor_domain,
                context_hash=context_hash,
                failed_action=failed_action,
                successful_action=successful_action,
                pattern_type=pattern_type,
                description=description,
                confidence_score=confidence_score,
                tags=tags or []
            )
            
            session.add(pattern)
            await session.commit()
            await session.refresh(pattern)
            
            self.logger.info(
                "Pattern stored",
                vendor_domain=vendor_domain,
                pattern_type=pattern_type,
                confidence=confidence_score
            )
            return pattern
    
    async def find_patterns(
        self,
        vendor_domain: str,
        context_hash: Optional[str] = None,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.5,
        limit: int = 10
    ) -> List[ProceduralMemoryPattern]:
        """
        Find matching patterns for a given context.
        
        Args:
            vendor_domain: Vendor domain to search
            context_hash: Specific context hash to match
            pattern_type: Type of pattern to find
            min_confidence: Minimum confidence score
            limit: Maximum number of patterns
            
        Returns:
            List of matching ProceduralMemoryPattern instances
        """
        async with get_async_session() as session:
            from sqlalchemy import select, and_
            
            query = select(ProceduralMemoryPattern).where(
                and_(
                    ProceduralMemoryPattern.vendor_domain == vendor_domain,
                    ProceduralMemoryPattern.confidence_score >= min_confidence
                )
            )
            
            if context_hash:
                query = query.where(ProceduralMemoryPattern.context_hash == context_hash)
            
            if pattern_type:
                query = query.where(ProceduralMemoryPattern.pattern_type == pattern_type)
            
            # Order by confidence score and recent usage
            query = query.order_by(
                desc(ProceduralMemoryPattern.confidence_score),
                desc(ProceduralMemoryPattern.last_used_at)
            ).limit(limit)
            
            result = await session.execute(query)
            patterns = result.scalars().all()
            
            self.logger.debug(
                "Patterns found",
                vendor_domain=vendor_domain,
                num_patterns=len(patterns)
            )
            return patterns
    
    async def update_pattern_usage(
        self,
        pattern_id: uuid.UUID,
        success: bool = True
    ) -> bool:
        """
        Update pattern usage statistics.
        
        Args:
            pattern_id: Pattern identifier
            success: Whether the pattern was used successfully
            
        Returns:
            True if updated successfully
        """
        async with get_async_session() as session:
            pattern = await session.get(ProceduralMemoryPattern, pattern_id)
            if pattern:
                pattern.usage_count += 1
                if success:
                    pattern.success_count += 1
                pattern.last_used_at = datetime.utcnow()
                
                # Update confidence score based on success rate
                success_rate = pattern.success_count / pattern.usage_count
                pattern.confidence_score = min(success_rate * 1.1, 1.0)  # Slight boost for successful patterns
                
                await session.commit()
                
                self.logger.debug(
                    "Pattern usage updated",
                    pattern_id=str(pattern_id),
                    success=success,
                    new_confidence=pattern.confidence_score
                )
                return True
            return False
    
    # ========== MEMORY ANALYSIS AND CLEANUP ==========
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        async with get_async_session() as session:
            stats_query = text("""
                SELECT 
                    (SELECT COUNT(*) FROM integration_jobs) as total_jobs,
                    (SELECT COUNT(*) FROM integration_jobs WHERE status = 'completed_success') as successful_jobs,
                    (SELECT COUNT(*) FROM execution_traces) as total_traces,
                    (SELECT COUNT(*) FROM knowledge_base_vectors) as total_knowledge_chunks,
                    (SELECT COUNT(*) FROM procedural_memory_patterns) as total_patterns,
                    (SELECT AVG(confidence_score) FROM procedural_memory_patterns) as avg_pattern_confidence,
                    (SELECT pg_size_pretty(pg_total_relation_size('integration_jobs'))) as jobs_table_size,
                    (SELECT pg_size_pretty(pg_total_relation_size('execution_traces'))) as traces_table_size,
                    (SELECT pg_size_pretty(pg_total_relation_size('knowledge_base_vectors'))) as knowledge_table_size
            """)
            
            result = await session.execute(stats_query)
            row = result.first()
            
            return {
                "total_jobs": row[0],
                "successful_jobs": row[1],
                "total_traces": row[2],
                "total_knowledge_chunks": row[3],
                "total_patterns": row[4],
                "avg_pattern_confidence": float(row[5]) if row[5] else 0.0,
                "table_sizes": {
                    "jobs": row[6],
                    "traces": row[7],
                    "knowledge": row[8]
                }
            }
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Clean up old data to manage storage.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_date = datetime.utcnow().replace(day=datetime.utcnow().day - days_to_keep)
        
        async with get_async_session() as session:
            from sqlalchemy import select, delete, and_
            
            # Delete old failed jobs and their associated data
            old_jobs_query = select(IntegrationJob).where(
                and_(
                    IntegrationJob.created_at < cutoff_date,
                    IntegrationJob.status == "completed_failure"
                )
            )
            
            old_jobs = await session.execute(old_jobs_query)
            deleted_jobs = len(old_jobs.scalars().all())
            
            # The foreign key cascades will handle related data
            delete_query = delete(IntegrationJob).where(
                and_(
                    IntegrationJob.created_at < cutoff_date,
                    IntegrationJob.status == "completed_failure"
                )
            )
            await session.execute(delete_query)
            
            # Clean up unused patterns with very low confidence
            delete_patterns_query = delete(ProceduralMemoryPattern).where(
                ProceduralMemoryPattern.confidence_score < 0.1
            )
            low_confidence_patterns = await session.execute(delete_patterns_query)
            
            await session.commit()
            
            cleanup_stats = {
                "deleted_jobs": deleted_jobs,
                "deleted_patterns": low_confidence_patterns
            }
            
            self.logger.info("Data cleanup completed", **cleanup_stats)
            return cleanup_stats