"""
Database models for Konfig.

This module defines the SQLAlchemy models for all database tables as specified
in the PRD Section 5.1.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class IntegrationJob(Base):
    """
    Master record for all integration tasks.
    
    This table serves as the master record for all integration tasks. It tracks
    the high-level status, inputs, and final state of each job, enabling users
    and the system to manage the queue and handle paused states.
    """
    
    __tablename__ = "integration_jobs"
    
    # Primary key
    job_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
        doc="Unique identifier for the integration job"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        doc="When the job was created"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        onupdate=datetime.utcnow,
        doc="When the job was last updated"
    )
    
    # Status tracking
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        doc="Current status of the integration job"
    )
    
    # Input data
    input_url = Column(
        Text,
        nullable=False,
        doc="URL to the vendor's SAML setup documentation"
    )
    
    vendor_name = Column(
        String(255),
        nullable=True,
        doc="Name of the vendor application being integrated"
    )
    
    # Okta integration details
    okta_domain = Column(
        String(255),
        nullable=True,
        doc="Okta domain for this integration"
    )
    
    okta_app_id = Column(
        String(255),
        nullable=True,
        doc="Okta application ID once created"
    )
    
    # State management
    state_snapshot = Column(
        JSONB,
        nullable=True,
        doc="Full agent state at time of pause or completion"
    )
    
    # Error handling
    error_message = Column(
        Text,
        nullable=True,
        doc="Error message if the job failed"
    )
    
    # Additional metadata
    job_metadata = Column(
        JSONB,
        nullable=True,
        default=dict,
        server_default="{}",
        doc="Additional metadata for the job"
    )
    
    # Relationships
    execution_traces = relationship(
        "ExecutionTrace",
        back_populates="job",
        cascade="all, delete-orphan",
        doc="All execution traces for this job"
    )
    
    knowledge_base_vectors = relationship(
        "KnowledgeBaseVector",
        back_populates="job",
        cascade="all, delete-orphan",
        doc="Knowledge base vectors for this job"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            status.in_([
                'pending',
                'running', 
                'paused_for_hitl',
                'completed_success',
                'completed_failure'
            ]),
            name='valid_status'
        ),
        Index('idx_integration_jobs_status', 'status'),
        Index('idx_integration_jobs_created_at', 'created_at'),
        Index('idx_integration_jobs_vendor', 'vendor_name'),
    )
    
    def __repr__(self) -> str:
        return f"<IntegrationJob(job_id={self.job_id}, status={self.status}, vendor={self.vendor_name})>"


class ExecutionTrace(Base):
    """
    Detailed execution traces for observability and learning.
    
    This table stores a detailed, step-by-step log of the agent's cognitive
    process, modeled after distributed tracing systems like OpenTelemetry.
    """
    
    __tablename__ = "execution_traces"
    
    # Primary key
    trace_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
        doc="Unique identifier for this trace"
    )
    
    # Foreign key to job
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("integration_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
        doc="Job this trace belongs to"
    )
    
    # Self-referencing foreign key for trace hierarchy
    parent_trace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("execution_traces.trace_id"),
        nullable=True,
        doc="Parent trace ID for building trace hierarchy"
    )
    
    # Timing information
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        doc="When this trace was created"
    )
    
    duration_ms = Column(
        BigInteger,
        nullable=True,
        doc="Duration of the operation in milliseconds"
    )
    
    # Trace classification
    trace_type = Column(
        String(20),
        nullable=False,
        doc="Type of trace (thought, tool_call, observation, hitl_request)"
    )
    
    # Trace content
    content = Column(
        JSONB,
        nullable=False,
        doc="The trace content (thought text, tool call details, observation result)"
    )
    
    # Status tracking
    status = Column(
        String(10),
        nullable=False,
        default="success",
        doc="Status of the trace (success, failure)"
    )
    
    error_details = Column(
        Text,
        nullable=True,
        doc="Detailed error information if status is failure"
    )
    
    # Additional context
    context = Column(
        JSONB,
        nullable=True,
        default=dict,
        server_default="{}",
        doc="Additional context information"
    )
    
    # Relationships
    job = relationship(
        "IntegrationJob",
        back_populates="execution_traces",
        doc="The job this trace belongs to"
    )
    
    children = relationship(
        "ExecutionTrace",
        backref="parent",
        doc="Child traces"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            trace_type.in_([
                'thought',
                'tool_call',
                'observation',
                'hitl_request'
            ]),
            name='valid_trace_type'
        ),
        CheckConstraint(
            status.in_(['success', 'failure']),
            name='valid_trace_status'
        ),
        Index('idx_traces_job_id', 'job_id'),
        Index('idx_traces_timestamp', 'timestamp'),
        Index('idx_traces_type', 'trace_type'),
        Index('idx_traces_status', 'status'),
    )
    
    def __repr__(self) -> str:
        return f"<ExecutionTrace(trace_id={self.trace_id}, type={self.trace_type}, status={self.status})>"


class ProceduralMemoryPattern(Base):
    """
    Learned heuristics and successful recovery strategies.
    
    This table implements the agent's procedural memory, storing learned
    heuristics and successful recovery strategies that allow the agent to
    become more efficient and resilient over time.
    """
    
    __tablename__ = "procedural_memory_patterns"
    
    # Primary key
    pattern_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
        doc="Unique identifier for this pattern"
    )
    
    # Pattern identification
    vendor_domain = Column(
        String(255),
        nullable=False,
        doc="Vendor domain this pattern applies to"
    )
    
    context_hash = Column(
        String(64),
        nullable=False,
        doc="Hash of the task context (e.g., 'login_form', 'saml_config_page')"
    )
    
    # Pattern details
    failed_action = Column(
        JSONB,
        nullable=True,
        doc="The tool call that previously failed"
    )
    
    successful_action = Column(
        JSONB,
        nullable=False,
        doc="The corrected tool call that succeeded"
    )
    
    # Pattern metadata
    pattern_type = Column(
        String(50),
        nullable=False,
        default="error_recovery",
        doc="Type of pattern (error_recovery, optimization, etc.)"
    )
    
    description = Column(
        Text,
        nullable=True,
        doc="Human-readable description of the pattern"
    )
    
    # Confidence and usage tracking
    confidence_score = Column(
        Float,
        nullable=False,
        default=1.0,
        doc="Confidence score for this pattern (0.0 to 1.0)"
    )
    
    usage_count = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of times this pattern has been used"
    )
    
    success_count = Column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of times this pattern was successful"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        doc="When this pattern was created"
    )
    
    last_used_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="When this pattern was last used"
    )
    
    # Additional context
    tags = Column(
        JSONB,
        nullable=True,
        default=list,
        server_default="[]",
        doc="Tags for categorizing patterns"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint(
            confidence_score >= 0.0,
            name='confidence_score_min'
        ),
        CheckConstraint(
            confidence_score <= 1.0,
            name='confidence_score_max'
        ),
        CheckConstraint(
            usage_count >= 0,
            name='usage_count_min'
        ),
        CheckConstraint(
            success_count >= 0,
            name='success_count_min'
        ),
        CheckConstraint(
            success_count <= usage_count,
            name='success_count_max'
        ),
        Index('idx_procedural_memory_domain_context', 'vendor_domain', 'context_hash'),
        Index('idx_procedural_memory_confidence', 'confidence_score'),
        Index('idx_procedural_memory_usage', 'last_used_at'),
        Index('idx_procedural_memory_type', 'pattern_type'),
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of this pattern."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    def __repr__(self) -> str:
        return f"<ProceduralMemoryPattern(pattern_id={self.pattern_id}, domain={self.vendor_domain}, confidence={self.confidence_score})>"


class KnowledgeBaseVector(Base):
    """
    Vector embeddings for semantic memory and RAG.
    
    This table implements the agent's semantic memory by storing and indexing
    vector embeddings of source documentation, enabling efficient semantic
    searches for relevant instructions.
    """
    
    __tablename__ = "knowledge_base_vectors"
    
    # Primary key
    chunk_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
        doc="Unique identifier for this knowledge chunk"
    )
    
    # Foreign key to job
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("integration_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
        doc="Job this knowledge chunk belongs to"
    )
    
    # Content
    chunk_text = Column(
        Text,
        nullable=False,
        doc="The text content of this chunk"
    )
    
    # Vector embedding (768 dimensions for sentence-transformers)
    embedding = Column(
        Vector(768),
        nullable=False,
        doc="Vector embedding of the chunk text"
    )
    
    # Metadata
    chunk_metadata = Column(
        JSONB,
        nullable=True,
        default=dict,
        server_default="{}",
        doc="Metadata about the chunk (source URL, section headers, etc.)"
    )
    
    # Content classification
    content_type = Column(
        String(50),
        nullable=False,
        default="documentation",
        doc="Type of content (documentation, configuration, troubleshooting, etc.)"
    )
    
    chunk_index = Column(
        Integer,
        nullable=False,
        doc="Index of this chunk within the source document"
    )
    
    # Quality metrics
    relevance_score = Column(
        Float,
        nullable=True,
        doc="Relevance score for this chunk (0.0 to 1.0)"
    )
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now(),
        doc="When this chunk was created"
    )
    
    # Relationships
    job = relationship(
        "IntegrationJob",
        back_populates="knowledge_base_vectors",
        doc="The job this knowledge chunk belongs to"
    )
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint(
            relevance_score >= 0.0,
            name='relevance_score_min'
        ),
        CheckConstraint(
            relevance_score <= 1.0,
            name='relevance_score_max'
        ),
        CheckConstraint(
            chunk_index >= 0,
            name='chunk_index_min'
        ),
        Index('idx_knowledge_vectors_job_id', 'job_id'),
        Index('idx_knowledge_vectors_type', 'content_type'),
        Index('idx_knowledge_vectors_relevance', 'relevance_score'),
        # HNSW index for fast approximate nearest neighbor search
        Index(
            'idx_knowledge_vectors_embedding_hnsw',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        ),
        # IVFFlat index as an alternative
        Index(
            'idx_knowledge_vectors_embedding_ivfflat', 
            'embedding',
            postgresql_using='ivfflat',
            postgresql_with={'lists': 100},
            postgresql_ops={'embedding': 'vector_l2_ops'}
        ),
    )
    
    def __repr__(self) -> str:
        return f"<KnowledgeBaseVector(chunk_id={self.chunk_id}, job_id={self.job_id}, type={self.content_type})>"


# Additional helper models for system management

class SystemHealth(Base):
    """System health monitoring."""
    
    __tablename__ = "system_health"
    
    check_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )
    
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now()
    )
    
    component = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)
    details = Column(JSONB, nullable=True)
    
    __table_args__ = (
        Index('idx_system_health_timestamp', 'timestamp'),
        Index('idx_system_health_component', 'component'),
    )


class AgentMetrics(Base):
    """Agent performance metrics."""
    
    __tablename__ = "agent_metrics"
    
    metric_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )
    
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        server_default=func.now()
    )
    
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    tags = Column(JSONB, nullable=True, default=dict, server_default="{}")
    
    __table_args__ = (
        Index('idx_agent_metrics_timestamp', 'timestamp'),
        Index('idx_agent_metrics_name', 'metric_name'),
    )