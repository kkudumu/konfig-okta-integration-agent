"""
Database connection and session management for Konfig.

This module provides database connection utilities, session management,
and initialization functions for the PostgreSQL database with pgvector.
"""

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from konfig.config.settings import get_settings
from konfig.database.models import Base
from konfig.utils.logging import get_logger

logger = get_logger(__name__, "database")


class DatabaseManager:
    """
    Database connection and session manager.
    
    This class manages both synchronous and asynchronous database connections,
    provides session factories, and handles database initialization.
    """
    
    def __init__(self):
        """Initialize the database manager."""
        self.settings = get_settings()
        
        # Synchronous engine and session factory
        self._sync_engine: Optional[Engine] = None
        self._sync_session_factory: Optional[sessionmaker] = None
        
        # Asynchronous engine and session factory
        self._async_engine = None
        self._async_session_factory: Optional[async_sessionmaker] = None
        
        # Connection status
        self._initialized = False
        
        logger.info("Database manager initialized")
    
    @property
    def sync_engine(self) -> Engine:
        """Get or create the synchronous database engine."""
        if self._sync_engine is None:
            self._sync_engine = self._create_sync_engine()
        return self._sync_engine
    
    @property
    def async_engine(self):
        """Get or create the asynchronous database engine."""
        if self._async_engine is None:
            self._async_engine = self._create_async_engine()
        return self._async_engine
    
    @property
    def sync_session_factory(self) -> sessionmaker:
        """Get or create the synchronous session factory."""
        if self._sync_session_factory is None:
            self._sync_session_factory = sessionmaker(
                bind=self.sync_engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._sync_session_factory
    
    @property
    def async_session_factory(self) -> async_sessionmaker:
        """Get or create the asynchronous session factory."""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._async_session_factory
    
    def _create_sync_engine(self) -> Engine:
        """Create the synchronous database engine."""
        database_url = str(self.settings.database.url)
        
        # Convert async URL to sync if needed
        if database_url.startswith("postgresql+asyncpg://"):
            database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
        elif not database_url.startswith("postgresql://"):
            database_url = f"postgresql://{database_url}"
        
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # 1 hour
            echo=self.settings.database.echo,
        )
        
        # Add connection event listeners
        event.listen(engine, "connect", self._on_connect)
        event.listen(engine, "checkout", self._on_checkout)
        
        logger.info("Synchronous database engine created", url=database_url.split("@")[-1])
        return engine
    
    def _create_async_engine(self):
        """Create the asynchronous database engine."""
        database_url = str(self.settings.database.url)
        
        # Convert to async URL if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        elif not database_url.startswith("postgresql+asyncpg://"):
            database_url = f"postgresql+asyncpg://{database_url}"
        
        engine = create_async_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # 1 hour
            echo=self.settings.database.echo,
        )
        
        logger.info("Asynchronous database engine created", url=database_url.split("@")[-1])
        return engine
    
    def _on_connect(self, dbapi_connection, connection_record):
        """Handle new database connections."""
        # Set connection-level settings
        with dbapi_connection.cursor() as cursor:
            # Enable pgvector extension if not already enabled
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Set search path
            cursor.execute("SET search_path TO public;")
            # Set timezone
            cursor.execute("SET timezone TO 'UTC';")
        
        logger.debug("Database connection established")
    
    def _on_checkout(self, dbapi_connection, connection_record, connection_proxy):
        """Handle connection checkout from pool."""
        logger.debug("Database connection checked out from pool")
    
    @contextmanager
    def get_sync_session(self) -> Generator[Session, None, None]:
        """
        Get a synchronous database session.
        
        Usage:
            with db_manager.get_sync_session() as session:
                # Use session
                pass
        """
        session = self.sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an asynchronous database session.
        
        Usage:
            async with db_manager.get_async_session() as session:
                # Use session
                pass
        """
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def initialize_database(self) -> None:
        """
        Initialize the database with required extensions and tables.
        
        This method:
        1. Ensures required extensions are installed
        2. Creates all tables if they don't exist
        3. Verifies the database is ready
        """
        if self._initialized:
            logger.info("Database already initialized")
            return
        
        logger.info("Initializing database...")
        
        try:
            # Create tables using sync engine
            async with self.async_engine.begin() as conn:
                # Enable extensions
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
                
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
            
            # Verify database health
            await self.health_check()
            
            self._initialized = True
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
            raise
    
    async def health_check(self) -> dict:
        """
        Perform a database health check.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            "database_connected": False,
            "extensions_installed": False,
            "tables_exist": False,
            "details": {}
        }
        
        try:
            async with self.get_async_session() as session:
                # Test basic connectivity
                result = await session.execute(text("SELECT 1"))
                health_status["database_connected"] = result.scalar() == 1
                
                # Check extensions
                extensions_query = text("""
                    SELECT extname FROM pg_extension 
                    WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm')
                """)
                extensions = await session.execute(extensions_query)
                installed_extensions = {row[0] for row in extensions}
                required_extensions = {'vector', 'uuid-ossp', 'pg_trgm'}
                health_status["extensions_installed"] = required_extensions.issubset(installed_extensions)
                health_status["details"]["installed_extensions"] = list(installed_extensions)
                
                # Check tables
                tables_query = text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """)
                tables = await session.execute(tables_query)
                existing_tables = {row[0] for row in tables}
                required_tables = {
                    'integration_jobs', 
                    'execution_traces', 
                    'procedural_memory_patterns', 
                    'knowledge_base_vectors'
                }
                health_status["tables_exist"] = required_tables.issubset(existing_tables)
                health_status["details"]["existing_tables"] = list(existing_tables)
                
                # Check database size and stats
                stats_query = text("""
                    SELECT 
                        pg_size_pretty(pg_database_size(current_database())) as db_size,
                        (SELECT count(*) FROM integration_jobs) as total_jobs,
                        (SELECT count(*) FROM execution_traces) as total_traces
                """)
                stats = await session.execute(stats_query)
                stats_row = stats.first()
                if stats_row:
                    health_status["details"]["database_size"] = stats_row[0]
                    health_status["details"]["total_jobs"] = stats_row[1]
                    health_status["details"]["total_traces"] = stats_row[2]
        
        except Exception as e:
            health_status["error"] = str(e)
            logger.error("Database health check failed", error=str(e))
        
        return health_status
    
    async def reset_database(self) -> None:
        """
        Reset the database by dropping and recreating all tables.
        
        WARNING: This will delete all data!
        """
        logger.warning("Resetting database - all data will be lost!")
        
        try:
            async with self.async_engine.begin() as conn:
                # Drop all tables
                await conn.run_sync(Base.metadata.drop_all)
                logger.info("All tables dropped")
                
                # Recreate all tables
                await conn.run_sync(Base.metadata.create_all)
                logger.info("All tables recreated")
            
            self._initialized = True
            logger.info("Database reset completed successfully")
            
        except Exception as e:
            logger.error("Database reset failed", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close all database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
            logger.info("Async database engine closed")
        
        if self._sync_engine:
            self._sync_engine.dispose()
            logger.info("Sync database engine closed")


# Global database manager instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


# Convenience functions for common operations

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a synchronous database session."""
    db_manager = get_db_manager()
    with db_manager.get_sync_session() as session:
        yield session


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an asynchronous database session."""
    db_manager = get_db_manager()
    async with db_manager.get_async_session() as session:
        yield session


async def init_database() -> None:
    """Initialize the database."""
    db_manager = get_db_manager()
    await db_manager.initialize_database()


async def health_check() -> dict:
    """Perform a database health check."""
    db_manager = get_db_manager()
    return await db_manager.health_check()


async def reset_database() -> None:
    """Reset the database."""
    db_manager = get_db_manager()
    await db_manager.reset_database()


async def close_connections() -> None:
    """Close all database connections."""
    db_manager = get_db_manager()
    await db_manager.close()