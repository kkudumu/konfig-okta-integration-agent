"""
Alembic environment configuration for Konfig database migrations.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import AsyncEngine

from konfig.config.settings import get_settings
from konfig.database.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

# Get database URL from settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", str(settings.database.url))


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # Enable batch mode for better PostgreSQL support
        compare_type=True,     # Compare column types
        compare_server_default=True,  # Compare server defaults
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations with a connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # Enable batch mode for better PostgreSQL support
        compare_type=True,     # Compare column types
        compare_server_default=True,  # Compare server defaults
        # Enable constraint naming conventions
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


def render_item(type_, obj, autogen_context):
    """Apply custom rendering for specific items."""
    if type_ == "type" and hasattr(obj, "item_type"):
        # Handle pgvector Vector type
        if obj.item_type.__class__.__name__ == "Vector":
            return f"sa.Text()  # Vector({obj.item_type.dim})"
    
    # Use default rendering for other items
    return False


async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    # Convert sync URL to async URL for async engine
    database_url = str(settings.database.url)
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    
    connectable = AsyncEngine(
        engine_from_config(
            {
                "sqlalchemy.url": database_url,
                "sqlalchemy.pool_pre_ping": True,
                "sqlalchemy.pool_recycle": 300,
            },
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Check if we're running in async mode
    if settings.database.url and "asyncpg" in str(settings.database.url):
        asyncio.run(run_async_migrations())
    else:
        # Synchronous mode
        connectable = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()