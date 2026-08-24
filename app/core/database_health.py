from sqlalchemy import text

from app.core.database import engine


async def check_database_connection() -> bool:
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        return True

    except Exception:
        return False