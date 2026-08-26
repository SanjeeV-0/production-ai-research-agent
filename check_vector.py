import asyncio
import selectors

from sqlalchemy import text

from app.core.database import engine


async def main():
    async with engine.connect() as c:
        result = await c.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        print(result.all())
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
    )
