from fastapi import FastAPI

app = FastAPI(
    title="Production AI Research & Knowledge Agent",
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}