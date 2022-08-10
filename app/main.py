"""main"""

from fastapi import FastAPI

from routers import text

app = FastAPI()
app.include_router(text.router, prefix='/quick-ml/api')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/quick-ml/")
async def root():
    return {"message": "Hello World"}
