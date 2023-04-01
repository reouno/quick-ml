"""main"""

from fastapi import FastAPI

from routers import text, vision, debug, chat

app = FastAPI()

router_prefix = '/quick-ml/api'
app.include_router(text.router, prefix=router_prefix)
app.include_router(vision.router, prefix=router_prefix)
app.include_router(chat.router, prefix=router_prefix)

app.include_router(debug.router, prefix=f'{router_prefix}/debug')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/quick-ml/")
async def root():
    return {"message": "Hello World"}
