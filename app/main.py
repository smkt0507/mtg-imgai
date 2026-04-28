from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routers.analyze import router as analyze_router
from app.routers.scrape import router as scrape_router

app = FastAPI(
    title="MTG Card Analyzer",
    description="MTGカード画像URLからセットコードとコレクター番号を返すAPI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(scrape_router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")
