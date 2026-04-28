import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")
