from fastapi import FastAPI

# The file where NeuralSearcher is stored
from neural_searcher import NeuralSearcher
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a neural searcher instance
neural_searcher = NeuralSearcher(collection_name="news")


@app.get("/api/search")
async def search_startup(question: str):
    return neural_searcher.ask(question)


@app.get("/api/latest")
def get_latest():
    question = "what is the latest news today"
    return neural_searcher.ask(question=question)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
