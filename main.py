from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from utils import cleaned_tokenized

app = FastAPI()

class Reviews(BaseModel):
    reviews: List[str]

@app.post("/submit-reviews/")
async def submit_reviews(reviews: Reviews):
    if not reviews.reviews:
        raise HTTPException(status_code=400, detail="No reviews provided")

    sentiments = [cleaned_tokenized(review) for review in reviews.reviews]
    positive_count = sum(sentiments)
    negative_count = len(sentiments) - positive_count

    positive_percentage = (positive_count / len(sentiments)) * 100
    negative_percentage = (negative_count / len(sentiments)) * 100

    return {
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
