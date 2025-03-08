from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "./t5-finetuned-es-en-run1"  # Path to your fine-tuned model
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Initialize FastAPI
app = FastAPI()

# Serve static files (for the frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the input schema
class TranslationRequest(BaseModel):
    text: str

# Define the translation endpoint
@app.post("/translate")
def translate(request: TranslationRequest):
    try:
        # Prepare input text
        input_text = "translate Spanish to English: " + request.text

        # Tokenize input
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        # Generate translation
        outputs = model.generate(input_ids)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return the translation
        return {"translation": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read())

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#Additional stuff coming soon