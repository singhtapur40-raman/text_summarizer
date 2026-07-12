from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

device = torch.device("cpu")

tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")

model.to(device)
model.eval()

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

# def summarize_dialogue(dialogue : str) -> str:
#     dialogue = clean_data(dialogue) # clean

#     # tokenize
#     inputs = tokenizer(
#     "summarize: " + dialogue,
#     return_tensors="pt",
#     truncation=True,
#     max_length=512).to(device)

#     # generate the summary => token ids
#     model.to(device)
#     targets = model.generate(
#         input_ids=inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         max_length=150,
#         num_beams=4,
#         early_stopping=True
#     )

def summarize_dialogue(dialogue):

    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        "summarize: " + dialogue,
        return_tensors="pt",
        max_length=256,
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_length=80,
        min_length=20,
        num_beams=6,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # # decoded our output
    # summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    # return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})