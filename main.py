from rag_retriver import vector_search, cohere_completion_with_vector_search
from fastapi import FastAPI 
from pydantic import BaseModel


class validation(BaseModel):
    prompt: str 


app = FastAPI()

@app.post('/rag')
async def retrival(item: validation):
    rag = vector_search(item.prompt)
    completion = cohere_completion_with_vector_search(item.prompt, rag)

    return completion