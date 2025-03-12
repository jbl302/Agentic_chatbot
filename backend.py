from pydantic import BaseModel
from typing import List, Dict
from fastapi import FastAPI
from agent import getting_respose_from_agent
import uvicorn
# defining the structure or form of information


# the information from frontend will be validate against this class
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    # Ensure each message is a dictionary with "role" and "content"
    messages: str

    allow_search: bool


allowed_model_names = ['qwen-qwq-32b', 'deepseek-r1-distill-qwen-32b',
                       'llama-3.3-70b-specdec', 'llama-3.2-3b-preview']
# creating endpoint
app = FastAPI(title="Langgraph AI agent")


@app.post("/chat")
def chat_endpoint(request: RequestState):

    # api for receiving the data in the form of request state
    if request.model_name not in allowed_model_names:
        return "Invalid model name maybe it is not available for now"

    else:
        # Ensure messages are properly formatted as a list of dictionaries
        # Add system prompt
        
        llm_name = request.model_name
        query = request.messages
        allowed_search = request.allow_search
        system_prompt = request.system_prompt
        system_prompt+=(' If you cannot find the answer, respond with I don\'t know')
        provider = request.model_provider
        response = getting_respose_from_agent(
            llm_name, query, allowed_search, system_prompt, provider)
        return response


if __name__ == "__main__":

    uvicorn.run(app=app, host='127.0.0.1', port=8000)
