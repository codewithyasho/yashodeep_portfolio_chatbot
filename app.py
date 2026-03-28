from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from main import main


# ========== Initialize FastAPI ==========
app = FastAPI(
    title="Yashodeep's Personal Chatbot API",
    description="A RAG-based chatbot that answers questions about Yashodeep Hundiwale's skills, education, experience and etc.",
    version="1.0.0"
)

# ========== Configure CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query to the chatbot.", examples=[
                       "What is Yashodeep's educational background?"])


@app.get("/")
def home():
    return {"message": "Welcome to Yashodeep's Personal Chatbot API! Use the /chat endpoint to interact with the chatbot."}


rag_chain = main()


@app.post("/chat")
async def chat(query_request: QueryRequest):
    try:
        response = rag_chain.invoke({
            "input": query_request.query
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": response["answer"]}
