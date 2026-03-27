from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()


loader = TextLoader("data.txt", encoding="utf-8")

docs = loader.load()
# print(docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5")

if not os.path.exists("faiss_index"):
    print("creating new vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
else:
    print("loading existing vectorstore...")
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)


retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,

        "fetch_k": 5,  # it gives 5 candidates, but only pick the 3 most diverse ones

        "lambda_mult": 0.5  # 1.0 = Pure similarity, 0.0 = Pure diversity
    }
)


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

# The chain expects {context} and {input}
prompt = ChatPromptTemplate.from_template("""
    You are a AI Assistant that only answers questions based on the provided context.
    Use the following retrieved context to answer the user's question.    
    Do not generate answers outside the given context.
    Do not make assumptions.             
                                                                   
    If the answer is not found in the context, reply with:
    "I only have information about Yashodeep Hundiwale."

    <context>
    {context}
    </context>

    Question: {input}
""")


# THE CHAIN

document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)


print("\n🚀 RAG System Ready! (Type '0' to exit)")
while True:
    user_query = input("\nYou: ")
    if user_query == "0":
        break

    response = rag_chain.invoke({
        "input": user_query
    })

    print(f"\n🧠 AI: {response['answer']}")
