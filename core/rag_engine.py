from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from .vector_store import build_vector_store, load_vector_store, get_retriever
from .summarize import get_model

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def load_rag_chain(vector_store=None, transcript: str = None):
    if vector_store is None:
        if transcript:
            vector_store = build_vector_store(transcript)
        else:
            vector_store = load_vector_store()

    retriever = get_retriever(vector_store)
    llm = get_model()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert video assistant. Answer the user's question 
based ONLY on the video transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the video transcript."

Always be concise, accurate, and direct.

Context from video transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    answer = rag_chain.invoke(question)
    return answer.strip()

