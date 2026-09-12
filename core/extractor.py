from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from .summarize import get_model

load_dotenv()

def build_chain(system_prompt: str):
    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', "{text}"),
    ])
    return prompt | model | StrOutputParser()

def extract_action_items(transcript: str) -> str:
    # If transcript is very large, limit to the first 12000 characters to prevent token limits
    input_text = transcript[:12000] if len(transcript) > 12000 else transcript
    system_prompt = (
        "You are an expert assistant. From the following video transcript, extract all key action items, "
        "tasks, key takeaways, and decisions made. Format them as clear numbered bullet points."
    )
    chain = build_chain(system_prompt)
    return chain.invoke({"text": input_text})

def extract_questions(transcript: str) -> str:
    input_text = transcript[:12000] if len(transcript) > 12000 else transcript
    system_prompt = (
        "You are an expert assistant. From the following video transcript, extract all open questions, "
        "unanswered queries, or topics that require further investigation. Format them as clear numbered bullet points."
    )
    chain = build_chain(system_prompt)
    return chain.invoke({"text": input_text})
