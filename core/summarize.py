import os
from functools import lru_cache
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

@lru_cache(maxsize=1)
def get_model():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=token)

        def _invoke(input_val):
            if hasattr(input_val, "to_messages"):
                messages = []
                for m in input_val.to_messages():
                    role = "user" if m.type == "human" else ("assistant" if m.type == "ai" else "system")
                    messages.append({"role": role, "content": str(m.content)})
            elif isinstance(input_val, list):
                messages = input_val
            elif isinstance(input_val, str):
                messages = [{"role": "user", "content": input_val}]
            else:
                messages = [{"role": "user", "content": str(input_val)}]

            res = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=messages,
                max_tokens=512,
                temperature=0.3,
            )
            return res.choices[0].message.content

        return RunnableLambda(_invoke)
    else:
        # Local CPU fallback
        from transformers import pipeline
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
        pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_new_tokens=200,
            max_length=None,
            clean_up_tokenization_spaces=False
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return ChatHuggingFace(llm=llm)

def split_transcript(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)

def summary(transcript: str) -> str:
    model = get_model()

    # If transcript is short enough, summarize directly in one pass
    if len(transcript) <= 3500:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert summarizer. Summarize the following video transcript clearly and concisely with key highlights."),
            ("human", "{text}"),
        ])
        chain = prompt | model | StrOutputParser()
        return chain.invoke({"text": transcript})

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a video transcript concisely with key points."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | model | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summary = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summary)

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert summarizer. Combine these partial summaries into one final comprehensive summary with key takeaways."),
        ("human", "{text}"),
    ])

    combined_chain = combined_prompt | model | StrOutputParser()
    return combined_chain.invoke({"text": combined})