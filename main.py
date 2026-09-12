from dotenv import load_dotenv
from core.transcribe import get_transcript
from core.summarize import summary as summarize
from core.extractor import extract_action_items, extract_questions
from core.vector_store import build_vector_store
from core.rag_engine import load_rag_chain, ask_question

load_dotenv()

def run_pipeline(video_link: str):
    print("\n🚀 Starting AI Video Assistant...")

    print("⏳ [1/5] Fetching video transcript...")
    transcript = get_transcript(video_link)
    print(f"   ✓ Transcript fetched successfully ({len(transcript)} characters)")

    print("⏳ [2/5] Generating summary...")
    summary_text = summarize(transcript)
    print("   ✓ Summary generated")

    print("⏳ [3/5] Extracting action items and key takeaways...")
    action_items = extract_action_items(transcript)
    print("   ✓ Action items extracted")

    print("⏳ [4/5] Extracting open questions...")
    questions = extract_questions(transcript)
    print("   ✓ Open questions extracted")

    print("⏳ [5/5] Building RAG vector index for Q&A...")
    vector_store = build_vector_store(transcript)
    rag_chain = load_rag_chain(vector_store=vector_store)
    print("   ✓ Vector index ready")

    return {
        "transcript": transcript,
        "summary": summary_text,
        "action_items": action_items,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    print("=" * 60)
    print("           🎬 AI YOUTUBE VIDEO ASSISTANT")
    print("=" * 60)
    video_link = input("Enter YouTube URL (e.g. https://www.youtube.com/watch?v=...): ").strip()
    if not video_link:
        print("❌ No URL provided. Exiting.")
        raise SystemExit(1)

    try:
        result = run_pipeline(video_link)
    except Exception as error:
        print(f"\n❌ Pipeline failed: {error}")
        raise SystemExit(1) from error

    print("\n" + "=" * 60)
    print(f"\n📋 SUMMARY:\n{result['summary']}")
    print(f"\n✅ ACTION ITEMS & KEY TAKEAWAYS:\n{result['action_items']}")
    print(f"\n❓ OPEN QUESTIONS:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your video via RAG
    print("\n💬 Chat with your video! Ask any question (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            if not question:
                continue
            print("⏳ Thinking...")
            answer = ask_question(rag_chain, question)
            print(f"\n🤖 Assistant: {answer}\n")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break