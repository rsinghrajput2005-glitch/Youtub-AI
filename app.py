import logging

import streamlit as st
from dotenv import load_dotenv

from core.transcribe import TranscriptError, get_transcript
from core.summarize import summary as summarize
from core.extractor import extract_action_items, extract_questions
from core.vector_store import build_vector_store
from core.rag_engine import load_rag_chain, ask_question

load_dotenv()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI YouTube Video Assistant",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "pipeline_result": None,   # dict returned by run_pipeline
    "chat_history": [],        # list of (role, text) tuples
    "current_video": None,     # last processed URL, so we know when to reset
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def reset_state():
    st.session_state.pipeline_result = None
    st.session_state.chat_history = []
    st.session_state.current_video = None


# ---------------------------------------------------------------------------
# Pipeline runner — mirrors run_pipeline() from the CLI script, but reports
# progress to a Streamlit status widget instead of print()
# ---------------------------------------------------------------------------
def run_pipeline_streamlit(video_link: str):
    result = {}

    with st.status("Starting AI Video Assistant...", expanded=True) as status:
        status.write("⏳ [1/5] Fetching video transcript...")
        transcript = get_transcript(video_link)
        result["transcript"] = transcript
        status.write(f"✓ Transcript fetched successfully ({len(transcript)} characters)")

        status.write("⏳ [2/5] Generating summary...")
        result["summary"] = summarize(transcript)
        status.write("✓ Summary generated")

        status.write("⏳ [3/5] Extracting action items and key takeaways...")
        result["action_items"] = extract_action_items(transcript)
        status.write("✓ Action items extracted")

        status.write("⏳ [4/5] Extracting open questions...")
        result["open_questions"] = extract_questions(transcript)
        status.write("✓ Open questions extracted")

        status.write("⏳ [5/5] Building RAG vector index for Q&A...")
        vector_store = build_vector_store(transcript)
        result["rag_chain"] = load_rag_chain(vector_store=vector_store)
        status.write("✓ Vector index ready")

        status.update(label="Pipeline complete ✅", state="complete", expanded=False)

    return result


# ---------------------------------------------------------------------------
# Sidebar — input + controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 Video Assistant")
    st.caption("Summarize, extract action items, and chat with any YouTube video.")

    video_link = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    col1, col2 = st.columns(2)
    process_clicked = col1.button("🚀 Process", type="primary", use_container_width=True)
    reset_clicked = col2.button("🔄 Reset", use_container_width=True)

    if reset_clicked:
        reset_state()
        st.rerun()

    if st.session_state.pipeline_result:
        st.divider()
        st.caption(f"Loaded: {st.session_state.current_video}")

# ---------------------------------------------------------------------------
# Handle "Process" click
# ---------------------------------------------------------------------------
if process_clicked:
    if not video_link.strip():
        st.sidebar.error("Please enter a YouTube URL first.")
    else:
        # New video → wipe old chat history
        if video_link.strip() != st.session_state.current_video:
            st.session_state.chat_history = []

        try:
            st.session_state.pipeline_result = run_pipeline_streamlit(video_link.strip())
            st.session_state.current_video = video_link.strip()
        except (ValueError, TranscriptError) as error:
            st.sidebar.error(str(error))
            st.session_state.pipeline_result = None
        except Exception:
            logger.exception("Unexpected Streamlit pipeline failure")
            st.sidebar.error(
                "We couldn't process this video right now. Please try again later."
            )
            st.session_state.pipeline_result = None

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.header("🎬 AI YouTube Video Assistant")

result = st.session_state.pipeline_result

if not result:
    st.info("👈 Paste a YouTube link in the sidebar and click **Process** to get started.")
else:
    tab_summary, tab_actions, tab_questions, tab_chat, tab_transcript = st.tabs(
        ["📋 Summary", "✅ Action Items", "❓ Open Questions", "💬 Chat", "📜 Transcript"]
    )

    with tab_summary:
        st.subheader("Summary")
        st.write(result["summary"])

    with tab_actions:
        st.subheader("Action Items & Key Takeaways")
        st.write(result["action_items"])

    with tab_questions:
        st.subheader("Open Questions")
        st.write(result["open_questions"])

    with tab_transcript:
        st.subheader("Full Transcript")
        st.text_area(
            "Transcript",
            result["transcript"],
            height=400,
            label_visibility="collapsed",
        )

    with tab_chat:
        st.subheader("Chat with your video")

        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(text)

        question = st.chat_input("Ask a question about the video...")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as error:
                        answer = f"⚠️ Something went wrong: {error}"
                st.markdown(answer)

            st.session_state.chat_history.append(("assistant", answer))