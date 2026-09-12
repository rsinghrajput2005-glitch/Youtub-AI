# YouTube AI

### AI-Powered YouTube Video Assistant using RAG

YouTube AI is an AI-powered chatbot that allows users to interact with YouTube videos using their transcripts.

Instead of watching an entire video to find specific information, users can ask questions about the video and receive relevant answers based on its content.

The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant parts of the transcript before generating an answer.

---

## 🚀 Features

* 🎥 YouTube video transcript extraction
* 🌐 Automatic transcript translation to English
* 🔎 Semantic search over video content
* 🤖 AI-powered question answering
* 📝 Video summarization
* ✅ Action-item extraction
* ❓ Open-question extraction
* 💬 Conversational interaction with video content
* 📚 Vector database-based retrieval

---

## 🧠 How It Works

```text
YouTube Video
      ↓
Extract Transcript
      ↓
Translate Transcript
      ↓
Clean & Process Text
      ↓
Split into Chunks
      ↓
Generate Embeddings
      ↓
Store in ChromaDB
      ↓
Retrieve Relevant Chunks
      ↓
LLM
      ↓
Answer / Summary / Action Items
```

---

## 🏗️ RAG Architecture

The project follows a Retrieval-Augmented Generation architecture.

### 1. Transcript Extraction

The YouTube video's transcript is obtained using:

```text
youtube-transcript-api
```

The transcript contains the spoken content of the video.

### 2. Translation

If the transcript is not in English, it can be translated using:

```text
deep-translator
```

This allows the system to work with transcripts in different languages.

### 3. Text Chunking

Large transcripts are divided into smaller chunks using:

```text
RecursiveCharacterTextSplitter
```

Example configuration:

```text
Chunk Size: 500
Chunk Overlap: 50
```

Chunk metadata such as `chunk_index` can also be stored.

### 4. Embeddings

Each chunk is converted into a numerical vector representation.

These vectors allow the system to perform semantic similarity search.

### 5. Vector Database

The embeddings are stored in:

```text
ChromaDB
```

This allows relevant sections of the YouTube transcript to be retrieved efficiently.

### 6. Retrieval

When the user asks a question:

```text
User Question
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
Top Relevant Chunks
```

The most relevant transcript chunks are passed to the LLM.

### 7. Generation

The retrieved context is given to the language model to generate the final answer.

```text
Retrieved Context
       +
User Question
       ↓
      LLM
       ↓
Final Answer
```

---

# 💬 Example

### User:

```text
What are the main concepts explained in this video?
```

### YouTube AI:

```text
The video mainly explains:

1. Retrieval-Augmented Generation
2. Vector databases
3. Embeddings
4. Semantic search
```

The answer is generated using relevant sections retrieved from the video's transcript.

---

# 📌 Supported Operations

## Ask Questions

Users can ask questions about the video:

```text
What is RAG?
Why are embeddings required?
How does the retriever work?
What is ChromaDB?
```

---

## Video Summary

The system can generate a concise summary of the video.

```text
YouTube Video
      ↓
Transcript
      ↓
LLM
      ↓
Summary
```

---

## Action Items

The system can identify actionable tasks mentioned in the video.

Example:

```text
Action Items:
- Implement a vector database
- Create document embeddings
- Build a retrieval pipeline
```

---

## Open Questions

The system can identify questions or unresolved points discussed in the video.

---

# 🛠️ Tech Stack

### Programming Language

* Python

### AI / LLM

* Hugging Face Transformers
* Qwen2.5-0.5B-Instruct
* LangChain

### RAG

* LangChain
* ChromaDB
* RecursiveCharacterTextSplitter
* Embeddings

### Data Extraction

* YouTube Transcript API

### Translation

* Deep Translator

---

# 📂 Project Structure

```text
Youtub AI/
│
├── app/
│   ├── transcript.py
│   ├── summary.py
│   ├── rag.py
│   ├── actions.py
│   └── questions.py
│
├── chroma_db/
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

*The structure may vary depending on the final implementation.*

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd "Youtub AI"
```

Create a virtual environment:

```bash
python -m venv youtubAI
```

Activate the environment on Windows:

```bash
youtubAI\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the application:

```bash
python main.py
```

Provide a YouTube video URL and start interacting with the video.

---

# 🔐 Environment Variables

If external APIs are used, create a `.env` file:

```text
HF_TOKEN=your_huggingface_token
```

Never commit `.env` or API keys to GitHub.

Add the following to `.gitignore`:

```text
.env
__pycache__/
venv/
youtubAI/
chroma_db/
```

---

# 🔄 Complete Pipeline

```text
                YouTube URL
                     ↓
             Transcript API
                     ↓
             Transcript Text
                     ↓
              Translation
                     ↓
              Text Cleaning
                     ↓
             Text Chunking
                     ↓
               Embeddings
                     ↓
                ChromaDB
                     ↓
                Retriever
                     ↓
              Relevant Context
                     ↓
                   LLM
                     ↓
          ┌──────────┼──────────┐
          ↓          ↓          ↓
       Answer     Summary    Action Items
```

---

# 🎯 Use Cases

YouTube AI can be useful for:

* 📚 Educational videos
* 💻 Programming tutorials
* 🎓 Online lectures
* 📰 Long interviews
* 🎤 Podcasts
* 📖 Technical tutorials
* 🧑‍💻 Developer documentation videos

Instead of manually searching through a long video, users can directly ask the AI.

---

# 🔮 Future Improvements

* Chrome extension
* Direct timestamp links
* Multilingual responses
* Voice-based interaction
* Conversation memory
* Better long-video summarization
* Multiple YouTube video comparison
* Playlist-level RAG
* Source citations with timestamps
* Streaming responses
* Web-based UI

---

# 👨‍💻 Project Goal

The goal of YouTube AI is to transform passive video watching into an **interactive learning experience** by allowing users to search, understand, summarize, and discuss YouTube videos using natural language.

---

# 📜 License

This project is developed for educational and learning purposes.
