# CHANOT-Industry-
AI Industry Chatbot (Flask App) A smart, domain-specific chatbot built with Flask, designed to answer user queries related to Construction, Technology, and Fashion. Powered by Gemini AI and advanced NLP agents, it processes PDFs, verifies domains, and delivers accurate, context-aware responses through a simple web interface.

# 🤖 AI Industry Chatbot (Flask + Gemini)

A smart, domain-specific chatbot built with **Flask**, designed to answer queries related to **Construction**, **Technology**, and **Fashion**. This bot uses **Google's Gemini 1.5 Flash** model (via OpenAI API) to deliver fast, accurate, and context-aware responses through a clean, user-friendly web interface.

---

## 🚀 Features

- 🔍 **Domain-Aware Responses**: Classifies and routes user queries to expert agents (Construction, Technology, or Fashion).
- 🧠 **Gemini AI Integration**: Powered by Google's Gemini 1.5 Flash via OpenAI wrapper for fast, intelligent replies.
- 💬 **Conversation History**: Tracks and displays chatbot history during sessions for context-aware interactions.
- 🛡️ **Input Guardrails**: Uses a verification agent to ensure the query fits within supported domains.
- 🌐 **Responsive Web Interface**: Simple HTML/CSS front-end for easy, interactive conversations.
- 📄 **PDF Upload (Coming Soon...)**: Support for PDF-based query context is under development.

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **AI Model**: Gemini 1.5 Flash (via OpenAI Python Wrapper)
- **Domain Logic**: Custom NLP Agents & IndustryOutput Classifier
- **Frontend**: HTML, CSS (Vanilla)
- **PDF Processing**: (In Progress using `fitz` / PyMuPDF)

---

## 📁 Project Structure

RAG/
├── crowe/
│   ├── __pycache__/            # Compiled bytecode
│   ├── .venv/                  # Virtual environment
│   ├── flask_session/          # Flask session-related data
│   ├── src/                    # Core logic (agents, processing, etc.)
│   ├── static/                 # CSS, JS, assets
│   ├── templates/              # HTML templates
│   ├── forni.py                # Main Flask app or logic handler
│   └── myworkers.py            # Worker logic, probably model/agent code
├── .env                        # Environment variables (API keys)
├── .gitignore
├── .python-version             # Python version specification
├── pyproject.toml              # Project dependencies and metadata
├── uv.lock                     # Dependency lock file
└── README.md                   # You’re reading it 😉
![image](https://github.com/user-attachments/assets/be9aed05-30ad-46ad-86af-ee9f3ba981d9)
