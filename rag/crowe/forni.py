from flask import Flask, request, render_template, session, jsonify
from flask_session import Session
import asyncio
import fitz
from dotenv import load_dotenv
import os
from myworkers import outfull  # Import the AI response function

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "default_secret")
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)

preset_questions = [
    "What are the types of foundations used in residential buildings?",
    "How does cloud computing differ from server infrastructure?",
    "What are the key fashion trends for this summer season?",
    "What are the main fashion trends 2025?"
]

def extract_from_pdf(file_stream):
    """Extracts text from an uploaded PDF file"""
    text = ""
    try:
        file_stream.seek(0) 
        with fitz.open(stream=file_stream.read(), filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text("\n")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return f"Error reading PDF file: {e}"
    return text

@app.route("/clear_session", methods=["POST"])
def clear_session():
    session.clear()
    session.modified = True 
    return jsonify({"message": "Chat history cleared successfully!"})

@app.route("/", methods=["GET", "POST"])
def main():
    response = ""

    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        user_input = request.form.get("input_text", "") or request.form.get("preset_questions")
        pdf_input = request.files.get("pdf_upload")

        if user_input and user_input.strip():
            session["history"].append({'user': user_input, 'bot': ''})
            session.modified = True

            try:
                context = "\n".join([f"User: {item['user']}\nBot: {item['bot']}" for item in session["history"]])
                prompt = f"Here is the conversation history:\n{context}\nUser: {user_input}\nBot:"
                response = asyncio.run(outfull(prompt))  
                session["history"][-1]['bot'] = response
                session.modified = True
            except Exception as e:
                response = f"Error generating response: {e}"
                session["history"][-1]['bot'] = response
                session.modified = True
        
        elif pdf_input:
            try:
                pdf_text = extract_from_pdf(pdf_input)
                if pdf_text.startswith("Error"):
                    response = pdf_text  # Handle extraction errors
                else:
                    prompt = f"Below is the extracted text from a PDF:\n{pdf_text[:3000]}\nPlease answer questions based on this document."
                    response = asyncio.run(outfull(prompt))  
                    session["history"].append({'user': "PDF Document", 'bot': response})
                    session.modified = True
            except Exception as e:
                response = f"Error processing PDF: {e}"
                session["history"].append({'user': "PDF Upload Error", 'bot': response})
                session.modified = True

        print(f"Response: {response}")

    return render_template("chatting.html", response=response, history=session.get("history", []), preset_questions=preset_questions)

if __name__ == "__main__":
    app.run(debug=True)
