# IGEL RFP Assistant

This project provides a Streamlit application for assisting with security questionnaires. It can answer questions using a cached knowledge base powered by a Chroma vector store or by querying OpenAI's API.

## Features
- Upload a CSV file of security questions and automatically fill in answers from the knowledge base.
- Optionally query the OpenAI assistant for unanswered questions.
- Chatbot mode for single question interaction.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file based on `.env.example` and set your OpenAI API key.
   - `OPENAI_API_KEY` – your OpenAI API key.
   - `CHROMA_PERSIST_DIR` – optional path for storing the Chroma database (defaults to `chroma_storage`).
3. Run the application with Streamlit:
   ```bash
   streamlit run app.py
   ```

## Usage
The app has two tabs:
1. **CSV Mode** – upload a CSV with a `Question` column to automatically generate answers.
2. **Chatbot Mode** – interactively ask questions and update the knowledge base.

All approved answers are stored in the vector database for future reuse.
