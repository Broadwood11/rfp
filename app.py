
import streamlit as st
from openai import OpenAI
import os
import time
import io
import pandas as pd
import chromadb
from dotenv import load_dotenv
import pyperclip

# -------- ENVIRONMENT SETUP --------
load_dotenv()
ASSISTANT_ID = "asst_c9vPDOjozUpwMFMmPLEG1J0A"
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Set the OPENAI_API_KEY environment variable.")
    st.stop()

client = OpenAI(api_key=api_key)
# Persist the Chroma database in a path that can be overridden via the
# `CHROMA_PERSIST_DIR` environment variable. Defaults to `chroma_storage`
# in the current working directory so the app works cross-platform.
persist_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_storage")
chroma_client = chromadb.PersistentClient(path=persist_dir)
collection = chroma_client.get_or_create_collection("rfp_answers")

def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Embedding error: {e}")
        return None

def search_cached_answer(question, threshold=0.75):
    emb = get_embedding(question)
    if emb is None:
        return None
    try:
        results = collection.query(query_embeddings=[emb], n_results=1)
        if results["documents"][0]:
            distance = results["distances"][0][0]
            meta = results["metadatas"][0][0]
            if distance < (1 - threshold):
                return meta.get("answer", results["documents"][0][0])
    except Exception as e:
        st.warning(f"Vector DB search error: {e}")
    return None

def add_to_vector_db(question, answer, source="openai"):
    doc_id = f"qa_{hash(question)}"
    emb = get_embedding(question)
    if emb is not None:
        try:
            try:
                collection.delete(ids=[doc_id])
            except Exception:
                pass
            collection.add(
                documents=[question],
                embeddings=[emb],
                metadatas=[{"answer": answer, "source": source}],
                ids=[doc_id]
            )
        except Exception as e:
            st.warning(f"Vector DB update error: {e}")

def ask_spike(question, context):
    try:
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Q: {question}\nA:"
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            instructions=context
        )
        while run.status in ("queued", "in_progress"):
            time.sleep(0.5)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        messages = client.beta.threads.messages.list(thread_id=thread.id).data
        for m in messages:
            if m.role == "assistant":
                return "".join(
                    part.text.value for part in m.content if hasattr(part, "text") and hasattr(part.text, "value")
                )
    except Exception as e:
        st.error(f"Spike API error: {e}")
    return "Error retrieving Spike response."

# -------- UI --------
st.set_page_config(page_title="IGEL RFP Assistant", layout="centered")

st.title("🛡️ IGEL Security Questionnaire Assistant")
tab1, tab2 = st.tabs(["📄 CSV Mode", "💬 Chatbot Mode"])

# -------- TAB 1: CSV Upload --------
with tab1:
    st.header("Upload Security Questionnaire (CSV)")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    sensitivity_thresholds = {"Standard": 0.75, "High": 0.85, "Very High": 0.95}
    sensitivity = st.selectbox("Compliance Sensitivity", ["Standard", "High", "Very High"])
    use_openai = st.checkbox("Use Spike (AI) if no cached answers", value=True)
    current_threshold = sensitivity_thresholds[sensitivity]
    additional_notes = st.text_area("Additional Notes (Optional)")

    if "csv_session" not in st.session_state:
        st.session_state.csv_session = {}

    if st.button("Process CSV") and uploaded_file:
        df = pd.read_csv(uploaded_file)
        kb_answers = {}
        openai_pending = []

        context_prompt = f"""
You are an IGEL security analyst completing a security questionnaire.
Additional Notes: {additional_notes}
Answer concisely, with sufficient compliance detail. If unsure, state 'Requires SME review'.
"""

        for idx, row in df.iterrows():
            question = row["Question"]
            if pd.isna(question) or not str(question).strip():
                kb_answers[idx] = ""
                continue
            cached_answer = search_cached_answer(question, threshold=current_threshold)
            if cached_answer:
                kb_answers[idx] = cached_answer + " (From Knowledge Base)"
                continue
            if not use_openai:
                kb_answers[idx] = "No cached answer found. Requires SME review."
                continue
            spike_ans = ask_spike(question, context_prompt)
            openai_pending.append({
                "idx": idx,
                "question": question,
                "openai_answer": spike_ans or "No response"
            })

        st.session_state.csv_session = {
            "df": df,
            "kb_answers": kb_answers,
            "openai_pending": openai_pending
        }
        st.success("Processing complete. Review answers below.")

    if st.session_state.csv_session:
        data = st.session_state.csv_session
        df = data["df"]
        kb_answers = data["kb_answers"]
        openai_pending = data["openai_pending"]
        approved_openai = {}

        st.markdown("### Review Spike's Answers")
        for qa in openai_pending:
            idx = qa["idx"]
            question = qa["question"]
            ai_answer = qa["openai_answer"]
            key_str = f"approval_{idx}"
            st.text(f"Q{idx+1}: {question}")
            approved = st.text_area("Edit or approve:", value=ai_answer, key=key_str)
            approved_openai[idx] = approved
            st.markdown("---")

        if st.button("Finalize & Export CSV", key="finalize_export"):
            for idx, ans in kb_answers.items():
                df.at[idx, "Answer"] = ans
            for idx, ans in approved_openai.items():
                df.at[idx, "Answer"] = ans
                add_to_vector_db(df.at[idx, "Question"], ans, source="Spike")
            output = io.StringIO()
            df.to_csv(output, index=False)
            st.download_button("Download Completed CSV", output.getvalue(), file_name="completed_questionnaire.csv")
            st.success("File ready for download.")

# -------- TAB 2: Chatbot --------
with tab2:
    st.header("Chat with IGEL RFP Assistant")
    question = st.text_area("Security Questionnaire Question:", height=150)
    threshold = st.slider("Answer Match Sensitivity", 0.7, 0.95, 0.75)
    allow_spike = st.checkbox("Use Spike (AI) if no cached answer", value=True)

    if "final_answer" not in st.session_state:
        st.session_state.final_answer = ""
    if "cached_hit" not in st.session_state:
        st.session_state.cached_hit = False

    if st.button("Get Answer", key="chat_get_answer") and question:
        cached = search_cached_answer(question, threshold)
        st.session_state.cached_hit = bool(cached)
        if cached:
            st.success("✅ Found cached answer.")
            st.session_state.final_answer = cached
        elif allow_spike:
            st.warning("⏳ Asking Spike...")
            ctx = "Answer as an IGEL security SME. Provide clear, accurate, and security-compliant responses."
            spike_ans = ask_spike(question, ctx)
            st.session_state.final_answer = spike_ans or "No response from Spike."
        else:
            st.session_state.final_answer = "No cached answer and Spike disabled."

    if st.session_state.get("final_answer"):
        edited = st.text_area("Edit or Approve the Answer", value=st.session_state.final_answer, height=160)
        if st.button("✅ Update and Copy", key="chat_update_copy"):
            pyperclip.copy(edited)
            source_type = "cache" if st.session_state.get("cached_hit") else "Spike"
            add_to_vector_db(question, edited, source=source_type)
            st.success("✔️ Copied and saved to Knowledge Base.")
