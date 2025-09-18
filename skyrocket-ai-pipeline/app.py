import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()

from src.preprocess import prepare_queries
from src.llm_utils import (
    classify_query,
    generate_synthetic_queries,
    evaluate_response,
    generate_chat_response,
    score_chat_response,
)
from src.entities import add_entities_column
from src.sentiment import add_sentiment_column


# ================= Page Config =================
st.set_page_config(page_title="SkyRocket AI Automation Pipeline", layout="wide")

# ================= Streamlit UI =================
st.title("📊 SkyRocket AI Automation Pipeline")

# ✅ Single uploader with unique key
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"], key="queries_upload")

if uploaded_file:
    queries_df = pd.read_excel(uploaded_file, sheet_name="Queries")

    # 🔹 Generate responses via LLM
    st.subheader("💬 Generating Responses via LLM")

    run_all = st.checkbox(
        "Generate responses for all queries (⚠️ may take long and use more credits)",
        value=False,
    )

    if run_all:
        query_list = queries_df["Queries"].astype(str).tolist()
    else:
        query_list = queries_df["Queries"].astype(str).tolist()[:5]

    responses, scores = [], []
    progress = st.progress(0)

    for i, q in enumerate(query_list):
        resp = generate_chat_response(q, model="gpt-4o")
        sc = score_chat_response(q, resp, model="gpt-4o")
        responses.append(resp)
        scores.append(sc)
        progress.progress((i + 1) / len(query_list))

    responses_df = pd.DataFrame(
        {"query": query_list, "response": responses, "scores": scores}
    )
    st.dataframe(responses_df.head())

    # 🔹 Preprocessing
    st.subheader("🔹 Preprocessed Queries")
    queries_df = prepare_queries(queries_df)   # advanced cleaning
    st.dataframe(queries_df[["Queries", "clean", "length"]].head())

    # 🔹 Query length distribution (smaller graph)
    st.subheader("📈 Query Length Distribution")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.histplot(queries_df["length"], bins=30, ax=ax)
    st.pyplot(fig)

    # 🔹 Embeddings + Clustering
    st.subheader("🔹 Topic Clustering")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(queries_df["clean"].tolist(), show_progress_bar=True)
    kmeans = KMeans(n_clusters=10, random_state=42)
    queries_df["topic"] = kmeans.fit_predict(embeddings)
    st.write(queries_df.groupby("topic").head(5)[["Queries", "topic"]])

    # 🔹 Entities
    st.subheader("🔹 Entity Extraction")
    queries_df = add_entities_column(queries_df, text_col="clean")
    st.dataframe(queries_df[["Queries", "entities"]].head(10))

    # 🔹 Sentiment on responses (binary model, smaller graph)
    st.subheader("📊 Sentiment Analysis on LLM Responses")
    responses_df = add_sentiment_column(responses_df, text_col="response")
    print(responses_df)
    fig2, ax2 = plt.subplots(figsize=(4,3))
    sns.countplot(
        x="sentiment",
        data=responses_df,
        order=[ "POSITIVE","NEGATIVE"],
        ax=ax2,
    )
    st.pyplot(fig2)

    proportions = responses_df["sentiment"].value_counts(normalize=True) * 100
    st.write("Sentiment distribution (%):")
    st.write(proportions)

    # 🔹 Show LLM evaluation scores
    st.subheader("📝 LLM-as-Judge Quality Scores")
    st.json(scores[:5])  # sample of first 5

    # 🔹 Download results
    st.subheader("⬇️ Download Processed Data")
    st.download_button(
        "Download Queries CSV",
        queries_df.to_csv(index=False),
        "queries_processed.csv"
    )
    st.download_button(
        "Download Responses CSV",
        responses_df.to_csv(index=False),
        "responses_processed.csv"
    )

# ================= Extra Tools =================
st.subheader("🔹 LLM Query Classifier")
user_q = st.text_input("Enter a customer query:")
if user_q:
    category = classify_query(user_q)
    st.write("Predicted Category:", category)

st.subheader("🔹 Synthetic Query Generator")
topic = st.selectbox("Choose topic:", ["ORDER", "SHIPPING", "REFUND", "ACCOUNT"])
if st.button("Generate Samples"):
    samples = generate_synthetic_queries(topic, n=5)
    st.write(samples)

# ✅ Auto-generate + evaluate responses
st.subheader("🔹 LLM Response Evaluator")
q = st.text_input("Enter a customer query for evaluation:")

if st.button("Evaluate with LLM"):
    if q.strip():
        # Step 1: Generate chatbot response
        llm_response = generate_chat_response(q, model="gpt-4o")

        # Step 2: Evaluate that response
        evaluation = score_chat_response(q, llm_response, model="gpt-4o")

        # Display side-by-side in wide layout
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Chatbot Response (LLM-generated):**")
            st.write(llm_response)

        with col2:
            st.markdown("**Evaluation (LLM-as-Judge):**")
            st.json(evaluation)
    else:
        st.warning("Please enter a query first.")
