import streamlit as st
import pandas as pd
from utils import extract_text, load_model, rank_resumes

# ─── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# ─── Header ────────────────────────────────────────────
st.title("📄 AI Resume Screening System")
st.markdown("*Powered by TF-IDF & Cosine Similarity*")
st.divider()

# ─── Load Model ────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model('tfidf_model.pkl')

model = get_model()

# ─── Layout: Two Columns ───────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Job Description")
    jd_input_method = st.radio(
        "Input method:", 
        ["Paste text", "Upload file"],
        horizontal=True
    )
    jd_text = ""
    if jd_input_method == "Paste text":
        jd_text = st.text_area(
            "Paste the job description here:",
            height=250,
            placeholder="We are looking for a Python developer..."
        )
    else:
        jd_file = st.file_uploader("Upload JD", type=['pdf', 'docx', 'txt'])
        if jd_file:
            jd_text = extract_text(jd_file)
            st.success(f"Loaded: {jd_file.name}")
            with st.expander("Preview JD text"):
                st.text(jd_text[:500] + "...")

with col2:
    st.subheader("📑 Upload Resumes")
    resume_files = st.file_uploader(
        "Upload multiple resumes (PDF, DOCX, TXT):",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    if resume_files:
        st.info(f"{len(resume_files)} resume(s) uploaded")

st.divider()

# ─── Screening Button ──────────────────────────────────
if st.button("🔍 Screen Resumes", type="primary", use_container_width=True):
    if not jd_text.strip():
        st.error("Please provide a job description.")
    elif not resume_files:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Analyzing resumes..."):
            resume_texts = [extract_text(f) for f in resume_files]
            resume_names = [f.name for f in resume_files]
            ranked = rank_resumes(jd_text, resume_texts, model)

        st.success("Screening complete!")
        st.subheader("🏆 Ranked Candidates")

        results = []
        for rank, (idx, score) in enumerate(ranked, 1):
            results.append({
                "Rank": rank,
                "Resume": resume_names[idx],
                "Match Score": f"{score:.2%}",
                "Score (raw)": round(score, 4),
                "Status": "✅ Top Match" if rank <= 3 else "⚪ Reviewed"
            })

        df = pd.DataFrame(results)

        # Color-coded results
        for _, row in df.iterrows():
            color = "#d4edda" if row["Rank"] <= 3 else "#f8f9fa"
            st.markdown(
                f"""
                

                #{row['Rank']} {row['Resume']} — 
                Match Score: {row['Match Score']} {row['Status']}
                

                """,
                unsafe_allow_html=True
            )

        st.divider()
        st.download_button(
            "⬇️ Download Results as CSV",
            df[["Rank","Resume","Match Score","Status"]].to_csv(index=False),
            "screening_results.csv",
            "text/csv"
        )

# ─── Footer ────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit · scikit-learn · NLTK | Portfolio Project")
