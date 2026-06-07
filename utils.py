import re
import nltk
import joblib
import PyPDF2
import docx
import io
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean and normalize resume/JD text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words 
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def extract_text_from_pdf(file_bytes):
    """Extract text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_docx(file_bytes):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error reading DOCX: {e}"

def extract_text(uploaded_file):
    """Auto-detect file type and extract text."""
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    elif name.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    elif name.endswith('.txt'):
        return file_bytes.decode('utf-8', errors='ignore')
    return ""

def load_model(model_path='model/tfidf_model.pkl'):
    """Load the saved TF-IDF model."""
    return joblib.load(model_path)

def rank_resumes(jd_text, resumes_texts, model):
    """
    Rank resumes by similarity to job description.
    jd_text: str — raw job description
    resumes_texts: list of str — raw resume texts
    model: loaded TF-IDF vectorizer
    Returns: list of (index, score) sorted by score descending
    """
    cleaned_jd = clean_text(jd_text)
    cleaned_resumes = [clean_text(r) for r in resumes_texts]
    
    jd_vector = model.transform([cleaned_jd])
    resume_vectors = model.transform(cleaned_resumes)
    
    scores = cosine_similarity(jd_vector, resume_vectors)[0]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked
