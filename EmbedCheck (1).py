# MUST BE THE VERY FIRST LINE
import streamlit as st
st.set_page_config(page_title="Plagiarism Detector PRO", layout="wide")

# Import libraries
import pandas as pd
from docx import Document
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Initialize models with error handling
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
        
        # Try to load sentence transformer with fallback
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            st.error("Sentence Transformers not installed. Please run: pip install sentence-transformers")
            return None, None
        except Exception as e:
            st.error(f"Sentence Transformer loading failed: {e}")
            return None, None
            
        return nlp, model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

nlp, model = load_models()

# Text extraction
def extract_text(file):
    try:
        if file.type == "text/plain":
            return file.read().decode("utf-8")
        elif file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() for page in reader.pages])
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading {file.name}: {str(e)}")
    return None

# FIXED: Proper highlighting with corrected threshold logic
def highlight_differences(text1, text2, overall_similarity):
    if not text1 or not text2:
        return "", []
    
    # Split into sentences
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    sentences1 = [sent.text for sent in doc1.sents if len(sent.text.strip()) > 10]
    sentences2 = [sent.text for sent in doc2.sents if len(sent.text.strip()) > 10]
    
    highlighted = []
    explanations = []
    
    # CORRECTED: Proper threshold logic based on overall similarity
    if overall_similarity < 20:
        # Very different documents - be very strict
        similarity_threshold = 0.85
        threshold_type = "Very Strict (85%)"
    elif overall_similarity < 50:
        # Somewhat similar - moderate strictness
        similarity_threshold = 0.75
        threshold_type = "Moderate (75%)"
    elif overall_similarity < 80:
        # Similar documents - normal threshold
        similarity_threshold = 0.65
        threshold_type = "Normal (65%)"
    else:
        # Very similar documents - be more permissive
        similarity_threshold = 0.55
        threshold_type = "Permissive (55%)"
    
    total_sentences = len(sentences1)
    plagiarized_count = 0
    
    # Compare each sentence with all sentences in the other document
    for sent1 in sentences1:
        best_similarity = 0
        best_match = ""
        
        for sent2 in sentences2:
            if len(sent1) > 15 and len(sent2) > 15:  # Only compare meaningful sentences
                similarity = SequenceMatcher(None, sent1.lower(), sent2.lower()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = sent2
        
        # Highlight based on dynamic threshold
        if best_similarity > similarity_threshold:
            highlighted.append(f'<span style="background-color: #ffcccc; border: 1px solid red; padding: 2px; margin: 2px; border-radius: 3px; display: inline-block;">{sent1}</span>')
            if best_similarity > 0.8:
                explanations.append(f"🔴 HIGH PLAGIARISM: {best_similarity:.0%} similar to: '{best_match[:100]}...'")
            elif best_similarity > 0.6:
                explanations.append(f"🟡 MODERATE SIMILARITY: {best_similarity:.0%} similar to: '{best_match[:100]}...'")
            else:
                explanations.append(f"🟠 LOW SIMILARITY: {best_similarity:.0%} similar to: '{best_match[:100]}...'")
            plagiarized_count += 1
        else:
            highlighted.append(f'<span style="padding: 2px; margin: 2px; display: inline-block;">{sent1}</span>')
    
    # Add summary explanation
    if plagiarized_count > 0:
        plagiarism_percentage = (plagiarized_count / total_sentences) * 100
        explanations.insert(0, f"📊 DETECTED: {plagiarism_percentage:.1f}% of sentences show similarity")
        explanations.insert(1, f"⚡ THRESHOLD: Using {threshold_type} (based on {overall_similarity:.1f}% overall similarity)")
        explanations.insert(2, f"📝 OVERALL: Documents are {overall_similarity:.1f}% semantically similar")
    
    return " ".join(highlighted), explanations

def create_meter(percent):
    fig, ax = plt.subplots(figsize=(4,4), facecolor='white')
    ax.set_facecolor('white')
    
    color = '#ff5555' if percent > 50 else '#55ff55'
    
    bg = Circle((0.5,0.5), 0.49, color='#f0f0f0')
    ax.add_patch(bg)
    
    wedge = Circle((0.5,0.5), 0.4, color=color, alpha=0.6)
    ax.add_patch(wedge)
    
    ax.text(0.5, 0.5, f"{percent:.0f}%", 
            ha='center', va='center', 
            fontsize=28, color='#333333', weight='bold')
    ax.text(0.5, 0.3, "similarity", 
            ha='center', va='center', 
            fontsize=12, color='#666666')
    
    ax.axis('off')
    plt.tight_layout()
    return fig

def analyze_documents(files):
    if model is None:
        st.error("Model not loaded. Please check installation.")
        return None
        
    documents = {}
    for file in files:
        text = extract_text(file)
        if text:
            documents[file.name] = text
    
    if len(documents) < 2:
        st.error("Need at least 2 valid documents for comparison")
        return None
        
    filenames = list(documents.keys())
    results = []
    similarity_matrix = np.zeros((len(filenames), len(filenames)))
    
    try:
        embeddings = {name: model.encode(text) for name, text in documents.items()}
        
        for i in range(len(filenames)):
            for j in range(i+1, len(filenames)):
                name1, name2 = filenames[i], filenames[j]
                similarity = cosine_similarity(
                    [embeddings[name1]], 
                    [embeddings[name2]]
                )[0][0] * 100
                
                highlighted, explanations = highlight_differences(
                    documents[name1], 
                    documents[name2],
                    similarity  # Pass overall similarity to adjust thresholds
                )
                
                results.append({
                    "file1": name1,
                    "file2": name2,
                    "similarity": similarity,
                    "highlighted": highlighted,
                    "explanations": explanations
                })
                
                similarity_matrix[i,j] = similarity
                similarity_matrix[j,i] = similarity
                
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return None
    
    # Calculate overall stats
    if len(results) > 0:
        overall_avg = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        overall_max = np.max(similarity_matrix)
    else:
        overall_avg = overall_max = 0
    
    return {
        "pairwise_results": results,
        "similarity_matrix": similarity_matrix,
        "filenames": filenames,
        "overall_avg": overall_avg,
        "overall_max": overall_max
    }

def main():
    # LIGHT THEME CSS
    st.markdown("""
    <style>
    .stApp { 
        background-color: #ffffff; 
        color: #333333; 
    }
    .highlight-box { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #dee2e6; 
        margin-bottom: 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        white-space: pre-wrap;
        line-height: 1.8;
        font-size: 14px;
    }
    .explanation-box { 
        background-color: #fff3cd; 
        padding: 12px; 
        border-radius: 6px; 
        margin: 8px 0;
        border-left: 4px solid #ffc107;
        color: #856404;
        font-size: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .plagiarism-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    .summary-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #0d47a1;
        font-weight: bold;
    }
    .threshold-box {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        color: #2e7d32;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔍Plagiarism Detector")
    st.markdown("Upload documents to detect plagiarism with accurate highlighting")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Select 2 or more documents for comparison"
    )

    if uploaded_files and len(uploaded_files) >= 2:
        with st.spinner("Analyzing documents..."):
            analysis = analyze_documents(uploaded_files)
        
        if analysis is None:
            return
            
        st.subheader("📊 Overall Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; text-align: center;">
                <h3 style="color: #6c757d; margin: 0;">Average Similarity</h3>
                <h2 style="color: #495057; margin: 5px 0 0 0; font-size: 24px;">{analysis["overall_avg"]:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; text-align: center;">
                <h3 style="color: #6c757d; margin: 0;">Highest Similarity</h3>
                <h2 style="color: #495057; margin: 5px 0 0 0; font-size: 24px;">{analysis["overall_max"]:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("🔍 Similarity Matrix")
        matrix_df = pd.DataFrame(
            analysis["similarity_matrix"],
            index=analysis["filenames"],
            columns=analysis["filenames"]
        )
        st.dataframe(
            matrix_df.style.background_gradient(cmap="YlOrRd", vmin=0, vmax=100)
            .format("{:.1f}%"),
            height=min(400, 50 + 35 * len(analysis["filenames"]))
        )

        st.subheader("🔬 Detailed Analysis")
        for result in analysis["pairwise_results"]:
            with st.expander(f"{result['file1']} vs {result['file2']} - {result['similarity']:.1f}% similarity"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.pyplot(create_meter(result['similarity']))
                    st.caption(f"Overall semantic similarity")
                
                with col2:
                    st.markdown("### Text Analysis")
                    if result['highlighted']:
                        st.markdown(
                            f'<div class="highlight-box">{result["highlighted"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("No significant text similarities detected")
                    
                    st.markdown("**Color Legend:**")
                    st.markdown("""
                    - <span style="background-color: #ffcccc; padding: 4px; border: 1px solid red; border-radius: 3px;">Red</span>: Potential plagiarism
                    - Normal text: Original content
                    """, unsafe_allow_html=True)
                
                # Show detailed explanations
                if result['explanations']:
                    st.markdown("### 📋 Analysis Results")
                    for i, exp in enumerate(result['explanations']):
                        if i == 0:
                            st.markdown(f'<div class="summary-box">{exp}</div>', unsafe_allow_html=True)
                        elif i == 1:
                            st.markdown(f'<div class="threshold-box">{exp}</div>', unsafe_allow_html=True)
                        elif i == 2:
                            st.markdown(f'<div class="summary-box">{exp}</div>', unsafe_allow_html=True)
                        elif "HIGH PLAGIARISM" in exp:
                            st.markdown(f'<div class="explanation-box plagiarism-box">{exp}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="explanation-box">{exp}</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ No plagiarism detected. Documents appear to be original")
                    
    elif uploaded_files and len(uploaded_files) < 2:
        st.warning("⚠️ Please upload at least 2 documents for comparison")
    else:
        st.info("📁 Upload 2 or more documents to begin analysis")

if __name__ == "__main__":
    main()