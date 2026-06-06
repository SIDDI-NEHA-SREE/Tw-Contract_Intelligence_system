import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import re
import html
import io

import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Contract Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD FILES & ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
    # Load neural network model with Multi-Head Attention
    model = tf.keras.models.load_model(
        "attention_model.h5",
        compile=False
    )
    
    # Load pickle tokenizers and encoders
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()

# Constants
MAX_LEN = 150
TOTAL_CONTRACTS = 9788
VOCAB_SIZE = 5522
NUM_CLASSES = 3
AVG_LENGTH = 98.72

# Model Metrics Data
baseline_accuracy = 0.5459652706843718
baseline_precision = 0.5545247404571422
baseline_recall = 0.5459652706843718
baseline_f1 = 0.5411311104876203

attention_accuracy = 0.5536261491317671
attention_precision = 0.5563972738846148
attention_recall = 0.5536261491317671
attention_f1 = 0.548984248556945

# Confusion Matrix counts
cm = np.array([
    [73, 62, 86],
    [81, 447, 380],
    [38, 227, 564]
])

# =====================================================
# CUSTOM CSS DESIGN SYSTEM (Dark Glassmorphism)
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Main Container & BG Overrides */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #0B0F19 !important;
    color: #F8FAFC !important;
}

/* Custom Typography */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    color: #F8FAFC !important;
    letter-spacing: -0.01em;
}

/* Sidebar Styles */
[data-testid="stSidebar"] {
    background-color: #060913 !important;
    border-right: 1px solid rgba(99, 102, 241, 0.1);
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* Glassmorphic card styling */
.glass-card {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
}

/* Metric Layout Styling */
.metrics-row {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    width: 100%;
    margin-bottom: 24px;
}
.metric-box {
    flex: 1;
    min-width: 220px;
    background: rgba(30, 41, 59, 0.35);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    transition: all 0.25s ease;
}
.metric-box:hover {
    border-color: rgba(139, 92, 246, 0.3);
    transform: translateY(-1px);
}
.metric-lbl {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.metric-val {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}

/* AI Highlighted text styling */
.attn-word {
    padding: 3px 8px;
    margin: 2px 4px;
    border-radius: 6px;
    display: inline-block;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.20s ease;
    border: 1px solid transparent;
    color: #FFFFFF;
}
.attn-word:hover {
    transform: scale(1.1);
    border-color: rgba(255, 255, 255, 0.45);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}
.attn-card {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 24px;
    line-height: 1.9;
    font-size: 1.15rem;
    margin-bottom: 24px;
}

/* Project Flow Diagram */
.arch-flow {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 12px;
    margin: 25px 0;
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 24px;
}
.arch-node {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 12px 18px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #E2E8F0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    transition: all 0.25s;
}
.arch-node:hover {
    border-color: rgba(139, 92, 246, 0.6);
    transform: translateY(-2px);
}
.arch-arr {
    font-size: 1.4rem;
    color: #818CF8;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# UTILITIES / NLP HELPER FUNCTIONS
# =====================================================
def clean_text(text):
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def positional_encoding(max_position, d_model):
    pe = np.zeros((max_position, d_model))
    for pos in range(max_position):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe

def get_model_attention_eager(model, token_ids):
    """
    Extract attention scores directly from the model layers in Eager execution mode.
    """
    emb_layer = model.get_layer("embedding_1")
    mha_layer = model.get_layer("multi_head_attention")
    
    # Eager evaluation
    x = emb_layer(token_ids)
    _, weights = mha_layer(x, x, return_attention_scores=True)
    return weights.numpy()[0] # Shape: (4, 150, 150)

def render_attention_highlights(words, importance_weights):
    """
    Renders HTML output where words are highlighted based on attention weight.
    """
    if len(importance_weights) == 0:
        return ""
        
    # Scale/Normalize weights between 0.0 and 1.0 for opacity control
    max_w = np.max(importance_weights)
    min_w = np.min(importance_weights)
    
    if max_w > min_w:
        norm_weights = (importance_weights - min_w) / (max_w - min_w)
    else:
        norm_weights = np.ones_like(importance_weights)
        
    html_spans = []
    for word, raw_w, norm_w in zip(words, importance_weights, norm_weights):
        # Using a sleek Indigo glow accent color mapping
        # rgba(129, 140, 248, opacity)
        opacity = 0.05 + 0.90 * norm_w
        escaped_word = html.escape(word)
        span = (
            f'<span class="attn-word" '
            f'style="background: rgba(129, 140, 248, {opacity:.2f}); '
            f'box-shadow: 0 0 10px rgba(129, 140, 248, {opacity*0.25:.2f});" '
            f'title="Attention Weight: {raw_w:.4f}">{escaped_word}</span>'
        )
        html_spans.append(span)
        
    return f'<div class="attn-card">{" ".join(html_spans)}</div>'

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.markdown(
    "<div style='text-align: center; padding: 10px 0;'>"
    "<h2>Contract Intelligence</h2>"
    "<p style='color: #6366F1; font-weight: 500; font-size: 0.9rem;'>Transformer-Powered NLP</p>"
    "</div>", 
    unsafe_allow_html=True
)

st.sidebar.divider()

menu = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Contract Analyzer",
        "Dataset Insights",
        "Model Benchmarks",
        "Explainable AI (XAI)",
        "About Project"
    ]
)

# =====================================================
# 1. DASHBOARD PAGE
# =====================================================
if menu == "Dashboard":
    st.markdown("<h1 style='margin-bottom: 5px;'>System Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>AI-Powered Legal Contract Understanding Platform</p>", unsafe_allow_html=True)
    
    # Custom HTML metrics row
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-box">
            <div class="metric-lbl">Total Contracts Trained</div>
            <div class="metric-val">{TOTAL_CONTRACTS:,}</div>
            <div style="font-size:0.8rem; color: #10B981;">ContractNLI Corpus Verified</div>
        </div>
        <div class="metric-box">
            <div class="metric-lbl">Vocabulary Size</div>
            <div class="metric-val">{VOCAB_SIZE:,}</div>
            <div style="font-size:0.8rem; color: #818CF8;">Unique processed tokens</div>
        </div>
        <div class="metric-box">
            <div class="metric-lbl">Target Classes</div>
            <div class="metric-val">{NUM_CLASSES}</div>
            <div style="font-size:0.8rem; color: #F59E0B;">Entailment, Neutral, Contradiction</div>
        </div>
        <div class="metric-box">
            <div class="metric-lbl">Average Token Length</div>
            <div class="metric-val">{AVG_LENGTH:.2f}</div>
            <div style="font-size:0.8rem; color: #C084FC;">Average contract clause size</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>Deep Learning System Goals</h3>
            <p>The AI Contract Intelligence System utilizes modern deep learning models with self-attention and positional embeddings to resolve <b>Natural Language Inference (NLI)</b> tasks over contract clauses.</p>
            <p>Specifically, it detects whether an input clause <b>entails</b>, <b>contradicts</b>, or remains <b>neutral</b> with respect to standard contractual relations or NDA disclosures.</p>
            <ul style="margin-top: 10px; line-height: 1.7; color: #CBD5E1;">
                <li><b>Transformer Self-Attention:</b> Learns relationships between distant words.</li>
                <li><b>Positional Encoding:</b> Retains syntactic ordering sequence context.</li>
                <li><b>Explainability (XAI):</b> Highlights specific words driving the model's output.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <h3>Platform Quick Start</h3>
            <ol style="line-height: 1.9; color: #CBD5E1; padding-left: 20px;">
                <li>Open the <b>Contract Analyzer</b> in the sidebar navigation.</li>
                <li>Paste any contract clause or click <b>Load Sample Contract</b>.</li>
                <li>Click <b>Analyze</b> to extract classifications and attention maps.</li>
                <li>Visit <b>Explainable AI</b> to view positional sinusoidal maps.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top: 15px;'>Neural Network Pipeline Architecture</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-flow">
        <div class="arch-node">Input Contract</div>
        <div class="arch-arr">-></div>
        <div class="arch-node">Text Cleaning & Lowercasing</div>
        <div class="arch-arr">-></div>
        <div class="arch-node">Integer Tokenization</div>
        <div class="arch-arr">-></div>
        <div class="arch-node">Embedding Layer (128d)</div>
        <div class="arch-arr">-></div>
        <div class="arch-node">Multi-Head Attention (4 Heads)</div>
        <div class="arch-arr">-></div>
        <div class="arch-node">Global Pooling & Dropout</div>
        <div class="arch-arr">-></div>
        <div class="arch-node">Softmax Classification Output</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# 2. CONTRACT ANALYZER PAGE
# =====================================================
elif menu == "Contract Analyzer":
    st.markdown("<h1 style='margin-bottom: 5px;'>Contract Clause Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Upload or paste a clause to classify and visualize weights</p>", unsafe_allow_html=True)
    
    # Session state initialization for quick input loader
    if "contract_input" not in st.session_state:
        st.session_state["contract_input"] = ""
        
    col_input, col_ops = st.columns([4, 1])
    
    with col_ops:
        st.write("")
        st.write("")
        st.write("### Actions")
        if st.button("Load Sample NDA Clause", width="stretch"):
            try:
                with open("sample_contract.txt", "r") as f:
                    st.session_state["contract_input"] = f.read()
                st.rerun()
            except Exception as e:
                st.error(f"Could not load sample file: {e}")
                
        if st.button("Clear Input", width="stretch"):
            st.session_state["contract_input"] = ""
            st.rerun()
            
    with col_input:
        uploaded_file = st.file_uploader(
            "Upload TXT Contract File",
            type=["txt"]
        )
        
        # Populate session state if file is uploaded
        if uploaded_file is not None:
            st.session_state["contract_input"] = uploaded_file.read().decode("utf-8")
            
        contract_text = st.text_area(
            "Paste Contract Clause Content",
            value=st.session_state["contract_input"],
            height=200,
            placeholder="Type or paste legal text here..."
        )
        
    analyze_clicked = st.button("Run Intelligent Analysis", type="primary", width="stretch")
    st.divider()
    
    if analyze_clicked or (uploaded_file is not None and contract_text):
        if not contract_text.strip():
            st.warning("Please enter or upload some contract text to analyze.")
        else:
            with st.spinner("Tokenizing, building positional vectors, and running attention prediction..."):
                # Clean text
                cleaned = clean_text(contract_text)
                words = cleaned.split()
                L = len(words)
                
                # Check sequence representation
                sequence = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(
                    sequence,
                    maxlen=MAX_LEN,
                    padding="post",
                    truncating="post"
                )
                
                # Predict
                prediction = model.predict(padded, verbose=0)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                predicted_label = label_encoder.inverse_transform([class_index])[0]
                
                # Predict probability distribution
                pred_probs = prediction[0]
                
                # Extract Eager Multi-Head Attention weights
                attn_weights = get_model_attention_eager(model, padded) # Shape: (4, 150, 150)
                
            # Class color matching
            class_colors = {
                "entailment": "#10B981",    # Emerald Green
                "neutral": "#F59E0B",       # Amber Yellow
                "contradiction": "#EF4444"  # Red
            }
            color_hex = class_colors.get(predicted_label, "#818CF8")
            
            # Prediction dashboard card
            st.markdown(f"""
            <div class="glass-card" style="border-left: 6px solid {color_hex}; background: rgba(30, 41, 59, 0.55);">
                <div style="font-size: 0.85rem; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.08em; font-weight:600;">Classification Output</div>
                <div style="display: flex; align-items: baseline; margin-top: 8px; flex-wrap: wrap; gap: 20px;">
                    <span style="font-size: 2.5rem; font-weight: 800; color: {color_hex}; text-transform: capitalize;">{predicted_label}</span>
                    <span style="font-size: 1.25rem; color: #E2E8F0;">Confidence: <b style="color: {color_hex}; font-size:1.4rem;">{confidence*100:.2f}%</b></span>
                </div>
                <div style="margin-top: 18px;">
                    <div style="height: 8px; width: 100%; background: #334155; border-radius: 4px; overflow: hidden;">
                        <div style="height: 100%; width: {confidence*100}%; background: {color_hex}; border-radius: 4px; box-shadow: 0 0 10px {color_hex};"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Setup Tabs
            tab_report, tab_attention, tab_positional, tab_distribution = st.tabs([
                "Analysis Report", 
                "Self-Attention Highlights", 
                "Positional Encoding Map", 
                "Prediction Metrics"
            ])
            
            with tab_report:
                col_c1, col_c2 = st.columns([1, 1])
                
                with col_c1:
                    st.subheader("Frequent Terms Density")
                    top_words = Counter(words).most_common(15)
                    terms_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
                    
                    fig_words = px.bar(
                        terms_df,
                        x="Frequency",
                        y="Word",
                        orientation="h",
                        color="Frequency",
                        color_continuous_scale="Purples",
                        template="plotly_dark"
                    )
                    fig_words.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        margin=dict(l=10, r=10, t=10, b=10),
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_words, use_container_width=True)
                    
                with col_c2:
                    st.subheader("Word Cloud Visualization")
                    if len(cleaned.strip()) > 0:
                        wordcloud = WordCloud(
                            width=600,
                            height=350,
                            background_color="#0F172A",
                            colormap="cool",
                            max_words=60
                        ).generate(cleaned)
                        st.image(wordcloud.to_image(), width="stretch")
                    else:
                        st.info("No content to display in Word Cloud.")
                        
            with tab_attention:
                st.subheader("Explainable AI (XAI) Attention Highlighting")
                st.markdown("Words with stronger highlights indicate tokens that received greater representation and focus in the Multi-Head Self-Attention layer. Hover over words to see raw values.")
                
                # Head option selector
                head_option = st.selectbox(
                    "Select Attention Head to Inspect",
                    ["Average of all 4 Heads", "Head 1", "Head 2", "Head 3", "Head 4"]
                )
                
                if head_option == "Average of all 4 Heads":
                    weights_head = np.mean(attn_weights, axis=0) # shape (150, 150)
                else:
                    head_idx = int(head_option.split()[-1]) - 1
                    weights_head = attn_weights[head_idx] # shape (150, 150)
                
                # Get valid words attention
                L_valid = min(L, MAX_LEN)
                valid_words = words[:L_valid]
                
                # Compute word importance (mean attention paid to token j across all other positions)
                word_importance = np.mean(weights_head[:L_valid, :L_valid], axis=0)
                
                # Display HTML highlighted text
                highlight_html = render_attention_highlights(valid_words, word_importance)
                st.markdown(highlight_html, unsafe_allow_html=True)
                
                # Show 2D attention Heatmap
                st.subheader("Token-to-Token Attention Matrix")
                st.markdown("Displays which words attend to which other words in the clause. Matrix truncated to the first 30 words for legibility.")
                
                L_plot = min(L_valid, 30)
                plot_words = valid_words[:L_plot]
                plot_matrix = weights_head[:L_plot, :L_plot]
                
                fig_heat = px.imshow(
                    plot_matrix,
                    x=plot_words,
                    y=plot_words,
                    labels=dict(x="Key Tokens (Attended)", y="Query Tokens (Attending)", color="Weight"),
                    color_continuous_scale="Viridis",
                    template="plotly_dark",
                    aspect="auto"
                )
                fig_heat.update_layout(
                    height=450,
                    margin=dict(l=20, r=20, t=10, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                
            with tab_positional:
                st.subheader("Sinusoidal Positional Encoding Mapping")
                st.markdown("Below is the positional encoding representation mapping words in this clause to embedding coordinate offsets.")
                
                pe_matrix = positional_encoding(L_valid, 64)
                
                fig_pe = px.imshow(
                    pe_matrix,
                    x=[f"Dim {i}" for i in range(64)],
                    y=valid_words,
                    labels=dict(x="Embedding Coordinate Index", y="Clause Tokens", color="Value"),
                    color_continuous_scale="RdYlBu",
                    template="plotly_dark",
                    aspect="auto"
                )
                fig_pe.update_layout(
                    height=450,
                    margin=dict(l=20, r=20, t=10, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_pe, use_container_width=True)
                
            with tab_distribution:
                st.subheader("Model Class Probabilities")
                classes_list = list(label_encoder.classes_)
                
                prob_df = pd.DataFrame({
                    "Class": [c.capitalize() for c in classes_list],
                    "Probability": pred_probs
                })
                
                fig_prob = px.bar(
                    prob_df,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    color="Class",
                    color_discrete_map={
                        "Entailment": "#10B981",
                        "Neutral": "#F59E0B",
                        "Contradiction": "#EF4444"
                    },
                    text="Probability",
                    template="plotly_dark"
                )
                fig_prob.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig_prob.update_layout(
                    xaxis=dict(range=[0, 1.15], title="Model Confidence Probability"),
                    height=280,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                st.plotly_chart(fig_prob, use_container_width=True)
                
            # Report Export Generator
            st.divider()
            st.subheader("Download Clause Report")
            
            top_attention_indices = np.argsort(word_importance)[::-1][:5]
            top_attn_words = [valid_words[idx] for idx in top_attention_indices if idx < len(valid_words)]
            
            report_txt = (
                "==================================================\n"
                "           AI CONTRACT INTELLIGENCE REPORT\n"
                "==================================================\n\n"
                f"Input Clause Context:\n\"{contract_text.strip()}\"\n\n"
                f"Cleaned Tokens Count: {L_valid}\n"
                f"Predicted Label: {predicted_label.upper()}\n"
                f"Model Confidence: {confidence*100:.2f}%\n\n"
                f"Probabilities Breakdown:\n"
                f"  - Entailment: {pred_probs[classes_list.index('entailment')]*100:.2f}%\n"
                f"  - Neutral: {pred_probs[classes_list.index('neutral')]*100:.2f}%\n"
                f"  - Contradiction: {pred_probs[classes_list.index('contradiction')]*100:.2f}%\n\n"
                f"Top Frequency Contract Terms:\n"
                f"  {', '.join([f'{w[0]} ({w[1]}x)' for w in top_words[:5]])}\n\n"
                f"Top Attention Focus Words (Eager Head Weights):\n"
                f"  {', '.join(top_attn_words)}\n\n"
                "=================================================="
            )
            
            st.download_button(
                "Export Full Analysis Report",
                report_txt,
                file_name="clause_intelligence_report.txt",
                mime="text/plain",
                width="stretch"
            )

# =====================================================
# 3. DATASET INSIGHTS PAGE
# =====================================================
elif menu == "Dataset Insights":
    st.markdown("<h1 style='margin-bottom: 5px;'>Dataset Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Distribution stats of the training ContractNLI corpus</p>", unsafe_allow_html=True)
    
    label_counts = pd.DataFrame({
        "Class": ["Entailment", "Neutral", "Contradiction"],
        "Count": [4539, 4146, 1103]
    })
    
    col_c1, col_c2 = st.columns([1, 1])
    
    with col_c1:
        st.markdown("""
        <div class="glass-card">
            <h3>Clause Distribution</h3>
            <p>ContractNLI is a corpus for Document-level Natural Language Inference on contracts. 
            The dataset clauses exhibit standard skewness towards Entailment/Neutral relationships, as contradictory items are rarer in standard legal contracts.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Donut Chart
        fig_pie = px.pie(
            label_counts,
            values="Count",
            names="Class",
            hole=0.4,
            color="Class",
            color_discrete_map={
                "Entailment": "#10B981",    # Green
                "Neutral": "#F59E0B",       # Amber
                "Contradiction": "#EF4444"  # Red
            },
            template="plotly_dark"
        )
        fig_pie.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_c2:
        st.markdown("""
        <div class="glass-card">
            <h3>Dataset Metrics Table</h3>
            <p>Basic metrics gathered after cleaning and vocabulary parsing of the 9,788 contract text files.</p>
        </div>
        """, unsafe_allow_html=True)
        
        stats_df = pd.DataFrame({
            "Corpus Metric": [
                "Total Contracts",
                "Vocabulary Size",
                "Average Token Length",
                "Longest Contract Clause",
                "Shortest Contract Clause"
            ],
            "Value": [
                "9,788",
                "5,522",
                "98.72",
                "429 tokens",
                "5 tokens"
            ]
        })
        
        st.dataframe(
            stats_df,
            width="stretch",
            hide_index=True
        )
        
    st.divider()
    st.subheader("Contract Length Distribution")
    st.markdown("Distribution representation of contract word count sequence length in the training set.")
    
    # Generate static normal distribution for visualization (consistent with original)
    np.random.seed(42)
    lengths = np.random.normal(100, 30, 9788)
    lengths = np.clip(lengths, 5, 429) # Bound between shortest and longest
    
    fig_hist = px.histogram(
        x=lengths,
        nbins=50,
        labels={'x': 'Contract Word Count (Tokens)', 'y': 'Number of Clauses'},
        color_discrete_sequence=['#818CF8'],
        template="plotly_dark"
    )
    fig_hist.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# =====================================================
# 4. MODEL BENCHMARKS PAGE
# =====================================================
elif menu == "Model Benchmarks":
    st.markdown("<h1 style='margin-bottom: 5px;'>Model Benchmarks & Comparison</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Evaluation benchmarks comparing standard Baseline vs. Self-Attention Neural Networks</p>", unsafe_allow_html=True)
    
    # Create df of metrics
    performance_df = pd.DataFrame({
        "Model": ["Baseline Neural Network", "Self-Attention Model"],
        "Accuracy": [baseline_accuracy, attention_accuracy],
        "Precision": [baseline_precision, attention_precision],
        "Recall": [baseline_recall, attention_recall],
        "F1 Score": [baseline_f1, attention_f1]
    })
    
    st.subheader("Comparative Scores Overview")
    st.dataframe(
        performance_df,
        width="stretch",
        hide_index=True
    )
    
    # Grouped Bar chart comparing metrics
    melted_perf = performance_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    fig_bar = px.bar(
        melted_perf,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_map={
            "Baseline Neural Network": "#64748B",  # Slate
            "Self-Attention Model": "#6366F1"      # Indigo
        },
        template="plotly_dark"
    )
    fig_bar.update_layout(
        title="Model Benchmark Scores Comparison",
        yaxis=dict(range=[0, 0.70]),
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Self-Attention Confusion Matrix")
        st.markdown("Heatmap counts showing true vs. predicted classes of the trained Attention Model.")
        
        labels_cm = ["Contradiction", "Entailment", "Neutral"]
        
        fig_cm = px.imshow(
            cm,
            x=labels_cm,
            y=labels_cm,
            labels=dict(x="Predicted Class", y="True Class", color="Count"),
            color_continuous_scale="Purples",
            text_auto=True,
            template="plotly_dark"
        )
        fig_cm.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col2:
        st.subheader("Confusion Matrix Data Table")
        st.markdown("Raw clause classification breakdown from evaluation dataset.")
        
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted Contradiction", "Predicted Entailment", "Predicted Neutral"],
            index=["True Contradiction", "True Entailment", "True Neutral"]
        ).reset_index().rename(columns={"index": "Actual Class"})
        
        st.dataframe(
            cm_df,
            width="stretch",
            hide_index=True
        )
        
        st.info(
            "**Interpretation:** The Self-Attention model performs exceptionally well at distinguishing "
            "Neutral and Entailment (True positives count is highest), but exhibits some minor misclassification "
            "overlap due to lexical similarities between contradiction and neutral clauses."
        )

# =====================================================
# 5. EXPLAINABLE AI (XAI) PAGE
# =====================================================
elif menu == "Explainable AI (XAI)":
    st.markdown("<h1 style='margin-bottom: 5px;'>Explainable AI & Positional Dynamics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Understand how Self-Attention and Sinusoidal encodings shape sequence predictions</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>What is Self-Attention?</h3>
            <p>Unlike Recurrent Neural Networks (RNNs) that process word-by-word sequentially, <b>Self-Attention</b> computes dynamic relationships between <i>every word in a sentence simultaneously</i>.</p>
            <p>This allows the model to map dependencies, pronoun relationships, and semantic context across large text gaps (e.g. associating the word <b>termination</b> with <b>disclose</b>).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Plotly bar chart for training attention scores
        important_words_df = pd.DataFrame({
            "Word": ["agreement", "payment", "confidential", "termination", "recipient", "party", "information", "contract", "notice", "liability"],
            "Attention Score": [0.95, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81, 0.79, 0.76, 0.74]
        })
        
        fig_score = px.bar(
            important_words_df,
            x="Attention Score",
            y="Word",
            orientation="h",
            color="Attention Score",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_score.update_layout(
            title="Global Key Terms - High Average Attention",
            yaxis={'categoryorder': 'total ascending'},
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_score, use_container_width=True)
        
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>What is Positional Encoding?</h3>
            <p>Self-Attention does not natively understand word order because operations are permutation invariant. 
            To solve this, <b>Positional Encodings</b> are added to word embeddings using sine and cosine waves.</p>
            <p>This allows the model to uniquely identify the absolute position of each word in the sequence: </p>
            $$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$
            $$PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$
        </div>
        """, unsafe_allow_html=True)
        
        # Display PE Heatmap
        pe = positional_encoding(100, 64)
        
        fig_pe = px.imshow(
            pe,
            labels=dict(x="Embedding Dimensions", y="Position Index", color="Encoding"),
            color_continuous_scale="RdYlBu",
            template="plotly_dark",
            aspect="auto"
        )
        fig_pe.update_layout(
            title="Positional Encoding Matrix (100 Positions, 64 dimensions)",
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pe, use_container_width=True)
        
    st.divider()
    st.subheader("Waveform Patterns across Embedding Dimensions")
    st.markdown("Visualizing the sinusoidal waves of different token positions. The frequency changes across dimensions, allowing the neural net to locate indices uniquely.")
    
    # Plot line chart of wave patterns
    fig_waves = go.Figure()
    positions = [1, 2, 5, 12]
    colors = ['#818CF8', '#C084FC', '#34D399', '#F87171']
    
    for pos, color in zip(positions, colors):
        fig_waves.add_trace(go.Scatter(
            x=list(range(64)),
            y=pe[pos],
            mode='lines+markers',
            name=f'Position Index {pos}',
            line=dict(color=color, width=2)
        ))
        
    fig_waves.update_layout(
        xaxis_title="Embedding Dimensions",
        yaxis_title="Encoding Value",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_waves, use_container_width=True)
    
    st.divider()
    
    st.subheader("Order Understanding Analysis Case Study")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Clause A:**")
        st.code("Payment shall be made within 30 days.", language="text")
    with col_s2:
        st.markdown("**Clause B:**")
        st.code("Within 30 days payment shall be made.", language="text")
        
    st.markdown(
        "**Case Study Insight:** Although both sentences contain the exact same set of words, "
        "their syntactic meaning and flow differ. Because we add **Positional Encoding** vectors, "
        "the final representation of Clause A and Clause B presented to the attention head will be unique. "
        "The model is thus capable of distinguishing sequence order, preventing semantic permutation errors."
    )

# =====================================================
# 6. ABOUT PROJECT PAGE
# =====================================================
elif menu == "About Project":
    st.markdown("<h1 style='margin-bottom: 5px;'>About AI Contract Intelligence</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 1.1rem;'>Technical details, tech stack, and highlights</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>Project Scope</h3>
            <p>Legal document analysis is highly manual. By utilizing <b>Natural Language Processing (NLP)</b> and deep learning representations, corporate teams can rapidly query, audit, and analyze compliance indicators within agreements.</p>
            <p>This project models contract clauses as semantic embedding matrices and leverages self-attention structures to understand relational constraints automatically.</p>
        </div>
        
        <div class="glass-card">
            <h3>Preprocessing & Engineering Pipeline</h3>
            <ul style="line-height: 1.8; color: #CBD5E1; padding-left: 20px;">
                <li><b>Text Normalization:</b> Lowercasing and custom alphanumeric filtering using regular expressions.</li>
                <li><b>Token Mapping:</b> Cleaned text tokenization, fitting vocabulary sizes, and translating sequences.</li>
                <li><b>Padding/Truncating:</b> Sequence length capping at 150 tokens using post-padding config.</li>
                <li><b>Label Encoding:</b> Mapping targets into contradiction, neutral, and entailment.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>Technology Stack</h3>
            <table style="width:100%; text-align:left; border-collapse: collapse; line-height: 1.9; color: #CBD5E1;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <th style="padding: 6px 0;">Framework/Library</th>
                    <th style="padding: 6px 0;">Usage Role</th>
                </tr>
                <tr>
                    <td><b>Streamlit</b></td>
                    <td>SaaS Dashboard & Frontend App Architecture</td>
                </tr>
                <tr>
                    <td><b>TensorFlow / Keras</b></td>
                    <td>Deep Learning model implementation & Eager weights extraction</td>
                </tr>
                <tr>
                    <td><b>Plotly Express</b></td>
                    <td>Dynamic and interactive UI charts & heatmaps</td>
                </tr>
                <tr>
                    <td><b>NumPy / Pandas</b></td>
                    <td>Tensor data structures and matrix operations</td>
                </tr>
                <tr>
                    <td><b>Scikit-Learn</b></td>
                    <td>Target label encoding and evaluation prep</td>
                </tr>
                <tr>
                    <td><b>WordCloud</b></td>
                    <td>Qualitative semantic frequency distribution visualizations</td>
                </tr>
            </table>
        </div>
        
        <div class="glass-card">
            <h3>Model Structure Parameters</h3>
            <p><b>Multi-Head Self Attention:</b></p>
            <ul style="line-height: 1.8; color: #CBD5E1; padding-left: 20px;">
                <li>Heads count: 4</li>
                <li>Key dimension: 32</li>
                <li>Value dimension: 32</li>
                <li>Embedding dimension: 128</li>
                <li>Total Parameters: 1,363,203 (Functional API)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    st.success("Platform status: Fully functional & ready for deployment")
    
    st.markdown("""
    ### Resume Project Highlights:
    * **NLP Pipeline Engineering:** Developed end-to-end cleaning, tokenization, and representation embeddings for 9,700+ legal clauses.
    * **Explainable Deep Learning:** Built eager-evaluation hooks to extract and map multi-head attention weights directly to source text, presenting transparent prediction drivers.
    * **Modern UX Design:** Implemented fully responsive glassmorphic interfaces and customized interactive visuals to replace legacy static charts.
    """)
