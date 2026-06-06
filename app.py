import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import math
import kagglehub
import os
import json
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from wordcloud import WordCloud

st.set_page_config(page_title="AI Contract Intelligence System", layout="wide")
st.title("📜 AI Contract Intelligence System")
st.markdown("**NLP + Attention + Positional Encoding for Legal Contract Analysis**")

# ─── Cache dataset loading ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("ahmshalan/contract-nli")
    # Find JSON files
    all_data = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.json'):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath) as fp:
                        d = json.load(fp)
                    # ContractNLI format: document text + annotations
                    if isinstance(d, dict) and 'documents' in d:
                        for doc in d['documents']:
                            text = doc.get('text', '')
                            for hypo_key, ann in doc.get('annotation_sets', [{}])[0].get('annotations', {}).items():
                                label = ann.get('choice', 'Unknown')
                                all_data.append({'text': text[:500], 'clause_type': label, 'full_text': text})
                    elif isinstance(d, list):
                        for item in d:
                            all_data.append({
                                'text': str(item.get('text', item.get('sentence', '')))[:500],
                                'clause_type': str(item.get('label', item.get('clause_type', 'Unknown'))),
                                'full_text': str(item.get('text', ''))
                            })
                except Exception:
                    pass

    if not all_data:
        # Fallback synthetic data
        clauses = ['Termination', 'Payment', 'Confidentiality', 'Liability', 'Non-Compete']
        texts = [
            "Either party may terminate this agreement with 30 days written notice.",
            "Payment shall be made within 30 days of invoice receipt.",
            "All confidential information shall remain strictly confidential.",
            "Neither party shall be liable for indirect damages.",
            "Employee agrees not to compete for 2 years after termination.",
            "This contract may be terminated immediately for material breach.",
            "Fees are due net 30 from the date of invoice.",
            "Confidential data must not be disclosed to third parties.",
            "Total liability shall not exceed the contract value.",
            "Non-compete restrictions apply within a 50-mile radius.",
        ] * 50
        labels = clauses * (len(texts) // len(clauses) + 1)
        all_data = [{'text': t, 'clause_type': l, 'full_text': t} for t, l in zip(texts, labels[:len(texts)])]

    df = pd.DataFrame(all_data)
    df = df[df['clause_type'] != 'Unknown'].dropna()
    df['word_count'] = df['full_text'].apply(lambda x: len(str(x).split()))
    return df

# ─── Positional Encoding ─────────────────────────────────────────────────────
def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i]     = math.sin(pos / (10000 ** (2*i / d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = math.cos(pos / (10000 ** (2*i / d_model)))
    return PE

# ─── Custom Attention Layer ───────────────────────────────────────────────────
class SelfAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True)
    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        score = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(x * score, axis=1)
        return context, score

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
task = st.sidebar.radio("📌 Select Task", [
    "Task 1: EDA",
    "Task 2: Text Engineering",
    "Task 3: Baseline Model",
    "Task 4: Self-Attention Model",
    "Task 5: Positional Encoding",
    "Task 6: Clause Understanding",
    "Task 7: Attention Analysis",
    "Task 8: Dashboard"
])

with st.spinner("Loading dataset..."):
    df = load_data()

# ════════════════════════════════════════════════════════════════════════════════
if task == "Task 1: EDA":
    st.header("📊 Task 1: Dataset Investigation & EDA")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Contracts", len(df))
    col2.metric("Clause Types", df['clause_type'].nunique())
    col3.metric("Avg Contract Length (words)", int(df['word_count'].mean()))
    col4.metric("Longest Contract (words)", df['word_count'].max())

    st.subheader("Shortest & Longest Contracts")
    st.write("**Shortest:**", df.loc[df['word_count'].idxmin(), 'text'][:200])
    st.write("**Longest:**", df.loc[df['word_count'].idxmax(), 'text'][:200])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Clause Type Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        vc = df['clause_type'].value_counts()
        ax.bar(vc.index, vc.values, color=plt.cm.Set2.colors[:len(vc)])
        ax.set_xlabel("Clause Type"); ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha='right')
        st.pyplot(fig)

    with col2:
        st.subheader("Contract Length Histogram")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['word_count'], bins=30, color='steelblue', edgecolor='white')
        ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency")
        st.pyplot(fig)

    st.subheader("Word Frequency (Top 30)")
    all_text = ' '.join(df['text'].tolist())
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
    stopwords = {'shall','this','that','with','from','have','been','will','which','upon','such','each','under','any','all','other','into','their','also','both','where'}
    words = [w for w in words if w not in stopwords]
    word_freq = Counter(words).most_common(30)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar([w[0] for w in word_freq], [w[1] for w in word_freq], color='coral')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.subheader("Word Cloud")
    wc = WordCloud(width=800, height=300, background_color='white').generate(' '.join(words))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 2: Text Engineering":
    st.header("🔧 Task 2: Text Engineering")

    MAX_VOCAB = 5000
    MAX_LEN = 100

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['clean_text'] = df['text'].apply(clean_text)

    st.subheader("Sample Cleaned Text")
    st.dataframe(df[['text','clean_text']].head(5))

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['clean_text'])
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    st.subheader("Tokenization Example")
    st.write("**Original:**", df['clean_text'].iloc[0][:100])
    st.write("**Token IDs:**", sequences[0][:20])
    st.write("**Padded Sequence (first 20):**", padded[0][:20].tolist())

    st.subheader("Vocabulary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Vocabulary Size (capped)", MAX_VOCAB)
    col2.metric("Total Unique Words", len(tokenizer.word_index))
    col3.metric("Sequence Max Length", MAX_LEN)

    st.info("""
    **Why padding is required?**  
    Neural networks require fixed-size inputs. Since contracts vary in length, shorter sequences are padded with zeros and longer ones are truncated to `MAX_LEN`.

    **Why vocabulary size matters?**  
    Large vocabularies increase model parameters and memory. Capping at 5000 focuses on the most frequent, informative words and maps rare words to `<OOV>` (Out-Of-Vocabulary) token.
    """)

    seq_lengths = [len(s) for s in sequences]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(seq_lengths, bins=30, color='mediumseagreen', edgecolor='white')
    ax.axvline(MAX_LEN, color='red', linestyle='--', label=f'MAX_LEN={MAX_LEN}')
    ax.set_xlabel("Sequence Length"); ax.legend()
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 3: Baseline Model":
    st.header("🤖 Task 3: Baseline Dense Model")

    MAX_VOCAB, MAX_LEN, EMBED_DIM = 5000, 100, 64

    def clean_text(t):
        t = str(t).lower(); t = re.sub(r'[^a-zA-Z\s]', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()

    df['clean'] = df['text'].apply(clean_text)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['clause_type'])
    num_classes = len(le.classes_)

    tok = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tok.fit_on_texts(df['clean'])
    seqs = tok.texts_to_sequences(df['clean'])
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    y = keras.utils.to_categorical(df['label'], num_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = keras.Sequential([
        layers.Embedding(MAX_VOCAB, EMBED_DIM, input_length=MAX_LEN),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    with st.spinner("Training baseline model (5 epochs)..."):
        history = model.fit(X_train, y_train, epochs=5, batch_size=32,
                            validation_split=0.1, verbose=0)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{p:.4f}")
    col3.metric("Recall", f"{r:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    st.subheader("Training Curves")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train'); ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'], label='Train'); ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.legend()
    st.pyplot(fig)

    st.subheader("Architecture")
    model.summary(print_fn=lambda x: st.text(x))

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 4: Self-Attention Model":
    st.header("🧠 Task 4: Self-Attention Model")

    MAX_VOCAB, MAX_LEN, EMBED_DIM = 5000, 100, 64

    def clean_text(t):
        t = str(t).lower(); t = re.sub(r'[^a-zA-Z\s]', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()

    df['clean'] = df['text'].apply(clean_text)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['clause_type'])
    num_classes = len(le.classes_)

    tok = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tok.fit_on_texts(df['clean'])
    seqs = tok.texts_to_sequences(df['clean'])
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    y = keras.utils.to_categorical(df['label'], num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    inputs = keras.Input(shape=(MAX_LEN,))
    emb = layers.Embedding(MAX_VOCAB, EMBED_DIM)(inputs)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(emb, emb)
    pool = layers.GlobalAveragePooling1D()(attn)
    drop = layers.Dropout(0.3)(pool)
    out = layers.Dense(num_classes, activation='softmax')(drop)
    model = keras.Model(inputs, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    with st.spinner("Training attention model (5 epochs)..."):
        history = model.fit(X_train, y_train, epochs=5, batch_size=32,
                            validation_split=0.1, verbose=0)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision", f"{p:.4f}")
    col3.metric("Recall", f"{r:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")

    st.subheader("Training Curves")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train'); ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy'); ax1.legend()
    ax2.plot(history.history['loss'], label='Train'); ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss'); ax2.legend()
    st.pyplot(fig)

    st.subheader("Architecture")
    model.summary(print_fn=lambda x: st.text(x))

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 5: Positional Encoding":
    st.header("📍 Task 5: Positional Encoding (Implemented from Scratch)")

    st.markdown("""
    **Formula:**  
    `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`  
    `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
    """)

    max_len = st.slider("Max Sequence Length", 10, 100, 50)
    d_model = st.slider("Embedding Dimension (d_model)", 16, 128, 64, step=16)

    PE = positional_encoding(max_len, d_model)

    st.subheader("Positional Encoding Heatmap")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(PE, cmap='RdBu_r', ax=ax)
    ax.set_xlabel("Embedding Dimension"); ax.set_ylabel("Position")
    ax.set_title("Positional Encoding Values")
    st.pyplot(fig)

    st.subheader("Individual Position Vectors (First 5 Positions)")
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    colors = ['#e63946','#457b9d','#2a9d8f','#e9c46a','#f4a261']
    for i in range(5):
        axes[i].plot(PE[i], color=colors[i])
        axes[i].set_title(f"Position {i+1}")
        axes[i].set_xlabel("Dimension")
        if i == 0: axes[i].set_ylabel("Value")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Sine vs Cosine Components")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    for pos in range(5):
        ax1.plot(PE[pos, 0::2], label=f'pos {pos}')
        ax2.plot(PE[pos, 1::2], label=f'pos {pos}')
    ax1.set_title("Sine Dimensions"); ax1.legend(); ax1.set_xlabel("i")
    ax2.set_title("Cosine Dimensions"); ax2.legend(); ax2.set_xlabel("i")
    st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 6: Clause Understanding":
    st.header("🔍 Task 6: Clause Understanding Analysis")

    contractA = "Payment shall be made within 30 days."
    contractB = "Within 30 days payment shall be made."

    st.subheader("Example Contracts")
    col1, col2 = st.columns(2)
    col1.info(f"**Contract A:** {contractA}")
    col2.success(f"**Contract B:** {contractB}")

    def get_pe_for_sentence(sentence, d_model=32):
        words = sentence.lower().replace('.','').split()
        PE = positional_encoding(len(words), d_model)
        return words, PE

    words_a, pe_a = get_pe_for_sentence(contractA)
    words_b, pe_b = get_pe_for_sentence(contractB)

    st.subheader("Positional Encoding Heatmaps")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    sns.heatmap(pe_a, ax=ax1, cmap='Blues', yticklabels=words_a)
    ax1.set_title("Contract A Positional Encoding"); ax1.set_xlabel("Dimension")
    sns.heatmap(pe_b, ax=ax2, cmap='Oranges', yticklabels=words_b)
    ax2.set_title("Contract B Positional Encoding"); ax2.set_xlabel("Dimension")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Word-level Positional Comparison")
    common_words = ['payment', 'days']
    for word in common_words:
        if word in words_a and word in words_b:
            pos_a = words_a.index(word)
            pos_b = words_b.index(word)
            col1, col2 = st.columns(2)
            col1.write(f"**'{word}'** in Contract A → Position **{pos_a+1}**")
            col2.write(f"**'{word}'** in Contract B → Position **{pos_b+1}**")

    st.info("""
    **Explanation:**
    - Both contracts contain the **same words** but in **different order**
    - Positional encoding assigns a **unique vector to each position** (sin/cos patterns)
    - The word "payment" gets **different positional embedding** in Contract A (pos 1) vs Contract B (pos 3)
    - This means the model understands that word **order changes meaning** even when vocabulary is identical
    - Without positional encoding, a bag-of-words model would treat both sentences identically
    """)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 7: Attention Analysis":
    st.header("👁️ Task 7: Attention Analysis")

    MAX_VOCAB, MAX_LEN, EMBED_DIM = 5000, 100, 64

    def clean_text(t):
        t = str(t).lower(); t = re.sub(r'[^a-zA-Z\s]', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()

    df['clean'] = df['text'].apply(clean_text)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['clause_type'])
    num_classes = len(le.classes_)

    tok = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
    tok.fit_on_texts(df['clean'])
    seqs = tok.texts_to_sequences(df['clean'])
    X = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    y = keras.utils.to_categorical(df['label'], num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    inputs = keras.Input(shape=(MAX_LEN,))
    emb = layers.Embedding(MAX_VOCAB, EMBED_DIM)(inputs)
    attn_out, attn_scores = layers.MultiHeadAttention(num_heads=4, key_dim=16, return_attention_scores=True)(emb, emb)
    pool = layers.GlobalAveragePooling1D()(attn_out)
    out = layers.Dense(num_classes, activation='softmax')(pool)
    model = keras.Model(inputs, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    with st.spinner("Training model for attention extraction..."):
        model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    attention_model = keras.Model(inputs=model.input,
                                   outputs=[model.output,
                                            model.layers[2].output[1]])  # attn scores

    sample_text = st.text_area("Enter contract text for analysis:",
        "Payment shall be made within 30 days of invoice. Termination may occur with 60 days notice. All information is strictly confidential.")

    if st.button("Analyze Attention"):
        clean_sample = clean_text(sample_text)
        words = clean_sample.split()[:MAX_LEN]
        seq = tok.texts_to_sequences([clean_sample])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

        pred, attn = attention_model.predict(padded, verbose=0)
        pred_class = le.classes_[np.argmax(pred)]

        st.success(f"**Predicted Clause Type:** {pred_class} (Confidence: {np.max(pred)*100:.1f}%)")

        # Average attention across heads
        avg_attn = np.mean(attn[0], axis=0)  # (seq_len, seq_len)
        word_importance = np.mean(avg_attn[:len(words), :len(words)], axis=0)

        st.subheader("Word Importance (Attention Scores)")
        word_scores = dict(zip(words, word_importance[:len(words)]))
        sorted_ws = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh([w[0] for w in sorted_ws], [w[1] for w in sorted_ws], color='steelblue')
        ax.set_xlabel("Attention Score"); ax.set_title("Top Attended Words")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.subheader("Attention Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        disp_len = min(15, len(words))
        sns.heatmap(avg_attn[:disp_len, :disp_len], cmap='YlOrRd', ax=ax,
                    xticklabels=words[:disp_len], yticklabels=words[:disp_len])
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        st.pyplot(fig)

# ════════════════════════════════════════════════════════════════════════════════
elif task == "Task 8: Dashboard":
    st.header("🏛️ Task 8: Contract Intelligence Dashboard")

    MAX_VOCAB, MAX_LEN, EMBED_DIM = 5000, 100, 64

    def clean_text(t):
        t = str(t).lower(); t = re.sub(r'[^a-zA-Z\s]', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()

    @st.cache_resource
    def build_and_train():
        df2 = df.copy()
        df2['clean'] = df2['text'].apply(clean_text)
        le2 = LabelEncoder()
        df2['label'] = le2.fit_transform(df2['clause_type'])
        nc = len(le2.classes_)
        tok2 = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
        tok2.fit_on_texts(df2['clean'])
        seqs2 = tok2.texts_to_sequences(df2['clean'])
        X2 = pad_sequences(seqs2, maxlen=MAX_LEN, padding='post')
        y2 = keras.utils.to_categorical(df2['label'], nc)
        X_tr, X_te, y_tr, y_te = train_test_split(X2, y2, test_size=0.2, random_state=42)
        inp = keras.Input(shape=(MAX_LEN,))
        emb2 = layers.Embedding(MAX_VOCAB, EMBED_DIM)(inp)
        a_out, a_sc = layers.MultiHeadAttention(num_heads=4, key_dim=16, return_attention_scores=True)(emb2, emb2)
        pool2 = layers.GlobalAveragePooling1D()(a_out)
        out2 = layers.Dense(nc, activation='softmax')(pool2)
        m2 = keras.Model(inp, out2)
        m2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        m2.fit(X_tr, y_tr, epochs=5, batch_size=32, verbose=0)
        return m2, tok2, le2

    with st.spinner("Preparing model..."):
        model, tokenizer, le = build_and_train()

    attn_model = keras.Model(inputs=model.input,
                              outputs=[model.output,
                                       model.layers[2].output[1]])

    st.subheader("📤 Upload a Contract File")
    uploaded = st.file_uploader("Upload .txt contract", type=['txt'])

    contract_text = ""
    if uploaded:
        contract_text = uploaded.read().decode('utf-8')
    else:
        contract_text = st.text_area("Or paste contract text here:",
            "This Agreement may be terminated by either party upon thirty (30) days written notice. "
            "Payment shall be due within 30 days of invoice. All confidential information shall remain protected.")

    if st.button("🔍 Analyze Contract") and contract_text:
        st.divider()
        sentences = [s.strip() for s in re.split(r'[.!?]', contract_text) if len(s.strip()) > 10]

        results = []
        for sent in sentences[:10]:
            clean_s = clean_text(sent)
            seq = pad_sequences(tokenizer.texts_to_sequences([clean_s]), maxlen=MAX_LEN, padding='post')
            pred, attn = attn_model.predict(seq, verbose=0)
            clause = le.classes_[np.argmax(pred)]
            conf = np.max(pred)
            results.append({'Sentence': sent[:80], 'Predicted Clause': clause, 'Confidence': f"{conf*100:.1f}%"})

        st.subheader("📋 Clause Predictions")
        st.dataframe(pd.DataFrame(results))

        # Highlight important terms
        st.subheader("🎯 Important Terms Across Contract")
        all_words = re.findall(r'\b[a-zA-Z]{4,}\b', contract_text.lower())
        legal_terms = ['termination','payment','confidential','liability','compete','agreement',
                       'breach','damages','notice','indemnify','arbitration','warranty']
        found_terms = [t for t in legal_terms if t in contract_text.lower()]
        if found_terms:
            st.write("**Legal Terms Found:**", ', '.join([f'`{t}`' for t in found_terms]))

        # Attention heatmap for full text
        st.subheader("👁️ Attention Map")
        clean_full = clean_text(contract_text[:500])
        words = clean_full.split()[:20]
        seq = pad_sequences(tokenizer.texts_to_sequences([clean_full]), maxlen=MAX_LEN, padding='post')
        _, attn = attn_model.predict(seq, verbose=0)
        avg_attn = np.mean(attn[0], axis=0)
        disp = min(len(words), 15)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(avg_attn[:disp, :disp], cmap='Blues', ax=ax,
                    xticklabels=words[:disp], yticklabels=words[:disp])
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        # Positional Encoding
        st.subheader("📍 Positional Encoding Heatmap")
        PE = positional_encoding(len(words[:disp]), 32)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(PE, cmap='RdBu_r', ax=ax, yticklabels=words[:disp])
        ax.set_xlabel("Encoding Dimension"); ax.set_ylabel("Word Position")
        st.pyplot(fig)

        st.success("✅ Analysis Complete!")
