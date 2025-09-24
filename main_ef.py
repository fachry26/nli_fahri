import streamlit as st
import pandas as pd
import torch
import re
import string
from transformers import pipeline
from io import BytesIO
import math
import warnings

# Suppress benign UserWarnings from PyTorch
warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="Contextual & Rule-Based Classifier", layout="wide")
st.title("🎯 Indonesian Contextual & Rule-Based Classifier")
st.markdown("Combines custom rules, NLI classification, and generative label discovery.")

# =========================
# Model Loading (Cached)
# =========================
@st.cache_resource
def load_models():
    """Loads both the NLI classifier and the summarization pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"🚀 Models running on: **{device.upper()}**")
    
    # --- OPTIMIZATION: FP16 for GPUs ---
    model_kwargs = {}
    if device == "cuda":
        st.sidebar.success("✅ GPU detected! Enabling FP16 for faster inference.")
        model_kwargs["torch_dtype"] = torch.float16
    else:
        st.sidebar.warning("CUDA not available. Running on CPU will be slower.")

    # Model 1: NLI Classifier (Using the original, more powerful model)
    nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    classifier = pipeline(
        "zero-shot-classification",
        model=nli_model_name,
        device=device,
        multi_label=True,
        model_kwargs=model_kwargs # Pass the GPU-specific arguments
    )

    # Model 2: Summarizer for Label Generation
    summarizer_model_name = "panggi/t5-base-indonesian-summarization-cased"
    summarizer = pipeline(
        "summarization",
        model=summarizer_model_name,
        device=device,
        model_kwargs=model_kwargs # Also apply to summarizer
    )
    
    return classifier, summarizer

classifier, summarizer = load_models()

# =========================
# Preprocessing & Rule Functions
# =========================
def nuanced_clean_text(text):
    """
    A more careful cleaning function that preserves case and some punctuation
    for better sentiment and sarcasm detection. Returns a tuple:
    (text_for_rules, text_for_nli)
    """
    text = str(text)
    
    # Minimal cleaning for the powerful NLI model. Remove only URLs and emojis.
    original_for_nli = re.sub(r"http\S+", "", text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
        "]+",
        flags=re.UNICODE)
    original_for_nli = emoji_pattern.sub(r'', original_for_nli).strip()

    # More aggressive cleaning for rule-based matching.
    text_for_rules = text.lower()
    text_for_rules = re.sub(r"http\S+", "", text_for_rules)
    text_for_rules = emoji_pattern.sub(r'', text_for_rules)
    
    punctuation_to_replace = string.punctuation.replace('@', '').replace('#', '')
    translator = str.maketrans(punctuation_to_replace, ' ' * len(punctuation_to_replace))
    text_for_rules = text_for_rules.translate(translator)
    
    text_for_rules = re.sub(r"\d+", "", text_for_rules)
    
    slang_dict = {'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak','tdk': 'tidak', 'bgt': 'banget', 'jg': 'juga', 'aja': 'saja','utk': 'untuk', 'dgn': 'dengan', 'dr': 'dari', 'sdh': 'sudah','blm': 'belum', 'gimana': 'bagaimana', 'kalo': 'kalau', 'udh': 'sudah'}
    words = text_for_rules.split()
    normalized_words = [slang_dict.get(w, w) for w in words]
    cleaned_for_rules = " ".join(normalized_words)
    
    return cleaned_for_rules, original_for_nli

def check_hate_speech(text, hate_speech_phrases):
    """Checks for hate speech, case-insensitive."""
    return any(phrase in text.lower() for phrase in hate_speech_phrases)

def match_rule(rule, text):
    """Matches a rule pattern against text, case-insensitive."""
    rule = str(rule).strip().lower()
    text_lower = text.lower()
    if not rule: return False
    if "+" in rule: return all(p.strip() in text_lower for p in rule.split("+"))
    elif "-" in rule:
        must, must_not = rule.split("-", 1)
        return must.strip() in text_lower and must_not.strip() not in text_lower
    elif "%" in rule: return any(p.strip() in text_lower for p in rule.split("%") if p.strip())
    else: return rule in text_lower

# =========================
# Main Processing Logic
# =========================
def generate_label(text):
    """Uses the summarization model to generate a potential label."""
    if len(text.split()) < 5: 
        return text 
    summary = summarizer(text, max_length=20, min_length=3, do_sample=False)
    return summary[0]['summary_text']

def classify_single_text_combined(text, candidate_labels, hate_speech_phrases, rules_df, discovery_mode=False):
    """Classifies a single text using the full pipeline."""
    rule_text, nli_text = nuanced_clean_text(text)
    
    if not rule_text and not nli_text:
        return {"labels": ["No text provided"], "scores": [1.0]}, "", text

    if check_hate_speech(rule_text, hate_speech_phrases):
        return {"labels": ["⚠️ Hate Speech"], "scores": [1.0]}, rule_text, text

    if rules_df is not None and not rules_df.empty:
        for _, row in rules_df.iterrows():
            if match_rule(row["words"], rule_text):
                return {"labels": [f"Rule: {row['sentiment']}"], "scores": [1.0]}, rule_text, text

    if discovery_mode:
        generated_topic = generate_label(nli_text)
        validation_result = classifier(nli_text, [generated_topic])
        return {
            "labels": [f"Discovered: {validation_result['labels'][0]}"],
            "scores": validation_result['scores']
        }, rule_text, text

    if not candidate_labels:
        return {"labels": ["No NLI labels provided"], "scores": [1.0]}, rule_text, text
    
    result = classifier(nli_text, candidate_labels)
    return result, rule_text, text

def get_parent_label(top_label, label_map):
    """Determines the parent category for a given label."""
    if not isinstance(top_label, str): return "Other"
    if top_label.startswith("Rule:"): return "Rule"
    elif top_label.startswith("⚠️"): return "Hate Speech"
    elif top_label.startswith("Discovered:"): return "Discovered"
    elif top_label in ["Empty Text", "No text provided", "No NLI labels provided"]: return "System"
    else: return label_map.get(top_label, "Uncategorized")

def process_bulk_data_combined(df, text_col, candidate_labels, hate_speech_phrases, rules_df, label_to_parent_map, discovery_mode=False):
    """Processes a DataFrame using the full pipeline with chunked progress updates."""
    if 'original_text' not in df.columns:
        df['original_text'] = df[text_col]
        
    texts = df['original_text'].astype(str).tolist()
    total_rows = len(texts)
    
    progress_bar = st.progress(0, text="Step 1/3: Cleaning text...")
    
    processed_texts = [nuanced_clean_text(t) for t in texts]
    df['cleaned_text_for_rules'] = [t[0] for t in processed_texts]
    df['text_for_nli'] = [t[1] for t in processed_texts]

    texts_to_process, indices_to_process = [], []
    top_labels, top_scores = [None] * len(df), [None] * len(df)
    
    rule_list = [(row["words"], row["sentiment"]) for _, row in rules_df.iterrows()] if rules_df is not None and not rules_df.empty else []

    # --- Step 2: Apply Rules and Hate Speech Checks ---
    for i, row in df.iterrows():
        current_progress = 0.2 + (0.3 * (i + 1) / total_rows)
        progress_bar.progress(current_progress, text=f"Step 2/3: Applying rules... (Row {i+1}/{total_rows})")
        rule_text = row['cleaned_text_for_rules']
        
        if not rule_text: 
            top_labels[i] = "Empty Text"; top_scores[i] = 0.0
        elif check_hate_speech(rule_text, hate_speech_phrases): 
            top_labels[i] = "⚠️ Hate Speech"; top_scores[i] = 1.0
        else:
            rule_applied = False
            if rule_list:
                for rule, sentiment in rule_list:
                    if match_rule(rule, rule_text):
                        top_labels[i] = f"Rule: {sentiment}"; top_scores[i] = 1.0
                        rule_applied = True; break
            if not rule_applied:
                texts_to_process.append(row['text_for_nli'])
                indices_to_process.append(i)
    
    # --- Step 3: Process remaining texts with AI model in chunks ---
    if texts_to_process:
        chunk_size = 32 # Process 32 texts at a time
        num_chunks = math.ceil(len(texts_to_process) / chunk_size)

        for i in range(0, len(texts_to_process), chunk_size):
            chunk_num = (i // chunk_size) + 1
            progress_val = 0.5 + (0.5 * chunk_num / num_chunks)
            progress_bar.progress(progress_val, text=f"Step 3/3: Processing AI batch {chunk_num}/{num_chunks}...")

            text_chunk = texts_to_process[i:i + chunk_size]
            index_chunk = indices_to_process[i:i + chunk_size]

            if discovery_mode:
                # Add discovery mode chunk logic if needed in the future
                pass 
            else: # Classification Mode
                batch_results = classifier(text_chunk, candidate_labels)
                for j, result_dict in enumerate(batch_results):
                    original_index = index_chunk[j]
                    top_labels[original_index] = result_dict['labels'][0]
                    top_scores[original_index] = result_dict['scores'][0]

    progress_bar.progress(1.0, text="✅ Analysis Complete!")
    df['top_label'] = top_labels
    df['top_score'] = top_scores
    df['parent_label'] = df['top_label'].apply(lambda x: get_parent_label(x, label_to_parent_map))
    
    # Reorder columns for clean output
    final_cols = ['original_text', 'top_label', 'parent_label', 'top_score', 'cleaned_text_for_rules', 'text_for_nli']
    existing_cols = [c for c in df.columns if c not in final_cols]
    df = df[final_cols + existing_cols]
        
    return df

def to_excel(df):
    """Converts a DataFrame to an Excel file in memory."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return output.getvalue()

# =========================
# Streamlit UI
# =========================
def main():
    st.sidebar.header("⚙️ Analysis Configuration")
    
    analysis_mode = st.sidebar.radio(
        "Select Analysis Mode",
        ('Classification (Predefined Labels)', 'Discovery (Generate Labels)'),
        help="**Classification:** Match text against a list of labels you provide. **Discovery:** The model tries to generate a relevant label for each text automatically."
    )
    discovery_mode = (analysis_mode == 'Discovery (Generate Labels)')

    st.sidebar.subheader("🎯 NLI Model Labels")
    st.sidebar.info("Define your labels below, separated by new lines.")

    positive_labels_input = st.sidebar.text_area("✅ Positive Labels", height=120, 
        value="Memberikan pujian atau apresiasi tulus secara langsung kepada BPS atas kinerja atau data yang akurat dan bermanfaat.\nMenyajikan laporan faktual atau rilis pers resmi dari BPS mengenai penyelenggaraan kegiatan seperti survei dan pelatihan.")
    negative_labels_input = st.sidebar.text_area("❌ Negative Labels", height=180,
        value='Menuduh secara langsung bahwa data BPS adalah hoax, berita bohong, atau tidak benar.\nMenyampaikan kritik atau keraguan pada metode dan keakuratan data yang dikeluarkan BPS.\nMenunjukkan sentimen negatif, cemoohan, penghinaan, atau ketidakpercayaan yang kuat pada BPS, seringkali dengan nada merendahkan.\nMenggunakan sindiran halus, sarkasme, atau bahasa berkonotasi negatif untuk mengkritik BPS secara tidak langsung.\nMengajukan pertanyaan retoris atau membeberkan kontradiksi untuk menyanggah atau meragukan klaim BPS.\nMenuduh BPS bekerja secara tidak profesional, tidak independen, atau memanipulasi data untuk tujuan tertentu.\nMempertanyakan atau mengkritik standar dan metodologi BPS, seringkali dengan membandingkannya dengan sumber data lain atau kondisi riil di masyarakat.\nMenyatakan ulang data atau klaim positif BPS dengan nada sarkastis atau skeptis untuk menunjukkan ketidakpercayaan.')
    neutral_labels_input = st.sidebar.text_area("➖ Neutral Labels", height=120,
        value='Menyampaikan informasi faktual, seperti judul berita atau pengumuman, mengenai rencana atau kegiatan yang akan dilakukan oleh BPS.\nSecara eksplisit membahas, menyebutkan, atau menyajikan angka data terkait Produk Domestik Bruto (PDB).\nMenyebutkan atau membagikan informasi statistik umum (contoh: inflasi, kemiskinan).\nTeks ini tidak relevan, menggunakan singkatan BPS untuk konteks lain (luar negeri, game, produk), atau menyebut BPS Indonesia tanpa opini yang jelas.')

    positive_labels = [label.strip() for label in positive_labels_input.split('\n') if label.strip()]
    negative_labels = [label.strip() for label in negative_labels_input.split('\n') if label.strip()]
    neutral_labels = [label.strip() for label in neutral_labels_input.split('\n') if label.strip()]
    
    candidate_labels = sorted(list(set(positive_labels + negative_labels + neutral_labels)))
    
    label_to_parent_map = {label: "Positive" for label in positive_labels}
    label_to_parent_map.update({label: "Negative" for label in negative_labels})
    label_to_parent_map.update({label: "Neutral" for label in neutral_labels})

    if candidate_labels and not discovery_mode: 
        st.sidebar.success(f"✅ {len(candidate_labels)} total NLI labels ready.")
    
    st.sidebar.divider()
    
    st.sidebar.header("📖 Hate Speech Dictionary")
    hate_input = st.sidebar.text_area("Add hate speech phrases (comma-separated):", "")
    hate_speech_phrases = [h.strip().lower() for h in hate_input.split(",") if h.strip()]
    if hate_speech_phrases: st.sidebar.success(f"Loaded {len(hate_speech_phrases)} phrases.")

    if "rules_df" not in st.session_state:
        st.session_state.rules_df = pd.DataFrame({"words": pd.Series(dtype='str'), "sentiment": pd.Series(dtype='str')})

    # --- MAIN PAGE TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["✍️ Manual Analysis", "🧪 Quick Bulk Test", "📂 Bulk Analysis (Excel)", "⚡ Custom Rule Config"])

    with tab1:
        st.subheader("Analyze a Single Piece of Text")
        text_input = st.text_area("Enter text:", height=100, key="manual_text")
        
        if st.button("🔍 Analyze Text", key="manual_button"):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    result, cleaned_text, original_text = classify_single_text_combined(
                        text_input, candidate_labels, hate_speech_phrases, st.session_state.rules_df, discovery_mode=discovery_mode
                    )
                
                top_label = result['labels'][0]
                parent_label = get_parent_label(top_label, label_to_parent_map)
                
                st.subheader("📌 Analysis Result")
                st.markdown(f"> **Original Text:** {original_text}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Top Label", top_label)
                col2.metric("Parent Label", parent_label)
                col3.metric("Confidence", f"{result['scores'][0]:.2%}")
                
                if not discovery_mode and not top_label.startswith(("Rule:", "⚠️", "No")):
                    st.subheader("Full NLI Score Breakdown")
                    result_df = pd.DataFrame({"Label": result['labels'], "Score": result['scores']})
                    st.dataframe(result_df, use_container_width=True)
            else:
                st.warning("Please enter some text.")

    with tab2:
        st.subheader("Quick Test on Multiple Texts")
        st.info("Paste your texts below, one per line. The original text will be preserved in the output.")

        texts_input = st.text_area("Enter texts (one per line):", height=250, key="quick_bulk_text")

        if st.button("🚀 Analyze All Texts", key="quick_bulk_button"):
            lines = [line.strip() for line in texts_input.split('\n') if line.strip()]
            if lines:
                quick_test_df = pd.DataFrame(lines, columns=["text_to_analyze"])

                result_df = process_bulk_data_combined(
                    quick_test_df, 
                    "text_to_analyze", 
                    candidate_labels, 
                    hate_speech_phrases, 
                    st.session_state.rules_df,
                    label_to_parent_map,
                    discovery_mode=discovery_mode
                )

                st.subheader("📊 Quick Test Results")
                st.dataframe(result_df)
            else:
                st.warning("Please enter at least one line of text to analyze.")

    with tab3:
        st.subheader("Analyze a Full Excel File")
        st.info("The original text column will be preserved in the output file.")
        if discovery_mode:
            st.warning("⚠️ **Discovery Mode is active.**")

        uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", key="bulk_upload_excel")
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.write("📄 Data Preview:")
            st.dataframe(df.head())
            text_col = st.selectbox("Select the text column to analyze:", df.columns, key="excel_col_select")
            
            if st.button("🚀 Run Bulk Analysis", key="bulk_button_excel"):
                if not discovery_mode and not candidate_labels and st.session_state.rules_df.empty:
                    st.warning("Please configure Custom Rules or provide NLI Labels for Classification Mode.")
                else:
                    with st.spinner("Processing file... This may take a while."):
                        result_df = process_bulk_data_combined(
                            df.copy(), text_col, candidate_labels, hate_speech_phrases, st.session_state.rules_df, label_to_parent_map, discovery_mode=discovery_mode
                        )
                    st.subheader("📊 Analysis Results")
                    st.dataframe(result_df)
                    st.download_button(
                        label="📥 Download Results as Excel", data=to_excel(result_df),
                        file_name="combined_analysis_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet"
                    )

    with tab4:
        st.subheader("⚡ Custom Rule Configuration")
        st.info("Rules are applied first, before the NLI model or Label Discovery.")
        
        uploaded_rules = st.file_uploader("Upload rules file (.xlsx)", type=["xlsx"], key="file_rules_upload")
        if uploaded_rules:
            try:
                rules_from_excel = pd.read_excel(uploaded_rules, dtype=str)
                if all(col in rules_from_excel.columns for col in ["words", "sentiment"]):
                    st.session_state.rules_df = pd.concat(
                        [st.session_state.rules_df, rules_from_excel], ignore_index=True
                    ).drop_duplicates(subset=["words"], keep="last")
                    st.success("✅ Merged rules from Excel!")
                else: st.error("⚠️ Rules file must have 'words' and 'sentiment' columns.")
            except Exception as e: st.error(f"Error reading rules file: {e}")

        st.markdown("#### Edit Rules Manually")
        st.caption("Operators: `+` (AND), `-` (MUST+NOT), `%` (OR)")
        
        edited_df = st.data_editor(
            st.session_state.rules_df, num_rows="dynamic", use_container_width=True, hide_index=True,
            column_config={
                "words": st.column_config.TextColumn("Rule Pattern", required=True),
                "sentiment": st.column_config.TextColumn("Assigned Label", required=True)
            }
        )
        st.session_state.rules_df = edited_df

if __name__ == '__main__':
    main()