import streamlit as st
from transformers import pipeline
import time
import torch
import PyPDF2
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Article Summarizer Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le chat
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .bot-message strong {
        color: #2e7d32;
    }
    .input-area {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: white;
        padding: 15px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    .stFileUploader {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Titre
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title("🤖 Article Summarizer Bot")
with col2:
    st.write("")
    if st.button("🔄 Nouveau Chat", key="reset_btn"):
        st.session_state.messages = []
        st.rerun()

st.markdown("Envoyez-moi vos articles ou importez un PDF pour les résumer automatiquement!")
st.markdown("---")

# Sidebar - Paramètres
st.sidebar.header("⚙️ Paramètres")
model_choice = st.sidebar.selectbox(
    "Modèle:",
    [
        "Votre modèle personnalisé",
        "distilbart-cnn-6-6",
        "facebook/bart-large-cnn"
    ],
    index=0
)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"📱 Appareil: {device.upper()}")
st.sidebar.info("ℹ️ Utilisation des paramètres par défaut du modèle")

# Initialiser l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# ⚙️ CONFIGURATION - MODIFIEZ LE CHEMIN ICI ⚙️
CUSTOM_MODEL_PATH = "C:/Users/abdel/Downloads/my_model/content/my_model"  # 👈 REMPLACEZ PAR LE CHEMIN DE VOTRE MODÈLE

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip(), None
    except Exception as e:
        return None, f"Erreur lors de la lecture du PDF: {str(e)}"

# Charger le modèle
@st.cache_resource
def load_summarizer(model_name):
    try:
        if model_name == "Votre modèle personnalisé":
            model_path = CUSTOM_MODEL_PATH
        else:
            model_path = model_name
        
        summarizer = pipeline(
            "summarization",
            model=model_path,
            device=0 if device == "cuda" else -1
        )
        return summarizer, None
    except Exception as e:
        return None, f"Erreur: {str(e)}"

summarizer, error = load_summarizer(model_choice)

if error:
    st.sidebar.error(f"❌ {error}")
else:
    st.sidebar.success(f"✅ {model_choice} chargé!")

# Zone d'upload de fichier PDF
st.markdown("### 📄 Importer un PDF")
uploaded_file = st.file_uploader(
    "Choisissez un fichier PDF",
    type=['pdf'],
    help="Téléchargez un PDF pour en extraire et résumer le contenu",
    key="pdf_uploader"
)

# Traiter le PDF uploadé
if uploaded_file is not None:
    with st.spinner("📖 Lecture du PDF..."):
        extracted_text, pdf_error = extract_text_from_pdf(uploaded_file)
        
        if pdf_error:
            st.error(pdf_error)
        elif extracted_text:
            # Ajouter le message utilisateur (extrait du PDF)
            preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
            
            st.session_state.messages.append({
                "role": "user",
                "content": f"📄 PDF: {uploaded_file.name}\n\n{extracted_text}"
            })
            
            # Afficher un aperçu
            with st.chat_message("user", avatar="👤"):
                st.write(f"**📄 PDF importé:** {uploaded_file.name}")
                with st.expander("📖 Aperçu du contenu"):
                    st.write(preview_text)
            
            # Générer le résumé
            with st.chat_message("assistant", avatar="🤖"):
                word_count = len(extracted_text.split())
                
                if word_count < 50:
                    bot_response = f"⚠️ Le contenu du PDF est trop court ({word_count} mots). Veuillez envoyer un document avec au moins 50 mots."
                elif summarizer is None:
                    bot_response = "❌ Le modèle n'a pas pu être chargé."
                else:
                    try:
                        with st.spinner("⏳ Génération du résumé..."):
                            start = time.time()
                            
                            # Limiter la taille de l'input
                            text_to_summarize = extracted_text[:1024]
                            
                            # Générer le résumé
                            summary_result = summarizer(
                                text_to_summarize,
                                do_sample=False
                            )
                            
                            elapsed = time.time() - start
                            summary_text = summary_result[0]['summary_text']
                            summary_words = len(summary_text.split())
                            compression = (1 - summary_words/word_count) * 100 if word_count > 0 else 0
                            
                            bot_response = f"""
**📄 RÉSUMÉ GÉNÉRÉ:**

{summary_text}

---

**📊 STATISTIQUES:**
- 📁 Fichier: {uploaded_file.name}
- 📝 Document: {word_count} mots
- ✂️ Résumé: {summary_words} mots  
- 📉 Compression: {compression:.1f}%
- ⏱️ Temps: {elapsed:.2f}s
- 🧠 Modèle: {model_choice}
"""
                            
                    except Exception as e:
                        bot_response = f"❌ Erreur: {str(e)}\n\nEssayez avec un document plus court ou changez de modèle."
                
                st.markdown(bot_response)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_response
                })
        else:
            st.warning("⚠️ Aucun texte n'a pu être extrait du PDF.")

st.markdown("---")

# Afficher les messages du chat
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                content = message["content"]
                if content.startswith("📄 PDF:"):
                    # Afficher uniquement le nom du fichier pour les PDF
                    file_name = content.split("\n")[0]
                    st.write(file_name)
                else:
                    st.write(content[:200] + "..." if len(content) > 200 else content)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])

# Zone de saisie
st.markdown("<br>" * 2, unsafe_allow_html=True)
st.markdown("### ✍️ Ou saisissez votre texte directement")

# Input utilisateur
user_input = st.chat_input(
    "📝 Collez votre article ici...",
    key="user_input"
)

# Traiter l'entrée utilisateur
if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user", avatar="👤"):
        st.write(user_input[:200] + "..." if len(user_input) > 200 else user_input)
    
    with st.chat_message("assistant", avatar="🤖"):
        word_count = len(user_input.split())
        
        if word_count < 50:
            bot_response = f"⚠️ L'article est trop court ({word_count} mots). Veuillez envoyer un article avec au moins 50 mots."
        elif summarizer is None:
            bot_response = "❌ Le modèle n'a pas pu être chargé."
        else:
            try:
                with st.spinner("⏳ Génération du résumé..."):
                    start = time.time()
                    text_to_summarize = user_input[:1024]
                    
                    summary_result = summarizer(
                        text_to_summarize,
                        do_sample=False
                    )
                    
                    elapsed = time.time() - start
                    summary_text = summary_result[0]['summary_text']
                    summary_words = len(summary_text.split())
                    compression = (1 - summary_words/word_count) * 100 if word_count > 0 else 0
                    
                    bot_response = f"""
**📄 RÉSUMÉ GÉNÉRÉ:**

{summary_text}

---

**📊 STATISTIQUES:**
- 📝 Article: {word_count} mots
- ✂️ Résumé: {summary_words} mots  
- 📉 Compression: {compression:.1f}%
- ⏱️ Temps: {elapsed:.2f}s
- 🧠 Modèle: {model_choice}
"""
                    
            except Exception as e:
                bot_response = f"❌ Erreur: {str(e)}\n\nEssayez avec un texte plus court ou changez de modèle."
        
        st.markdown(bot_response)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_response
        })

# Pied de page
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 11px;'>
🤖 Article Summarizer Bot | Powered by DistilBart & Streamlit | Supporte PDF
</div>
""", unsafe_allow_html=True)