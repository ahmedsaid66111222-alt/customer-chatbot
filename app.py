import streamlit as st
import openai
from pathlib import Path
import PyPDF2
import docx
import io
import pickle
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ====== إعدادات الصفحة ======
st.set_page_config(
    page_title="مساعد العملاء الذكي",
    page_icon="🤖",
    layout="wide"
)

# ====== تهيئة Session State ======
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = []

# ====== وظائف مساعدة ======

def extract_text_from_pdf(file):
    """استخراج النص من PDF"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    """استخراج النص من Word"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file):
    """استخراج النص من TXT"""
    return file.read().decode('utf-8')

def split_text_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """تقسيم النص إلى أجزاء صغيرة"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_embedding(text: str, api_key: str) -> List[float]:
    """الحصول على embedding من OpenAI"""
    openai.api_key = api_key
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def find_relevant_chunks(query: str, api_key: str, top_k: int = 3) -> List[str]:
    """البحث عن أكثر الأجزاء صلة بالسؤال"""
    if not st.session_state.knowledge_base or not st.session_state.embeddings:
        return []
    
    # الحصول على embedding للسؤال
    query_embedding = get_embedding(query, api_key)
    
    # حساب التشابه
    similarities = cosine_similarity(
        [query_embedding],
        st.session_state.embeddings
    )[0]
    
    # ترتيب النتائج
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [st.session_state.knowledge_base[i] for i in top_indices]

def chat_with_bot(user_message: str, api_key: str) -> str:
    """التحدث مع البوت"""
    openai.api_key = api_key
    
    # البحث عن معلومات ذات صلة
    relevant_info = find_relevant_chunks(user_message, api_key)
    
    # بناء السياق
    context = "\n\n".join(relevant_info) if relevant_info else "لا توجد معلومات متاحة."
    
    # بناء الرسائل
    system_message = f"""أنت مساعد ذكي لخدمة العملاء. استخدم المعلومات التالية للإجابة على أسئلة العملاء:

{context}

قواعد مهمة:
- أجب بناءً على المعلومات المتوفرة فقط
- إذا لم تجد الإجابة في المعلومات، قل "عذراً، لا أملك معلومات كافية للإجابة على هذا السؤال"
- كن مهذباً ومحترفاً
- أجب باللغة العربية"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    # إرسال الطلب لـ OpenAI
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# ====== الواجهة الرئيسية ======

st.title("🤖 مساعد العملاء الذكي")
st.markdown("---")

# Sidebar للإعدادات
with st.sidebar:
    st.header("⚙️ الإعدادات")
    
    # إدخال API Key
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="أدخل مفتاح API الخاص بك من OpenAI"
    )
    
    st.markdown("---")
    st.header("📁 رفع البيانات")
    
    uploaded_files = st.file_uploader(
        "ارفع ملفات شركتك (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    if st.button("🔄 معالجة الملفات", type="primary"):
        if not api_key:
            st.error("⚠️ من فضلك أدخل OpenAI API Key أولاً!")
        elif uploaded_files:
            with st.spinner("جاري معالجة الملفات..."):
                all_text = ""
                
                for file in uploaded_files:
                    if file.name.endswith('.pdf'):
                        text = extract_text_from_pdf(file)
                    elif file.name.endswith('.docx'):
                        text = extract_text_from_docx(file)
                    elif file.name.endswith('.txt'):
                        text = extract_text_from_txt(file)
                    
                    all_text += text + "\n\n"
                
                # تقسيم النص
                chunks = split_text_into_chunks(all_text)
                
                # الحصول على embeddings
                embeddings = []
                progress_bar = st.progress(0)
                for i, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, api_key)
                    embeddings.append(embedding)
                    progress_bar.progress((i + 1) / len(chunks))
                
                # حفظ في Session State
                st.session_state.knowledge_base = chunks
                st.session_state.embeddings = embeddings
                
                st.success(f"✅ تم معالجة {len(chunks)} قطعة من المعلومات!")
        else:
            st.warning("⚠️ من فضلك ارفع ملف واحد على الأقل")
    
    # إحصائيات
    st.markdown("---")
    st.metric("📊 عدد المعلومات المحفوظة", len(st.session_state.knowledge_base))
    
    # زر مسح المحادثة
    if st.button("🗑️ مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()

# المحادثة الرئيسية
st.header("💬 المحادثة")

# عرض الرسائل
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# إدخال الرسالة
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    if not api_key:
        st.error("⚠️ من فضلك أدخل OpenAI API Key من القائمة الجانبية!")
    else:
        # إضافة رسالة المستخدم
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # الحصول على الرد
        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                try:
                    response = chat_with_bot(prompt, api_key)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"❌ حدث خطأ: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>مساعد العملاء الذكي | مدعوم بـ OpenAI & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
