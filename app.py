import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import pandas as pd
from groq import Groq

st.set_page_config(
    page_title="HealAI - Mental Health Chatbot",
    page_icon="🧠",
    layout="wide"
)

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_model():
    with open('emotion_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

def predict_emotion(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    emotion = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max() * 100
    return emotion, confidence

emotion_map = {
    'joy':      ('😊', 'Happy'),
    'sadness':  ('😢', 'Sad'),
    'anger':    ('😡', 'Angry'),
    'fear':     ('😰', 'Stressed'),
    'love':     ('🥰', 'Loved'),
    'surprise': ('😮', 'Surprised'),
}

def get_groq_response(user_message, emotion, chat_history):
    system_prompt = """You are PsycheAI, a warm and empathetic mental health support chatbot.
Your personality:
- Talk like a caring, understanding friend — warm and natural, never robotic
- Keep responses to 2-3 sentences maximum
- Always respond to exactly what the user said
- Ask one gentle follow-up question at the end
- Never use bullet points or lists
- Never give medical diagnoses

IMPORTANT: You must accurately analyze the user's emotional state based on their latest message.
Begin your response with exactly one of these tags:
[EMOTION: 😊 Happy]
[EMOTION: 😢 Sad]
[EMOTION: 😡 Angry]
[EMOTION: 😰 Stressed]
[EMOTION: 🥰 Loved]
[EMOTION: 😮 Surprised]
[EMOTION: 😐 Neutral]

Then, on a new line, write your conversational response."""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=200,
        temperature=0.7
    )

    full_text = response.choices[0].message.content
    final_emoji = "😐"
    final_label = "Neutral"
    clean_response = full_text

    if "[EMOTION:" in full_text:
        parts = full_text.split("]", 1)
        if len(parts) > 1:
            tag = parts[0]
            clean_response = parts[1].strip()
            if "Happy"    in tag: final_emoji, final_label = "😊", "Happy"
            elif "Sad"     in tag: final_emoji, final_label = "😢", "Sad"
            elif "Angry"   in tag: final_emoji, final_label = "😡", "Angry"
            elif "Stressed"in tag: final_emoji, final_label = "😰", "Stressed"
            elif "Loved"   in tag: final_emoji, final_label = "🥰", "Loved"
            elif "Surprised"in tag:final_emoji, final_label = "😮", "Surprised"
            else:                  final_emoji, final_label = "😐", "Neutral"

    return clean_response, final_emoji, final_label

# ── Session state ──
import uuid

if 'session_id' not in st.session_state:
    st.session_state.session_id  = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.mood_history = []
    st.session_state.active_tab  = 'chat'
    st.session_state.input_value = ''

if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'mood_history' not in st.session_state: st.session_state.mood_history = []
if 'active_tab'   not in st.session_state: st.session_state.active_tab   = 'chat'
if 'input_value'  not in st.session_state: st.session_state.input_value  = ''

# ── Dark mode & Alignment CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], * { font-family: 'Inter', sans-serif !important; }

.stApp { background-color: #0d0d0d !important; }

section[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #2a2a2a !important;
}
section[data-testid="stSidebar"] * { color: #d1d5db !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

.brand-box {
    background: #111111;
    border-bottom: 1px solid #2a2a2a;
    padding: 22px 18px 20px;
    margin-bottom: 8px;
}

/* Force text left-alignment on sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid #2a2a2a !important;
    color: #888 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    transition: all 0.2s !important;
    justify-content: flex-start !important;
    padding-left: 15px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e1e1e !important;
    color: #a78bfa !important;
    border-color: #7c3aed !important;
}

/* General button helper for content clear chat button */
.clear-btn-container .stButton > button {
    background: transparent !important;
    border: 1px solid #2a2a2a !important;
    color: #888 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    transition: all 0.2s !important;
}
.clear-btn-container .stButton > button:hover {
    background: #1e1e1e !important;
    color: #ef4444 !important;
    border-color: #ef4444 !important;
}

.chat-header {
    background: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 18px;
}

.user-msg {
    background: #7c3aed;
    color: white !important;
    padding: 11px 16px;
    border-radius: 14px 4px 14px 14px;
    display: inline-block;
    max-width: 75%;
    font-size: 0.88rem;
    line-height: 1.6;
}
.bot-msg {
    background: #1a1a1a;
    color: #d1d5db !important;
    padding: 13px 16px;
    border-radius: 4px 14px 14px 14px;
    display: inline-block;
    max-width: 75%;
    font-size: 0.88rem;
    line-height: 1.6;
    border: 1px solid #2a2a2a;
}
.emotion-pill {
    background: #2a1a4a;
    color: #a78bfa !important;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 7px;
}
.msg-time { font-size: 0.68rem; color: #3a3a3a; margin-top: 4px; }

.crisis-box {
    background: #1a0a0a;
    border: 1px solid #3a1a1a;
    border-radius: 10px;
    padding: 12px;
    margin: 6px 10px 14px;
}

.tip-card {
    background: #111111;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 14px;
}

div[data-testid="stVerticalBlock"] { gap: 0rem; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:36px;height:36px;background:#7c3aed;border-radius:10px;
                        display:flex;align-items:center;justify-content:center;font-size:1.1rem">🧠</div>
            <div>
                <div style="color:#f1f1f1;font-weight:700;font-size:1rem">HealAI</div>
                <div style="color:#555;font-size:0.7rem">Mental Health Chatbot</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("💬 Chat",           use_container_width=True, key="btn_chat"):
        st.session_state.active_tab = 'chat';  st.rerun()
    if st.button("📊 Mood Tracker",   use_container_width=True, key="btn_mood"):
        st.session_state.active_tab = 'mood';  st.rerun()
    if st.button("🌿 Self Care Tips", use_container_width=True, key="btn_self"):
        st.session_state.active_tab = 'selfcare'; st.rerun()

    st.markdown("""
    <div class="crisis-box">
        <div style="color:#ef4444;font-weight:600;font-size:0.82rem">📞 Crisis Help</div>
        <div style="color:#7a3a3a;font-size:0.72rem;margin-top:4px;line-height:1.6">
            Pakistan: 0311-7786264<br>Umang helpline: 0317-4288665
        </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════
# CHAT TAB
# ════════════════════════════════
if st.session_state.active_tab == 'chat':

    st.markdown("""
    <div class="chat-header">
        <div style="color:#f1f1f1;font-weight:600;font-size:1.1rem">
            Hello, I'm here for you 
        </div>
        <div style="color:#555;font-size:0.82rem;margin-top:3px">
            How are you feeling today?
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div style="display:flex;align-items:flex-start;justify-content:flex-start;gap:12px;margin-bottom:18px;width:100%;">
            <div style="width:34px;height:34px;background:#7c3aed;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.95rem;flex-shrink:0">🧠</div>
            <div style="flex-grow: 1; max-width: 100%;">
                <div class="bot-msg" style="display:block; max-width:75%;">
                    Welcome to HealAI.
                    I am your AI assistant, dedicated to providing a safe, non-judgmental space for your thoughts.
                    How are you feeling today?
                </div>
                <div class="msg-time">Just now</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:14px">
                <div>
                    <div class="user-msg">{msg['content']}</div>
                    <div class="msg-time" style="text-align:right">{msg['time']} ✓✓</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;justify-content:flex-start;gap:12px;margin-bottom:14px;width:100%;">
                <div style="width:34px;height:34px;background:#7c3aed;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;
                            font-size:0.95rem;flex-shrink:0">🧠</div>
                <div style="flex-grow: 1; max-width: 100%;">
                    <div class="bot-msg" style="display:block; max-width:75%;">
                        <span class="emotion-pill">{msg['emotion_emoji']} {msg['emotion_label']}</span><br>
                        {msg['content']}
                    </div>
                    <div class="msg-time">{msg['time']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Clear Chat placed ABOVE the Text Input field
    st.markdown('<div class="clear-btn-container">', unsafe_allow_html=True)
    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("🗑️ Clear chat", key="clear_btn"):
            st.session_state.chat_history = []
            st.session_state.mood_history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Native merged/unified chat input block
    final_input = st.chat_input("Type your message here...")

    if final_input:
        user_text = final_input.strip()
        now = datetime.now().strftime("%I:%M %p")

        st.session_state.chat_history.append({
            'role': 'user', 'content': user_text, 'time': now
        })

        with st.spinner("HealAI is typing..."):
            try:
                emotion, confidence = predict_emotion(user_text)
                groq_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]
                ]
                reply, emoji, label = get_groq_response(
                    user_text, emotion, groq_history
                )
            except Exception as e:
                reply = f"I'm sorry, I had trouble responding. Error: {str(e)}"
                emoji, label = "😐", "Neutral"

        st.session_state.chat_history.append({
            'role': 'assistant', 'content': reply,
            'emotion_emoji': emoji, 'emotion_label': label,
            'time': datetime.now().strftime("%I:%M %p")
        })
        st.session_state.mood_history.append({
            'emoji': emoji, 'label': label, 'time': now,
            'message': user_text[:40] + '...' if len(user_text) > 40 else user_text
        })
        st.rerun()

# ── MOOD TRACKER TAB ──
elif st.session_state.active_tab == 'mood':
    st.markdown("""
    <div class="chat-header">
        <div style="color:#f1f1f1;font-weight:600;font-size:1.1rem">📊 Mood Tracker</div>
        <div style="color:#555;font-size:0.82rem;margin-top:3px">Your emotional journey</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.mood_history:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px">
            <div style="font-size:3rem;margin-bottom:16px">📊</div>
            <div style="color:#555;font-size:0.95rem">
                Start chatting and your mood history will appear here!
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = pd.DataFrame(st.session_state.mood_history)
        st.markdown("##### Mood Distribution")
        st.bar_chart(df['label'].value_counts())
        st.markdown("##### Recent Entries")
        for entry in reversed(st.session_state.mood_history[-10:]):
            a, b, c = st.columns([1, 6, 2])
            with a:
                st.markdown(f"<div style='font-size:1.4rem'>{entry['emoji']}</div>",
                           unsafe_allow_html=True)
            with b:
                st.markdown(f"<span style='color:#d1d5db;font-weight:500'>{entry['label']}</span><br>"
                           f"<span style='color:#555;font-size:0.78rem'>{entry['message']}</span>",
                           unsafe_allow_html=True)
            with c:
                st.markdown(f"<span style='color:#444;font-size:0.78rem'>{entry['time']}</span>",
                           unsafe_allow_html=True)
            st.divider()

# ── SELF CARE TAB ──
elif st.session_state.active_tab == 'selfcare':
    st.markdown("""
    <div class="chat-header">
        <div style="color:#f1f1f1;font-weight:600;font-size:1.1rem">🌿 Self Care Tips</div>
        <div style="color:#555;font-size:0.82rem;margin-top:3px">
            Small steps for a healthier mind
        </div>
    </div>
    """, unsafe_allow_html=True)

    tips = [
        ("🧘", "Mindfulness",
         "Take 5 minutes to focus only on your breathing. Inhale for 4 counts, hold for 4, exhale for 6. This immediately calms your nervous system."),
        ("🚶", "Movement",
         "A 10-minute walk outside can significantly improve your mood and reduce stress. You don't need a gym — just fresh air and movement."),
        ("📔", "Journaling",
         "Write down 3 things you are grateful for every morning. This simple habit shifts your mindset from scarcity to abundance over time."),
        ("💤", "Sleep",
         "Maintain a consistent sleep schedule. Going to bed and waking at the same time every day dramatically improves mental health."),
        ("📵", "Digital Detox",
         "Take 1 hour away from your phone every day. Constant notifications keep your brain in a stressed state without you realizing it."),
        ("🤝", "Connection",
         "Reach out to one person you trust today. Even a short message can strengthen bonds and remind you that you are not alone."),
    ]

    c1, c2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(tips):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f"""
            <div class="tip-card">
                <div style="font-size:1.8rem;margin-bottom:10px">{icon}</div>
                <div style="color:#f1f1f1;font-weight:600;font-size:0.95rem;margin-bottom:6px">{title}</div>
                <div style="color:#666;font-size:0.83rem;line-height:1.7">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
