import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# ===============================
# アプリケーションの設定
# ===============================
st.set_page_config(
    page_title="StreamlitとLangChainによるチャットボット",
    page_icon="🤖",
    layout="wide"
)
st.title("StreamlitとLangChainによるチャットボット")

# ===============================
# モデルおよびAPIの設定
# ===============================
MODEL = "cyberagent/open-calm-small"
HUGGINGFACEHUB_API_TOKEN = st.secrets['HF_API_TOKEN']
TEMPERATURE = 0.7

SYSTEM_PROMPT = "あなたは役に立つアシスタントです。すべて日本語で応答してください。"

# ===============================
# Chatクライアントの初期化
# ===============================
chat = HuggingFaceHub(
    repo_id=MODEL,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    model_kwargs={"temperature": TEMPERATURE, "max_length": 150}
)

# ===============================
# セッション状態の初期化
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# ===============================
# メッセージ変換関数
# ===============================
def convert_messages(messages):
    converted = []
    for msg in messages:
        if msg["role"] == "system":
            converted.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            converted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            converted.append(AIMessage(content=msg["content"]))
    return converted

# ===============================
# チャット履歴表示
# ===============================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===============================
# ユーザー入力と応答処理
# ===============================
if prompt := st.chat_input("AIに聞きたいことを書いてね"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_for_model = convert_messages(st.session_state.messages)
    response = chat.invoke("\n".join([msg.content for msg in messages_for_model]))

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)