import streamlit as st

st.set_page_config(page_title="Debug gdown", layout="centered")

st.title("🔧 Debugging gdown import")

try:
    import gdown
    st.success(f"gdown imported successfully! Version: {gdown.__version__}")
except Exception as e:
    st.error(f"Failed to import gdown: {e!r}")
