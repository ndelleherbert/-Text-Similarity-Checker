# app.py
import streamlit as st
from cosine_similarity import cosine_similarity

st.title("🔍 Text Similarity Checker")
st.write("Enter two pieces of text to compare their semantic similarity.")

text1 = st.text_area("Text 1", placeholder="e.g. My pet name is cherry")
text2 = st.text_area("Text 2", placeholder="e.g. My dog is called cherry")

if st.button("Compare"):
    if not text1.strip() or not text2.strip():
        st.warning("Please enter both texts before comparing.")
    else:
        with st.spinner("Generating embeddings..."):
            score = cosine_similarity(text1, text2)

        st.metric(label="Cosine Similarity", value=f"{score:.4f}")

        if score >= 0.85:
            st.success("✅ Very similar")
        elif score >= 0.65:
            st.info("🟡 Somewhat similar")
        else:
            st.error("❌ Not very similar")



