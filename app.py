import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

st.title("T5 Text Translation")

source_text = st.text_input("Enter text to translate:")
target_language = st.selectbox("Select target language:", ["German", "French", "Spanish", "Chinese"])

if st.button("Translate"):
    input_ids = tokenizer(source_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    translated_text = tokenizer.decode(outputs[0])

    st.write(f"Translated text ({target_language}): {translated_text}")
