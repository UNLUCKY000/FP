import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

def translate_text(text):
  input_ids = tokenizer(text, return_tensors="pt").input_ids

  outputs = model.generate(input_ids)
  return tokenizer.decode(outputs[0])


st.title("Multi-lingual Translator")

source_lang = st.selectbox("Source Language", ["English", "French", "German", "Spanish"])
target_lang = st.selectbox("Target Language", ["English", "French", "German", "Spanish"])

text = st.text_area("Enter text to translate")

if st.button("Translate"):
  st.write("Translation:")
  st.write(translate_text(text))
