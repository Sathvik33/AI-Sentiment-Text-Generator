import streamlit as st
import model
import logging

st.set_page_config(
    page_title="AI Sentiment Text Generator (No API)",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("🤖 AI Sentiment Text Generator ")
st.write(
    "Enter a topic. A local AI will analyze its sentiment and write text to match."
)

with st.sidebar:
    st.header("Configuration")
    output_length = st.radio(
        "Select Output",
        ('Paragraph', 'Essay'),
        horizontal=True
    )
    
    st.markdown("---")
 

prompt = st.text_area("Enter your prompt or topic here:", height=150, placeholder="e.g., 'The excitement of learning a new skill' or 'The frustration of traffic jams'")

if st.button("Generate Text", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        try:
            with st.spinner('AI models are loading and generating text... This may take a few minutes on first run...'):
                
                st.write("Analyzing sentiment...")
                sentiment = model.analyze_sentiment_local(prompt)

                st.subheader("Results")
                if sentiment == 'positive':
                    st.success(f"**Detected Sentiment:** Positive")
                elif sentiment == 'negative':
                    st.error(f"**Detected Sentiment:** Negative")
                else:
                    st.info(f"**Detected Sentiment:** Neutral")

                st.write("Generating text...")
                generated_text = model.generate_text_local(
                    prompt, sentiment, output_length
                )
                
            st.success("Generation complete!")
            st.markdown("### Generated Text")
            st.write(generated_text)

        except (ConnectionError, RuntimeError, ValueError) as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Caught an error in Streamlit app: {e}")
        except Exception as e:
            st.error(f"An unexpected critical error occurred: {e}")
            logging.error(f"Caught an unexpected error in Streamlit app: {e}")


