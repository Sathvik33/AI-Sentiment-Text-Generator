# Import the required libraries
import streamlit as st
import model
import logging

# Configure the Streamlit page's title, icon, and layout
st.set_page_config(
    page_title="AI Sentiment Text Generator (No API)",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)
# Display the main title of the application
st.title("🤖 AI Sentiment Text Generator ")
# Add a short description of the app's function
st.write(
    "Enter a topic. A local AI will analyze its sentiment and write text to match."
)

# Create a sidebar for configuration options
with st.sidebar:
    # Add a header to the sidebar
    st.header("Configuration")
    # Create radio buttons for selecting the output length
    output_length = st.radio(
        "Select Output",
        ('Paragraph', 'Essay'),
        horizontal=True
    )
    
    # Add a visual separator line
    st.markdown("---")

# Create a text area for the user to input their prompt
prompt = st.text_area("Enter your prompt or topic here:", height=150, placeholder="e.g., 'The excitement of learning a new skill' or 'The frustration of traffic jams'")

# Create a primary button that spans the container width to trigger the process
if st.button("Generate Text", type="primary", use_container_width=True):
    # Check if the user has entered any text
    if not prompt.strip():
        # Display an error if the prompt is empty
        st.error("Please enter a prompt.")
    else:
        # Use a try-except block to gracefully handle potential errors
        try:
            # Show a loading spinner while the AI is working
            with st.spinner('AI models are loading and generating text... This may take a few minutes on first run...'):
                
                # Inform the user that sentiment analysis is in progress
                st.write("Analyzing sentiment...")
                # Call the sentiment analysis function from the model.py file
                sentiment = model.analyze_sentiment_local(prompt)

                # Display a subheader for the results section
                st.subheader("Results")
                # Display the detected sentiment with color-coding based on the result
                if sentiment == 'positive':
                    st.success(f"**Detected Sentiment:** Positive")
                elif sentiment == 'negative':
                    st.error(f"**Detected Sentiment:** Negative")
                else:
                    st.info(f"**Detected Sentiment:** Neutral")

                # Inform the user that text generation is in progress
                st.write("Generating text...")
                # Call the text generation function from model.py
                generated_text = model.generate_text_local(
                    prompt, sentiment, output_length
                )
                
            # Show a success message once the process is complete
            st.success("Generation complete!")
            # Add a header for the generated text output
            st.markdown("### Generated Text")
            # Display the final generated text
            st.write(generated_text)

        # Catch specific, expected errors during the process
        except (ConnectionError, RuntimeError, ValueError) as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Caught an error in Streamlit app: {e}")
        # Catch any other unexpected errors that might occur
        except Exception as e:
            st.error(f"An unexpected critical error occurred: {e}")
            logging.error(f"Caught an unexpected error in Streamlit app: {e}")
