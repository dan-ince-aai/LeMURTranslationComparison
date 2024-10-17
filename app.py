import streamlit as st
import assemblyai as aai
from deep_translator import GoogleTranslator

aai.settings.api_key = st.secrets["assemblyai_api_key"]

def process_google_translate(transcript, original_language, target_language):

    try:
        translator = GoogleTranslator(source=original_language, target=target_language)
        translated_text = translator.translate(transcript.text)
        return translated_text
    except Exception as e:
        st.error(f"Translation failed: {e}")
        return ""

def process_llm_translation(transcript, original_language, target_language):

    try:
        prompt = (
            f"Translate the following text from {original_language} to {target_language} with exceptional accuracy. Do not include any preamble."
        )
        print(prompt)
        result = transcript.lemur.task(
            prompt=prompt,
            final_model=aai.LemurModel.claude3_haiku,
            max_output_size=4000
        )
        return result.response
    except Exception as e:
        st.error(f"LLM Translation failed: {e}")
        return ""

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Translation Quality Comparison",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header Section
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Translation Quality Comparison</h1>
            <p>Compare translations from Google Translate and Claude Haiku</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    # Sidebar for Language Selection
    st.sidebar.header("🔧 Language Settings")
    languages = {
        'English': 'en',
        'Spanish': 'es',
        'French': 'fr',
        'German': 'de',
        'Chinese': 'zh',
        'Japanese': 'ja',
        'Korean': 'ko',
        'Italian': 'it',
        'Portuguese': 'pt',
        'Russian': 'ru',
        # Add more languages as needed
    }

    original_language = st.sidebar.selectbox(
        "Original Language",
        options=list(languages.keys()),
        format_func=lambda x: x
    )

    target_language = st.sidebar.selectbox(
        "Target Language",
        options=list(languages.keys()),
        format_func=lambda x: x
    )

    original_language_code = languages[original_language]
    target_language_code = languages[target_language]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Instructions:**
        - Upload an audio file to transcribe.
        - Click "Process" to perform translation and view differences.
        """
    )

    # File Upload Section
    st.header("📁 Upload Audio File")
    st.markdown("Upload an audio file to transcribe its content for translation comparison.")
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=["wav", "mp3", "mp4", "m4a"], 
        help="Supported formats: wav, mp3, mp4, m4a."
    )

    # Centered Process Button
    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
    with button_col2:
        process_button = st.button("🔄 Process", use_container_width=True)

    # Placeholder for Results
    placeholder = st.empty()

    if process_button:
        if not uploaded_file:
            st.error("⚠️ Please upload an audio file before processing.")
        else:
            with placeholder.container():
                # Transcription Section
                st.subheader("📝 Original Transcript")
                with st.spinner("Transcribing audio file..."):
                    try:
                        transcript = aai.Transcriber().transcribe(uploaded_file)
                        st.text_area("Transcribed Text:", value=transcript.text, height=400)
                        st.success("✅ Transcription completed!")
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")


                # Translation Section
                if transcript:
                    st.subheader("🈯️ Translations")
                    with st.spinner("🟢 Translating with Google Translate..."):
                        google_translated_text = process_google_translate(
                            transcript, original_language_code, target_language_code
                        )
                        st.text_area("Google Translated Text:", value=google_translated_text, height=500)
                        st.success("✅ Google Translate completed!")

                    with st.spinner("🤖 Translating with Claude Haiku..."):
                        llm_translated_text = process_llm_translation(
                            transcript, original_language_code, target_language_code
                        )
                        st.text_area("Claude Haiku Translated Text:", value=llm_translated_text, height=500)
                        st.success("✅ Claude Haiku Translation completed!")

    # Footer Section
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <p>Built by AssemblyAI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
