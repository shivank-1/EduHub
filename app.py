import os
import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import google.generativeai as genai
from youtube import transcribe_from_audio
# Load environment variables
#load_dotenv()
groqkey = st.secrets["GROQ_API_KEY"]
gemini_api_key = st.secrets["GEMENI_API_KEY"]

# Streamlit App Configuration
st.set_page_config(page_title="Structured Notes Generator", page_icon="📝")
st.title("📝 Structured Notes Generator")

# Sidebar for selecting functionality
st.sidebar.title("Choose Functionality")
option = st.sidebar.radio("Select an option:", ["YouTube Video Notes", "PDF Note Generator"])

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groqkey, 
    model_name="llama-3.3-70b-versatile", 
    temperature=0.7
)

# Initialize Gemini
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Function to extract YouTube video ID
def extract_youtube_video_id(url):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?&]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Function to retrieve YouTube transcript
def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ' '.join([entry['text'] for entry in transcript])
        return full_transcript[:10000]  # Limit to first 10000 characters
    except Exception as e:
        #st.error(f"Error retrieving transcript: {e}")
        return transcribe_from_audio(video_id)

# Function to convert PDF page to image
def convert_pdf_page_to_image(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

# Function to extract text and questions from scanned PDF using Gemini
def extract_text_and_questions_from_scanned_pdf(pdf_path):
    """
    Extracts text and questions from a scanned PDF using Gemini Vision.
    Returns two dictionaries: one for notes and one for questions.
    """
    try:
        doc = fitz.open(pdf_path)
        extracted_notes = {}
        extracted_questions = {}

        for page_num in range(len(doc)):
            # Convert PDF page to image
            img = convert_pdf_page_to_image(pdf_path, page_num)
            
            # Prompt for extracting notes
            notes_prompt = """
            Use this image as a context and generate structured notes for educational purposes.
            Include any diagrams or shapes by describing them in detail.
            """
            notes_response = gemini_model.generate_content([notes_prompt, img])
            extracted_notes[page_num + 1] = notes_response.text.strip() if notes_response.text else ""
            
            # Prompt for extracting questions
            questions_prompt = """
            Extract all the questions mentioned in this image. 
            If the questions contain diagrams or shapes, describe them in detail.
            List the questions clearly and separately.
            """
            questions_response = gemini_model.generate_content([questions_prompt, img])
            extracted_questions[page_num + 1] = questions_response.text.strip() if questions_response.text else ""

        doc.close()
        return extracted_notes, extracted_questions
    except Exception as e:
        raise Exception(f"Error analyzing PDF: {str(e)}")

# Function to combine and structure all extracted content
def combine_and_structure_content(transcript=None, notes=None, questions=None):
    """
    Combines transcript, notes, and questions into a single structured output.
    """
    combined_content = ""

    # Add transcript summary if available
    if transcript:
        combined_content += "### Video Transcript Summary\n"
        combined_content += transcript + "\n\n"

    # Add notes if available
    if notes:
        combined_content += "### Extracted Notes\n"
        for page_num, text in notes.items():
            combined_content += f"#### Page {page_num}\n"
            combined_content += text + "\n\n"

    # Add questions if available
    if questions:
        combined_content += "### Extracted Questions\n"
        for page_num, text in questions.items():
            if text:  # Only add if questions are found
                combined_content += f"#### Page {page_num}\n"
                combined_content += text + "\n\n"

    return combined_content

# Main function for YouTube Video Summarizer
def youtube_summarizer():
    st.header("YouTube Video Notes")
    youtube_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=example")
    
    if st.button("Generate Notes"):
        if not youtube_url:
            st.warning("Please enter a YouTube URL")
            return
        
        video_id = extract_youtube_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please check the link.")
            return
        
        with st.spinner('Extracting transcript and generating summary...'):
            transcript = get_youtube_transcript(video_id)
            if not transcript:
                st.error("Could not retrieve video transcript. The video might not have captions.")
                return
            
            docs = [Document(page_content=transcript)]
            prompt_template = """
            provide a small description of what the video has to tell then
            Provide easy to understand notes 
            and in the end provide a clear short summary at the end of the following YouTube video transcript. 
            Capture the main points, key insights, and overall message of the video. 
            Ensure the summary is concise, informative, and approximately 300-350 words.

            Transcript:{text}
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            output_summary = chain.run(docs)
            
            # Combine and structure the output
            structured_output = combine_and_structure_content(transcript=output_summary)
            
            st.success("Structured Output:")
            st.write(structured_output)

# Main function for PDF Note Generator
def pdf_note_generator():
    st.header("PDF Note Generator")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner('Extracting text, notes, and questions...'):
            try:
                # Save the uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract notes and questions from PDF
                extracted_notes, extracted_questions = extract_text_and_questions_from_scanned_pdf("temp.pdf")
                
                # Combine and structure the output
                structured_output = combine_and_structure_content(notes=extracted_notes, questions=extracted_questions)
                
                st.success("Structured Output:")
                st.write(structured_output)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Run the selected functionality
if option == "YouTube Video Summarizer":
    youtube_summarizer()
elif option == "PDF Note Generator":
    pdf_note_generator()