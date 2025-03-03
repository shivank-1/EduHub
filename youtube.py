from pytubefix import YouTube
from pytubefix.cli import on_progress
from pathlib import Path
from pydub import AudioSegment
from openai import OpenAI
import streamlit as st
import os

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def download_youtube_audio(url):
    """Download audio from a YouTube video."""
    yt = YouTube(url, on_progress_callback=on_progress)
    print(f"Downloading: {yt.title}")
    
    ys = yt.streams.get_audio_only()
    output_path=ys.download()
    wav=AudioSegment.from_file(output_path).export(output_path[:-4]+".wav",format="wav")
    output_path=output_path[:-4]+".wav"
    print(f"Audio downloaded and saved as {output_path}")
    return output_path

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI's Whisper model."""
    print("Generating Transcription...")
    
    # Load the audio file
    audio = AudioSegment.from_file(audio_path, format="wav")
    
    # Define chunk length in milliseconds (e.g., 1 minute = 60,000 ms)
    chunk_length_ms = 60000
    chunks = make_chunks(audio, chunk_length_ms)

    # Iterate over each chunk
    full_transcription = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}")
        chunk_path = f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        
        with open(chunk_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            full_transcription += transcription.text + " "
        
        # Clean up chunk file
        os.remove(chunk_path)
    
    return full_transcription

def extract_summary_and_questions(transcription):
    """Extract summary and questions from the transcription using GPT-3.5."""
    prompt = f"""
    Below is a transcription of a meeting or discussion. Please provide a concise summary of the key points and list any questions that were asked during the discussion.

    Transcription:
    {transcription}

    Summary:
    - 

    Questions:
    - 
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def make_chunks(audio, chunk_length_ms):
    """Split audio into chunks."""
    return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

def transcribe_from_audio(video_id):
    # YouTube video URL
    url = "https://youtu.be/{}".format(video_id)
    
    # Download audio from YouTube
    audio_output_path = download_youtube_audio(url)
    
    
    # Transcribe the audio
    transcription = transcribe_audio(audio_output_path)
    print("Transcription completed.")
    return transcription
    
    # Extract summary and questions
    #summary_and_questions = extract_summary_and_questions(transcription)
    #print("\nSummary and Questions:")
    #print(summary_and_questions)
    #return summary_and_questions

#if __name__ == "__main__":
 #   main()