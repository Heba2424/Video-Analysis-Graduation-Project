import os
import cv2
import torch
import streamlit as st
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration, pipeline
from sentence_transformers import SentenceTransformer, util
from ultralytics import YOLO
import matplotlib.pyplot as plt


# Function to save frames to folder
def save_frames(video_path, output_folder):
    # Load YOLOv8 model with custom weights for violence detection
    model = YOLO('D:\Skills\Traning\Generative AI\video analysis\violence_weights.pt')

    # Run inference on the video
    results = model(video_path, stream=True)

    # Create folder if doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process and save frames
    for idx, result in enumerate(results):
        annotated_frame = result.plot(boxes=False)
        image_filename = os.path.join(output_folder, f'frame_{idx:04d}.png')
        cv2.imwrite(image_filename, annotated_frame)
    
    return output_folder

# Load BLIP-2 model for caption generation
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
caption_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")

# Function to generate caption for a frame
def generate_caption(frame):
    inputs = processor(images=frame, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        generated_ids = caption_model.generate(**inputs, max_length=50)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# Load BART summarizer pipeline for report generation
summarizer = pipeline('summarization', model="facebook/bart-large-cnn", device=0)

# Function to summarize text chunks
def condense_text_with_prompt(chunks):
    condensed = []
    for chunk in chunks:
        prompt = f"Please summarize the following text, keeping only the key points: \n\n{chunk}"
        summary = summarizer(prompt, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        condensed.append(summary)
    return ' '.join(condensed)

# Chunk text for summarization
def chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

# Function to load captions from text file
def load_captions(caption_file):
    captions = {}
    with open(caption_file, 'r', encoding='utf-8') as f:
        for line in f:
            if ": " in line:
                frame, caption = line.strip().split(": ", 1)
                captions[frame.strip()] = caption.strip()
    return captions

# Precompute embeddings for the frame captions
def compute_caption_embeddings(captions, model):
    return {frame: model.encode(caption) for frame, caption in captions.items()}

# Search function to find best matching frame based on a query
def search_frame_by_caption(query, caption_embeddings, model):
    query_embedding = model.encode(query)
    frame_similarities = {frame: util.pytorch_cos_sim(query_embedding, caption_embedding).item()
                          for frame, caption_embedding in caption_embeddings.items()}
    best_frame = max(frame_similarities, key=frame_similarities.get)
    return best_frame

# Function to display frame using PIL and Streamlit
def display_frame(frame_path):
    img = Image.open(frame_path)
    st.image(img)

# Main Streamlit app function
def main():
    st.title("Video Processing and Caption-based Frame Search")
    
    # Upload video section
    st.header("Upload Video to Generate Report")
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])
    
    if video_file:
        # Process the video and extract frames
        output_folder = 'D:\Skills\Traning\Generative AI\video analysis\extracted_frames'
        st.write("Processing video...")
        output_folder = save_frames(video_file, output_folder)
        
        # Generate captions for each frame and save to text file
        captions_file = 'D:\Skills\Traning\Generative AI\video analysis\captions.txt3'
        with open(captions_file, 'w') as f:
            frames_folder = output_folder
            frame_files = [os.path.join(frames_folder, file) for file in os.listdir(frames_folder) if file.endswith(('.png', '.jpg'))]
            for idx, frame_path in enumerate(frame_files):
                frame = cv2.imread(frame_path)
                caption = generate_caption(frame)
                f.write(f"Frame {idx:04d}: {caption}\n")
        
        # Generate summarized report from captions
        st.write("Generating report...")
        with open(captions_file, 'r') as f:
            captions_text = f.read()
        text_chunks = chunk_text(captions_text)
        report = condense_text_with_prompt(text_chunks)
        st.write("**Summary Report:**")
        st.write(report)
    
    # Search for frames by input caption section
    st.header("Search Frame by Caption")
    query = st.text_input("Enter a caption to search for the best matching frame")
    
    if query:
        # Load precomputed caption embeddings and find best frame
        st.write("Searching for best matching frame...")
        caption_file = 'D:\Skills\Traning\Generative AI\video analysis\captions.txt3'
        frame_captions = load_captions(caption_file)
        caption_embeddings = compute_caption_embeddings(frame_captions, SentenceTransformer('paraphrase-MiniLM-L6-v2'))
        best_frame = search_frame_by_caption(query, caption_embeddings, SentenceTransformer('paraphrase-MiniLM-L6-v2'))
        
        # Display the best matching frame
        frame_dir = 'D:\Skills\Traning\Generative AI\video analysis\extracted_frames'
        best_frame_path = os.path.join(frame_dir, f"{best_frame.replace('Frame ', 'frame_')}.png")
        display_frame(best_frame_path)

# Run the Streamlit app
if __name__ == "__main__":
    main()
