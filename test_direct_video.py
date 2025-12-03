#!/usr/bin/env python3
"""Test sending video directly without upload"""
import os
from pathlib import Path
import google.generativeai as genai

# Load API key
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

print("Testing direct video input (no upload)...")

# Try sending video directly in the prompt
video_path = "data/videos/youtube_SMRbZEbxepA.mp4"

try:
    # Read video file
    with open(video_path, 'rb') as f:
        video_data = f.read()
    
    print(f"Video size: {len(video_data) / 1024 / 1024:.1f} MB")
    
    # Try with gemini-1.5-flash (supports video)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Create a file-like object
    import io
    video_file = io.BytesIO(video_data)
    
    print("Sending video directly to model...")
    response = model.generate_content([
        "What sport is being shown in this video?",
        {"mime_type": "video/mp4", "data": video_data}
    ])
    
    print(f"✓ Response: {response.text[:100]}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    print(f"  Error type: {type(e).__name__}")
