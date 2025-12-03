#!/usr/bin/env python3
"""Test Gemini file upload"""
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

print("Testing Gemini API...")
print(f"API Key: {api_key[:20]}...")

# Try simple text generation first
print("\n1. Testing text generation...")
model = genai.GenerativeModel('gemini-2.0-flash-exp')
response = model.generate_content("Say hello")
print(f"✓ Text generation works: {response.text[:50]}")

# Try file upload
print("\n2. Testing file upload...")
video_path = "data/videos/youtube_SMRbZEbxepA.mp4"
try:
    video_file = genai.upload_file(video_path)
    print(f"✓ File uploaded: {video_file.name}")
    print(f"  State: {video_file.state}")
except Exception as e:
    print(f"✗ File upload failed: {e}")
    print(f"  Error type: {type(e).__name__}")
