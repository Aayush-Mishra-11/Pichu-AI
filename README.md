# Pichu AI Assistant

A smart, voice-powered AI assistant with chat interface, image analysis, and system automation features.

## Features

- **AI Chat** - NVIDIA-powered conversational AI
- **Voice Input/Output** - Speak to Pichu and hear responses
- **Image Analysis** - Upload images for text extraction and face detection
- **Image Generation** - Create AI-generated images
- **Weather Info** - Get current weather for any city
- **Web Search** - Search Google, YouTube, Wikipedia
- **System Control** - Open apps, adjust volume/brightness
- **Music Playback** - Play songs from your music library

## Deploy to Render

### 1. Deploy Backend

1. Fork this repository
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Root Directory**: `Backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variable:
   - `API_KEY` = your NVIDIA API key

### 2. Deploy Frontend (Optional - for demo mode)

1. Deploy the `public/` folder to Netlify
2. Set `BACKEND_URL` environment variable to your Render backend URL

## Local Development

```bash
cd Backend
pip install -r requirements.txt
python app.py
```

Open http://localhost:10000

## Tech Stack

- **Backend**: Flask, Python
- **AI**: NVIDIA API (minimaxai/minimax-m2.5)
- **Frontend**: HTML, CSS, JavaScript, Tailwind CSS
- **Features**: Speech Recognition, Web Speech API, OpenCV

## Project Structure

```
Pichu/
├── Backend/
│   ├── app.py           # Flask app
│   ├── main.py          # Query handler
│   ├── Model.py         # AI model
│   ├── Automation.py    # Features
│   ├── os_utils.py     # OS controls
│   ├── image.py        # Image processing
│   ├── static/         # Static files
│   └── templates/      # HTML templates
├── public/             # Netlify deploy
└── RENDER_DEPLOY.md   # Deployment guide
```

## Note

Some features (OS automation, music playback) require Windows and work only with local backend deployment.
