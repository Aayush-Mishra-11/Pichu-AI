# Deploying Pichu AI

## Architecture Options

### Option 1: Full Stack on Render (Recommended for Cloud)

**Frontend + Backend both on Render**

| Component | Platform | Features |
|----------|----------|----------|
| Frontend | Netlify or Render Static | UI, Voice, Spark Effects |
| Backend | Render Web Service | AI, Weather, Web Search |
| **Missing** | - | OS Controls, Music, Image Gen |

### Option 2: Frontend on Netlify + Backend Locally

| Component | Platform | Features |
|----------|----------|----------|
| Frontend | Netlify | Full UI |
| Backend | Your Local PC | **All Features** |

This gives you **full features** but requires local backend running.

---

## Deploy to Render (Full Backend)

### Step 1: Prepare Files

1. Push your code to GitHub
2. Create a new repository or use existing

### Step 2: Deploy Backend to Render

1. Go to [render.com](https://render.com) → Sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo
4. Configure:

   | Setting | Value |
   |---------|-------|
   | **Name** | `pichu-backend` |
   | **Region** | Choose closest |
   | **Branch** | `main` |
   | **Root Directory** | `Backend` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn app:app --host 0.0.0.0 --port $PORT` |

5. Add Environment Variables:
   - `API_KEY` = `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`

6. Click **"Create Web Service"**

7. Wait for deployment (3-5 mins)

8. Note your backend URL: `https://pichu-backend.onrender.com`

### Step 3: Deploy Frontend to Netlify

1. Push `public/` folder to a new GitHub repo
2. Go to [netlify.com](https://netlify.com)
3. Click **"Add new site"** → **"Import an existing project"**
4. Connect your repo
5. Set **Publish directory**: `public`
6. Click **"Deploy"**

### Step 4: Connect Frontend to Backend

1. In your Netlify site → **Site settings** → **Environment variables**
2. Add: `BACKEND_URL` = `https://pichu-backend.onrender.com`

Or edit `public/index.html`:
```javascript
const BACKEND_URL = 'https://pichu-backend.onrender.com';
```

---

## Deploy to Netlify Only (Demo Mode)

If you only want the frontend on Netlify:

1. Push `public/` folder to GitHub
2. Connect to Netlify
3. Deploy

**Features available in demo mode:**
- Chat UI with canned responses
- Voice input/output
- Theme toggle
- Spark effects
- Robot animation

---

## Feature Comparison

| Feature | Render Backend | Local Backend | Demo Only |
|---------|---------------|---------------|-----------|
| AI Chat (NVIDIA) | ✅ | ✅ | ❌ |
| Weather | ✅ | ✅ | ❌ |
| Web Search | ✅ | ✅ | ❌ |
| Voice Input | ✅ | ✅ | ✅ |
| Voice Output | ✅ | ✅ | ✅ |
| Theme Toggle | ✅ | ✅ | ✅ |
| Spark Effects | ✅ | ✅ | ✅ |
| Robot Animation | ✅ | ✅ | ✅ |
| OS Controls | ❌ | ✅ | ❌ |
| Music Playback | ❌ | ✅ | ❌ |
| Image Generation | ❌ | ✅* | ❌ |
| Image Analysis | ❌ | ✅ | ❌ |

*Image Generation requires CUDA GPU

---

## Why Some Features Won't Work on Render

Render runs **Linux containers**, not Windows. These features require Windows:
- Opening Notepad, Calculator, etc.
- Volume/Brightness control
- Playing music files
- Shutdown/Restart/Sleep

**Solution:** Run backend locally for full features, frontend on Netlify.

---

## Quick Setup: Netlify + Local Backend

1. **Deploy frontend to Netlify** (public folder)

2. **Run backend locally:**
```bash
cd Backend
pip install -r requirements.txt
python app.py
```

3. **Update frontend** (`public/index.html`):
```javascript
const BACKEND_URL = 'http://YOUR_PC_IP:10000';
```

4. **Find your PC IP:**
```bash
ipconfig
# Look for IPv4 Address
```

5. **For local access from anywhere**, use ngrok:
```bash
ngrok http 10000
# Use the generated URL as BACKEND_URL
```

---

## File Structure

```
Pichu/
├── Backend/
│   ├── app.py              # Flask entry point
│   ├── main.py             # Query handler
│   ├── Model.py            # AI model (NVIDIA)
│   ├── Automation.py       # Weather, search
│   ├── os_utils.py         # OS controls (local only)
│   ├── image.py           # Image processing
│   ├── requirements.txt     # Python dependencies
│   └── static/             # Static files
│       ├── pichu.png
│       └── audio/
│           └── spark.mp3
│
├── public/                 # Netlify deploy
│   ├── index.html
│   └── static/
│       ├── pichu.png
│       └── audio/
│           └── spark.mp3
│
├── netlify.toml           # Netlify config
└── RENDER_DEPLOY.md      # This file
```
To Push to GitHub

  Run these commands in your project folder:

  # Initialize git (if not already)
  git init

  # Add all files
  git add .

  # Commit
  git commit -m "Updated Pichu AI with all features"

  # Add remote (if not already)
  git remote add origin https://github.com/Aayush-Mishra-11/Pichu-AI.git

  # Push
  git push -u origin main