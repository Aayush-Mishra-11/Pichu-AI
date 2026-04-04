import sys
import datetime
import webbrowser

# Only import speech related modules when running as CLI
if sys.platform == "win32" and len(sys.argv) > 1 and sys.argv[1] == "--cli":
    try:
        import speech_recognition as sr
        import pyttsx3
        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty('voices')
        if len(voices) > 2:
            engine.setProperty('voice', voices[2].id)

        def speak(text):
            print(f"Pichu: {text}")
            engine.say(text)
            engine.runAndWait()

        def take_command():
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Listening...")
                r.pause_threshold = 1
                r.adjust_for_ambient_noise(source, duration=1)
                audio = r.listen(source)
            try:
                print("Recognizing...")
                query = r.recognize_google(audio, language='en-in')
                print(f"==> Master : {query}")
                return query
            except Exception:
                return ""
    except ImportError:
        def speak(text):
            print(f"Pichu: {text}")
        def take_command():
            return ""
else:
    def speak(text):
        print(f"Pichu: {text}")
    def take_command():
        return ""

from Model import PichuModel
from Automation import PichuAI, play_music, play_specific_song, list_songs, search_google, search_wiki, search_youtube, search_chatgpt
import os_utils

# --- API Configuration ---
API_KEY = "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"

# Initialize once at module load
pichu_model = PichuModel(api_key=API_KEY)
automation = PichuAI(api_key=API_KEY)

def handle_query(query):
    query_lower = query.lower().strip()
    if not query:
        return "I didn't receive any input. Please try again."

    # ---- OS Commands ----
    if "open notepad" in query_lower or "start notepad" in query_lower:
        return os_utils.open_app("notepad")
    elif "open calculator" in query_lower or "start calculator" in query_lower or "calculate" in query_lower:
        return os_utils.open_app("calculator")
    elif "open command prompt" in query_lower or "open cmd" in query_lower or "start cmd" in query_lower:
        return os_utils.open_app("cmd")
    elif "open file explorer" in query_lower or "open explorer" in query_lower or "open files" in query_lower or "start explorer" in query_lower:
        return os_utils.open_app("explorer")
    elif "open settings" in query_lower or "open control panel" in query_lower:
        return os_utils.open_app("settings")
    elif "open chatgpt" in query_lower or "open chat gpt" in query_lower or "start chatgpt" in query_lower:
        return os_utils.open_app("chatgpt")
    elif "open brave" in query_lower or "open browser" in query_lower or "start brave" in query_lower or "start browser" in query_lower:
        return os_utils.open_app("brave")

    elif "shutdown" in query_lower or "shut down" in query_lower:
        os_utils.shutdown()
        return "Shutting down the system."
    elif "restart" in query_lower or "reboot" in query_lower:
        os_utils.restart()
        return "Restarting the system."
    elif "sleep" in query_lower or "hibernate" in query_lower:
        os_utils.sleep()
        return "Putting the system to sleep."

    elif "increase brightness" in query_lower or "brightness up" in query_lower or "make screen brighter" in query_lower or "brightness increase" in query_lower:
        os_utils.increase_brightness()
        return "Brightness increased."
    elif "decrease brightness" in query_lower or "brightness down" in query_lower or "make screen darker" in query_lower or "brightness decrease" in query_lower:
        os_utils.decrease_brightness()
        return "Brightness decreased."

    elif "increase volume" in query_lower or "volume up" in query_lower or "louder" in query_lower or "turn up volume" in query_lower or "volume increase" in query_lower:
        os_utils.increase_volume()
        return "Volume increased."
    elif "decrease volume" in query_lower or "volume down" in query_lower or "quieter" in query_lower or "turn down volume" in query_lower or "volume decrease" in query_lower:
        os_utils.decrease_volume()
        return "Volume decreased."
    elif "mute volume" in query_lower or "mute" in query_lower or "silent" in query_lower or "mute sound" in query_lower:
        os_utils.mute_volume()
        return "Volume muted."
    elif "unmute volume" in query_lower or "unmute" in query_lower or "unmute sound" in query_lower:
        os_utils.unmute_volume()
        return "Volume unmuted."

    elif "turn on wifi" in query_lower or "enable wifi" in query_lower or "wifi on" in query_lower or "connect wifi" in query_lower:
        os_utils.handle_wifi_bluetooth("turn on wifi", lambda x: x)
        return "WiFi enabled."
    elif "turn off wifi" in query_lower or "disable wifi" in query_lower or "wifi off" in query_lower or "disconnect wifi" in query_lower:
        os_utils.handle_wifi_bluetooth("turn off wifi", lambda x: x)
        return "WiFi disabled."
    elif "turn on bluetooth" in query_lower or "enable bluetooth" in query_lower or "bluetooth on" in query_lower:
        os_utils.handle_wifi_bluetooth("turn on bluetooth", lambda x: x)
        return "Bluetooth enabled."
    elif "turn off bluetooth" in query_lower or "disable bluetooth" in query_lower or "bluetooth off" in query_lower:
        os_utils.handle_wifi_bluetooth("turn off bluetooth", lambda x: x)
        return "Bluetooth disabled."

    # ---- Music Commands ----
    elif "play music" in query_lower or "play song" in query_lower or "play songs" in query_lower or "start music" in query_lower:
        return play_music()
    elif "next song" in query_lower or "next music" in query_lower or "skip song" in query_lower or "skip this song" in query_lower or "next track" in query_lower:
        return play_music(next_song=True)
    elif "list of songs" in query_lower or "list of music" in query_lower or "show songs" in query_lower or "what songs" in query_lower or "song list" in query_lower or "all songs" in query_lower:
        return list_songs()
    elif query_lower.startswith("play ") and len(query.split()) > 1:
        return play_specific_song(query)

    # ---- Automation & Info ----
    elif "what time" in query_lower or "time now" in query_lower or "current time" in query_lower or "tell me the time" in query_lower:
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}."
    elif "what date" in query_lower or "today date" in query_lower or "what's today's date" in query_lower or "current date" in query_lower:
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        return f"Today's date is {current_date}."

    elif "open google" in query_lower or "google" in query_lower:
        webbrowser.open("https://google.com")
        return "Opening Google."
    elif "search" in query_lower and len(query.split()) > 1:
        return search_google(query)
    elif "open youtube" in query_lower or "youtube" in query_lower:
        return search_youtube(query)
    elif "open chatgpt" in query_lower or "chat gpt" in query_lower:
        return search_chatgpt(query)
    elif "wikipedia" in query_lower or "who is" in query_lower or "what is" in query_lower:
        result = search_wiki(query)
        if result:
            return result

    elif "weather" in query_lower or "temperature" in query_lower:
        city = automation._extract_city(query) or "Bilaspur"
        return automation.get_weather(city)

    # ---- Image Analysis Commands ----
    elif "analyze image" in query_lower or "analyze this image" in query_lower or "detect image" in query_lower or "text from image" in query_lower or "image recognition" in query_lower or "read image" in query_lower or "extract text from image" in query_lower or "what's in this image" in query_lower or "describe this image" in query_lower or "ocr this" in query_lower or "scan this" in query_lower:
        return "To analyze an image, please upload it using the attachment button and I'll extract text and detect faces."

    # ---- Image Generation Commands ----
    elif "generate an image" in query_lower or "generate image" in query_lower or "text to image" in query_lower or "create an image" in query_lower or "create image" in query_lower or "image generation" in query_lower or "draw" in query_lower or "make an image" in query_lower or "make image" in query_lower or "generate a picture" in query_lower or "generate picture" in query_lower or "create a picture" in query_lower or "create picture" in query_lower or "ai image" in query_lower:
        triggers = ["generate an image of ", "generate image of ", "generate a picture of ", "create an image of ", "create image of ", "create a picture of ", "draw ", "make an image of ", "make image of ", "text to image "]
        prompt = query
        for trigger in triggers:
            if trigger in query_lower:
                prompt = query.split(trigger, 1)[1].strip()
                break
        if prompt == query or not prompt:
            return "What would you like me to generate an image of? Try saying 'generate an image of a sunset'"
        from image import text_to_image
        return text_to_image(prompt)

    # ---- History ----
    elif "past conversation" in query_lower or "history" in query_lower or "show conversation" in query_lower or "previous messages" in query_lower or "conversation history" in query_lower:
        history = automation.get_history()
        if not history:
            return "No conversation history found."
        formatted = []
        for idx, item in enumerate(history[-10:], 1):
            role = item.get("role", "user")
            content = item.get("content", "")
            formatted.append(f"{idx}. {role.capitalize()}: {content[:100]}...")
        return "\n".join(formatted)

    # ---- Intent-based responses ----
    response = pichu_model.get_intent_response(query)
    if response:
        automation.conversation_history.append({"role": "user", "content": query})
        automation.conversation_history.append({"role": "assistant", "content": response})
        return response

    # ---- Fallback to NVIDIA AI ----
    automation.conversation_history.append({"role": "user", "content": query})
    response = pichu_model.handle_ai_query(query)
    automation.conversation_history.append({"role": "assistant", "content": response})
    return response

# --- CLI Entry Point ---
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--cli":
    hour = int(datetime.datetime.now().hour)
    greeting = "Good Morning!" if hour < 12 else "Good Afternoon!" if hour < 18 else "Good Evening!"
    speak(greeting)
    speak("I am Pichu. Please tell me how may I help you")

    while True:
        query = take_command()
        if not query:
            continue
        if 'quit' in query.lower() or 'exit' in query.lower():
            speak("Goodbye!")
            break
        response = handle_query(query)
        speak(response)
