import sys
import os
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

if sys.platform != 'win32':
    raise EnvironmentError("This script is intended to run on Windows only.")

def mute_volume():
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMute(1, None)
        return True
    except Exception as e:
        print(f"Failed to mute: {e}")
        return False

def unmute_volume():
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMute(0, None)
        return True
    except Exception as e:
        print(f"Failed to unmute: {e}")
        return False

def increase_volume():
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current = volume.GetMasterVolumeLevelScalar()
        volume.SetMasterVolumeLevelScalar(min(current + 0.1, 1.0), None)
        return True
    except Exception as e:
        print(f"Failed to increase volume: {e}")
        return False

def decrease_volume():
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current = volume.GetMasterVolumeLevelScalar()
        volume.SetMasterVolumeLevelScalar(max(current - 0.1, 0.0), None)
        return True
    except Exception as e:
        print(f"Failed to decrease volume: {e}")
        return False

def increase_brightness():
    try:
        current = sbc.get_brightness(display=0)
        sbc.set_brightness(min(current[0] + 10, 100), display=0)
        return True
    except Exception as e:
        print(f"Failed to increase brightness: {e}")
        return False

def decrease_brightness():
    try:
        current = sbc.get_brightness(display=0)
        sbc.set_brightness(max(current[0] - 10, 0), display=0)
        return True
    except Exception as e:
        print(f"Failed to decrease brightness: {e}")
        return False

def handle_wifi_bluetooth(query, speak_func=None):
    def safe_speak(msg):
        if speak_func:
            speak_func(msg)
        else:
            print(msg)

    if 'turn on wifi' in query or 'enable wifi' in query or 'connect wifi' in query:
        try:
            os.system("netsh interface set interface Wi-Fi enabled")
            safe_speak("WiFi enabled.")
            return True
        except Exception:
            safe_speak("Sorry, I couldn't enable WiFi.")
            return False
    elif 'turn off wifi' in query or 'disable wifi' in query or 'disconnect wifi' in query:
        try:
            os.system("netsh interface set interface Wi-Fi disabled")
            safe_speak("WiFi disabled.")
            return True
        except Exception:
            safe_speak("Sorry, I couldn't disable WiFi.")
            return False
    elif 'turn on bluetooth' in query or 'enable bluetooth' in query:
        try:
            os.system("powershell Start-Service bthserv")
            safe_speak("Bluetooth enabled.")
            return True
        except Exception:
            safe_speak("Sorry, I couldn't enable Bluetooth.")
            return False
    elif 'turn off bluetooth' in query or 'disable bluetooth' in query:
        try:
            os.system("powershell Stop-Service bthserv")
            safe_speak("Bluetooth disabled.")
            return True
        except Exception:
            safe_speak("Sorry, I couldn't disable Bluetooth.")
            return False
    return False

def open_app(app_name):
    apps = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "cmd": "cmd.exe",
        "explorer": os.path.expanduser("~"),
        "settings": "ms-settings:",
        "chatgpt": "https://chat.openai.com",
        "brave": "brave.exe"
    }

    if app_name in apps:
        try:
            app_path = apps[app_name]
            if app_name == "chatgpt":
                import webbrowser
                webbrowser.open(app_path)
            elif app_name == "explorer":
                os.startfile(app_path)
            else:
                os.startfile(app_path)
            return f"Opening {app_name}."
        except Exception as e:
            return f"Failed to open {app_name}: {str(e)}"
    return f"App '{app_name}' not found."

def shutdown():
    try:
        os.system("shutdown /s /t 1")
        return True
    except Exception as e:
        print(f"Failed to shutdown: {e}")
        return False

def restart():
    try:
        os.system("shutdown /r /t 1")
        return True
    except Exception as e:
        print(f"Failed to restart: {e}")
        return False

def sleep():
    try:
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        return True
    except Exception as e:
        print(f"Failed to sleep: {e}")
        return False
