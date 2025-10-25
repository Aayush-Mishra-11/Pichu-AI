import cv2
import pytesseract
from tkinter import Tk, filedialog
from PIL import Image
from main import speak

from diffusers import StableDiffusionPipeline
import torch

def select_image_via_dialog():
    root = Tk()
    root.withdraw()
    speak("Please select an image file.")
    file_path = filedialog.askopenfilename(
        title="Select an image",
        initialdir="/",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    root.destroy()
    return file_path

def analyze_image_text_and_faces(image_path=None):
    if image_path is None:
        file_path = select_image_via_dialog()
    else:
        file_path = image_path
    if not file_path:
        speak("No file selected.")
        return "No file selected."

    speak("Working on it! Please wait.")
    text = extract_text_from_image(file_path)
    face_count = detect_faces(file_path)

    summary = []
    summary.append(f"Extracted text: {text or 'No text found.'}")
    if face_count is None:
        summary.append("Faces detected: Error loading image.")
    # Truncate extracted text for speaking if too long
    spoken_text = text if len(text) <= 200 else text[:200] + "..."
    spoken_result = f"Extracted text: {spoken_text or 'No text found.'}\nFaces detected: {face_count}"
    speak(spoken_result)
    summary.append(f"Faces detected: {face_count}")
    result = "\n".join(summary)
    return result

def extract_text_from_image(image_path):
    pil_img = Image.open(image_path)
    text = pytesseract.image_to_string(pil_img)
    return text.strip()

def detect_faces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image for face detection.")
        return None
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(imgGray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return len(faces)

# Image generation ----  with the help of stable-diffusion v1.5 and after doing some changes in commmand now this can genrate images,
#The model which i have used here is from Hugging Face name Stable Diffusion v1.5 which is for Text to Image 
def text_to_image(prompt):
    speak(f"Okay, creating an image of '{prompt}'. This will take some time, please wait.")
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(prompt).images[0]
    image.show()
    return "Image generation complete. The image is now displayed."
