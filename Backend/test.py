import cv2
import pytesseract
from tkinter import Tk, filedialog
from PIL import Image
from main import speak

from diffusers import StableDiffusionPipeline, FluxPipeline
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

# Image generation --- text to image using FluxPipeline (Krea/FLUX model)
def text_to_image(prompt):
    """Generate an image from `prompt` using the FLUX pipeline.

    Keeps the same signature as before. Uses GPU (CUDA) when available, otherwise CPU.
    The generated image is shown (PIL .show()) and saved to `flux-krea-dev.png`.
    """
    speak(f"Okay, creating an image of '{prompt}'. This will take some time, please wait.")

    # choose device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_id = "black-forest-labs/FLUX.1-Krea-dev"
    try:
        # Load the Flux pipeline with the selected dtype
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)

        # Offload model to CPU when helpful to save VRAM; safe to call even on CUDA
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

        # Move pipeline to the desired device
        try:
            pipe = pipe.to(device)
        except Exception:
            # Some pipelines handle device placement internally; continue
            pass

        # Generate the image
        output = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=4.5,
        )

        image = output.images[0]

        # Show and save the image
        try:
            image.show()
        except Exception:
            # show may not work in some headless environments; continue
            pass
        try:
            image.save("flux-krea-dev.png")
        except Exception as e:
            print("Warning: failed to save image:", e)

        return "Image generation complete. The image is now saved as flux-krea-dev.png and displayed."

    except Exception as e:
        speak("Image generation failed due to an error.")
        print("Error during image generation:", e)
        return f"Image generation failed: {e}"
