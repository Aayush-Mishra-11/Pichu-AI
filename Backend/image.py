import cv2
import pytesseract
import os

def analyze_image_text_and_faces(image_path):
    if not image_path or not os.path.exists(image_path):
        return "Error: No image file provided or file not found."

    try:
        text = extract_text_from_image(image_path)
        face_count = detect_faces(image_path)

        result_parts = []
        result_parts.append(f"Extracted text: {text.strip() if text.strip() else 'No text found.'}")
        result_parts.append(f"Faces detected: {face_count if face_count is not None else 'Error detecting faces.'}")

        return "\n".join(result_parts)
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def extract_text_from_image(image_path):
    try:
        pil_img = cv2.imread(image_path)
        if pil_img is None:
            # Try with PIL as fallback
            from PIL import Image
            pil_img = Image.open(image_path)
            text = pytesseract.image_to_string(pil_img)
        else:
            # Use OpenCV image
            text = pytesseract.image_to_string(pil_img)
        return text.strip()
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def detect_faces(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(imgGray, 1.2, 5)
        return len(faces)
    except Exception as e:
        print(f"Face detection error: {e}")
        return 0

# Text-to-image generation (requires CUDA GPU)
def text_to_image(prompt):
    try:
        from diffusers import StableDiffusionPipeline
        import torch

        print(f"Generating image for: {prompt}")
        model_id = "sd-legacy/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        image = pipe(prompt).images[0]
        output_path = "generated_image.png"
        image.save(output_path)

        # Try to display the image
        try:
            image.show()
        except Exception:
            pass

        return f"Image generated and saved as '{output_path}'. The image is also displayed on your screen."
    except Exception as e:
        return f"Error generating image: {str(e)}. Image generation requires a CUDA GPU."
