from PIL import Image
import base64
from io import BytesIO

buffer = BytesIO()

def open_pil(path):
    img = Image.open(path).convert("RGB")
    return resize_store_ratio(img)

def pil_to_b64(img):
    img=img.convert("RGB")
    img.save(buffer, format="jpeg")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_image

def resize_store_ratio(image, min_side=256):

    width, height = image.size

    if width < height:
        new_width = min_side
        new_height = int((height / width) * min_side)
    else:
        new_width = int((width / height) * min_side)
        new_height = min_side

    resized_image = image.resize((new_width, new_height))

    return resized_image