import cv2
print(cv2.__version__)

import numpy as np
from PIL import Image

# Load your image
image_path = "your_image.jpg"
image = cv2.imread(image_path)

# Initialize the super-resolution model
sr = cv2.dnn_superres.DnnSuperResImpl_create()
model_path = "EDSR_x2.pb"  # Path to your pre-trained model
sr.readModel(model_path)
sr.setModel("edsr", 2)  # Model name and upscaling factor

# Upscale the image
upscaled_image = sr.upsample(image)

# Save or display the upscaled image
upscaled_image_path = "upscaled_image.jpg"
cv2.imwrite(upscaled_image_path, upscaled_image)
print(f"Upscaled image saved at: {upscaled_image_path}")