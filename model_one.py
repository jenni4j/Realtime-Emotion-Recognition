from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import sys

# Read in the file from the command line
filename = sys.argv[1]
image = Image.open(filename).convert("RGB")

# Load the image processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Load the model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Process the input PIL.Image into a tensor
inputs = processor(images=image, return_tensors="pt")

# Run the model on the image
outputs = model(**inputs)

# Get the logits (proxy for probability)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
probs = logits.softmax(dim=1)

class_label = model.config.id2label[predicted_class_idx]
probability = probs[0][predicted_class_idx].item()

# Print the predicted class
print("Predicted class:", class_label)
print("Predicted probability:", probability)


