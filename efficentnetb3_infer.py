from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

def predict(image_path):
    processor = AutoImageProcessor.from_pretrained("Haaaaaaaaaax/efficientnet-b3-finetuned-food101")
    model = AutoModelForImageClassification.from_pretrained("Haaaaaaaaaax/efficientnet-b3-finetuned-food101")
    # Load an image

    image = Image.open(image_path)

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt") # return pythorch tensor

    # Perform inference
    with torch.no_grad(): #stop training
        outputs = model(**inputs)

    # Get the predicted label
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()

    # Map the predicted class index to a label
    labels = model.config.id2label
    predicted_label = labels[predicted_class]

    return predicted_label, predictions[0][predicted_class]


if __name__ == "__main__":
    predicted_label, confidence = predict("test_data/test_image.jpg")
    print(f"Predicted label: {predicted_label}, Confidence: {confidence}")
