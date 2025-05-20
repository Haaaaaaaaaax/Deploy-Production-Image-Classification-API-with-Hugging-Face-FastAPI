# Food101 Image Classification API

This repository contains a production-ready image classification API built with FastAPI and Hugging Face models. The API can classify food images using three different state-of-the-art models.

## Models Used

1. **Facebook/DINOv2** - A self-supervised vision transformer model
2. **Google/ViT** - Vision Transformer model
3. **Google/EfficientNetB3** - EfficientNet model

## Model Performance

The models were evaluated on the Food101 test dataset with the following accuracy scores:

| Model | Test Accuracy |
|-------|---------------|
| Facebook/DINOv2 | 97.8% |
| Google/ViT | 96.5% |
| Google/EfficientNetB3 | 87.8% |

## Features

- Multiple model endpoints for image classification
- FastAPI-based REST API
- Production-ready deployment setup
- Support for Food101 dataset classification

## API Endpoints

The API provides three different endpoints, one for each model:

- `/predict/dinov2` - DINOv2 model endpoint
- `/predict/vit` - ViT model endpoint
- `/predict/efficientnet` - EfficientNetB3 model endpoint

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Haaaaaaaaaax/Deploy-Production-Image-Classification-API-with-Hugging-Face-FastAPI.git
cd Deploy-Production-Image-Classification-API-with-Hugging-Face-FastAPI
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:
```bash
uvicorn main:app --reload
```

## Usage

You can make predictions by sending POST requests to any of the model endpoints with an image file. The API will return the classification results.

Example using curl:
```bash
curl -X POST "http://localhost:8000/predict/dinov2" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@your_image.jpg"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Repository

[GitHub Repository](https://github.com/Haaaaaaaaaax/Deploy-Production-Image-Classification-API-with-Hugging-Face-FastAPI.git)
