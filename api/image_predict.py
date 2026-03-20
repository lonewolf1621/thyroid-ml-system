from fastapi import UploadFile
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

MODEL_PATH = "image_models/cancer_efficientnet.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

async def predict_image(file: UploadFile):

    image = Image.open(file.file).convert("RGB")

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

    pred = torch.argmax(probs).item()
    confidence = float(probs[0][pred])

    label = "malignant" if pred == 1 else "benign"

    return {
        "prediction": label,
        "confidence": confidence
    }