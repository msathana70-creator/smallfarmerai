import torch
from torchvision import models, transforms, datasets
from PIL import Image

data_dir = "dataset/color"

dataset = datasets.ImageFolder(data_dir)
classes = dataset.classes

model = models.mobilenet_v2(weights=None)

model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(torch.load("plant_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_disease(image):

    img = Image.open(image).convert("RGB")

    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    confidence, predicted = torch.max(probabilities,0)

    disease = classes[predicted.item()]
    confidence = round(confidence.item()*100,2)

    return disease, confidence