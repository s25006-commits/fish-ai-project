import torch
from torchvision import transforms, models
from PIL import Image
import os
import torch.nn.functional as F
from fish_info import get_fish_info


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

classes = sorted(os.listdir("dataset"))

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)

    label = classes[predicted.item()]
    info = get_fish_info(label)

    return label, conf.item(), info




