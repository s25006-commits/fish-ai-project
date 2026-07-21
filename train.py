import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 📌 数据增强（重点！！）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),   # 小角度
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


dataset = datasets.ImageFolder(root="dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


num_classes = len(dataset.classes)
print("类别：", dataset.classes)


model = models.resnet18(pretrained=True)


model.fc = nn.Linear(model.fc.in_features, num_classes)


criterion = nn.CrossEntropyLoss()
# 冻结所有层
for param in model.parameters():
    param.requires_grad = False

# 只训练最后一层
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.fc.parameters(), lr=0.003)


epochs = 20
best_loss = 999
for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}")
    if running_loss < best_loss:
        best_loss = running_loss
        torch.save(model.state_dict(), "best_model.pth")

        print("トレーニング終わった！保存した")



