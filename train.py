import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from PIL import Image

import random


# ============================================================
# 1. 基本设置
# ============================================================

DATASET_DIR = "dataset"

BATCH_SIZE = 32
EPOCHS = 20

# 每张图片在训练过程中产生多少种随机变化
AUGMENT_TIMES = 15

TRAIN_RATIO = 0.8

random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. 数据增强
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        224,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        15
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# 验证集不要数据增强
val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# 3. 读取原始数据
# ============================================================

# 先读取，不进行增强
base_dataset = datasets.ImageFolder(
    DATASET_DIR
)

print("================================")
print("鱼种类别：")
print(base_dataset.classes)
print("================================")

num_classes = len(base_dataset.classes)

print(
    "原始图片数量：",
    len(base_dataset)
)


# ============================================================
# 4. 每个类别进行 80% / 20% 划分
# ============================================================

train_indices = []
val_indices = []


for class_id in range(num_classes):

    # 找出当前鱼种的所有图片
    class_indices = [
        i
        for i, label in enumerate(
            base_dataset.targets
        )
        if label == class_id
    ]

    random.shuffle(class_indices)

    train_size = int(
        len(class_indices) * TRAIN_RATIO
    )

    train_indices.extend(
        class_indices[:train_size]
    )

    val_indices.extend(
        class_indices[train_size:]
    )


print()
print("训练原图：", len(train_indices))
print("验证原图：", len(val_indices))


# ============================================================
# 5. 自定义训练 Dataset
#
# 每张原图每次读取时随机增强
# ============================================================

class AugmentedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset,
        indices,
        transform,
        augment_times=15
    ):

        self.dataset = dataset

        self.indices = indices

        self.transform = transform

        self.augment_times = augment_times


    def __len__(self):

        return (
            len(self.indices)
            * self.augment_times
        )


    def __getitem__(self, index):

        # 找到原始图片
        original_index = self.indices[
            index // self.augment_times
        ]

        # 获取图片路径
        image_path, label = (
            self.dataset.samples[
                original_index
            ]
        )

        # 打开图片
        image = Image.open(
            image_path
        ).convert("RGB")

        # 随机增强
        image = self.transform(
            image
        )

        return image, label


# ============================================================
# 6. 验证 Dataset
# ============================================================

class ValidationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset,
        indices,
        transform
    ):

        self.dataset = dataset

        self.indices = indices

        self.transform = transform


    def __len__(self):

        return len(self.indices)


    def __getitem__(self, index):

        real_index = self.indices[index]

        image_path, label = (
            self.dataset.samples[
                real_index
            ]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return image, label


# ============================================================
# 7. 创建 Dataset
# ============================================================

train_dataset = AugmentedDataset(

    base_dataset,

    train_indices,

    train_transform,

    AUGMENT_TIMES
)


val_dataset = ValidationDataset(

    base_dataset,

    val_indices,

    val_transform
)


print()
print("================================")
print("实际训练样本：")
print(len(train_dataset))

print("验证样本：")
print(len(val_dataset))
print("================================")


# ============================================================
# 8. DataLoader
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ============================================================
# 9. ResNet18
# ============================================================

model = models.resnet18(

    weights=models.ResNet18_Weights.DEFAULT
)


# ============================================================
# 10. 冻结前面的网络
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# 11. 解冻 layer4
# ============================================================

for param in model.layer4.parameters():

    param.requires_grad = True


# ============================================================
# 12. 修改最后分类层
# ============================================================

model.fc = nn.Linear(

    model.fc.in_features,

    num_classes
)


# ============================================================
# 13. Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 14. Optimizer
# ============================================================

optimizer = optim.Adam(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=0.0003
)


# ============================================================
# 15. Training
# ============================================================

best_val_loss = float("inf")


for epoch in range(EPOCHS):


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        _, predicted = torch.max(
            outputs,
            1
        )


        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_loader)
    )


    train_accuracy = (
        train_correct /
        train_total
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


            val_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss /
        len(val_loader)
    )


    val_accuracy = (
        val_correct /
        val_total
    )


    # --------------------------------------------------------
    # 显示训练结果
    # --------------------------------------------------------

    print(
        f"Epoch "
        f"{epoch + 1}/{EPOCHS} | "

        f"Train Loss: "
        f"{train_loss:.4f} | "

        f"Train Acc: "
        f"{train_accuracy:.2%} | "

        f"Val Loss: "
        f"{val_loss:.4f} | "

        f"Val Acc: "
        f"{val_accuracy:.2%}"
    )


    # --------------------------------------------------------
    # 保存最佳模型
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(

            model.state_dict(),

            "best_model.pth"
        )

        print(
            "✅ 最佳模型已保存！"
        )


# ============================================================
# 16. 完成
# ============================================================

print()
print("================================")
print("训练完成！")
print("================================")

print(
    "模型已保存：best_model.pth"
)


