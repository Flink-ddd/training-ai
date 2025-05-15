
from google.colab import drive
drive.mount('/content/drive')


!pip install -q torchvision torch matplotlib


import os, shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt


# 路径设置
origin_data_dir = "/content/drive/MyDrive/emotion_classified_masked_faces_7class"
base_dir = "/content/mobilenet_emotion"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")


# 清理旧目录
if os.path.exists(base_dir):
   shutil.rmtree(base_dir)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)


# 数据划分
for emotion in os.listdir(origin_data_dir):
   emotion_path = os.path.join(origin_data_dir, emotion)
   if not os.path.isdir(emotion_path): continue


   image_paths = [os.path.join(emotion_path, img) for img in os.listdir(emotion_path)
                  if img.lower().endswith((".jpg", ".jpeg", ".png"))]


   if len(image_paths) == 0:
       print(f"跳过空类别：{emotion}")
       continue


   train_imgs, val_imgs = train_test_split(image_paths, test_size=0.2, random_state=42)
   os.makedirs(os.path.join(train_dir, emotion), exist_ok=True)
   os.makedirs(os.path.join(val_dir, emotion), exist_ok=True)


   for img in train_imgs:
       shutil.copy(img, os.path.join(train_dir, emotion))
   for img in val_imgs:
       shutil.copy(img, os.path.join(val_dir, emotion))


print("数据集划分完成")


# 每类图片统计
print("\n每类训练图片数量：")
for emotion in os.listdir(train_dir):
   count = len(os.listdir(os.path.join(train_dir, emotion)))
   print(f"{emotion:<10}: {count}")


# 模型训练部分
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


batch_size = 32
num_epochs = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 图像增强和预处理
data_transforms = {
   'train': transforms.Compose([
       transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
       transforms.RandomHorizontalFlip(),
       transforms.ColorJitter(brightness=0.2, contrast=0.2),
       transforms.ToTensor(),
   ]),
   'val': transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
   ]),
}


# 加载数据
image_datasets = {
   'train': datasets.ImageFolder(train_dir, data_transforms['train']),
   'val': datasets.ImageFolder(val_dir, data_transforms['val']),
}
dataloaders = {
   'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
   'val': DataLoader(image_datasets['val'], batch_size=batch_size),
}
class_names = image_datasets['train'].classes
num_classes = len(class_names)


# 模型
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# 开始训练
best_val_acc = 0.0
train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []


print("\n开始训练 MobileNetV2...\n")
for epoch in range(num_epochs):
   print(f"Epoch {epoch+1}/{num_epochs}")
   for phase in ['train', 'val']:
       model.train() if phase == 'train' else model.eval()
       running_loss, running_corrects = 0.0, 0


       dataloader = dataloaders[phase]
       loop = tqdm(dataloader, desc=f"[{phase.upper()}]", leave=False)
       for inputs, labels in loop:
           inputs, labels = inputs.to(device), labels.to(device)
           optimizer.zero_grad()


           with torch.set_grad_enabled(phase == 'train'):
               outputs = model(inputs)
               loss = criterion(outputs, labels)
               preds = torch.argmax(outputs, dim=1)


               if phase == 'train':
                   loss.backward()
                   optimizer.step()


           running_loss += loss.item() * inputs.size(0)
           running_corrects += torch.sum(preds == labels)
           loop.set_postfix(loss=loss.item())


       epoch_loss = running_loss / len(image_datasets[phase])
       epoch_acc = running_corrects.double() / len(image_datasets[phase])
       print(f"{phase.capitalize():<5} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


       if phase == 'train':
           train_loss_list.append(epoch_loss)
           train_acc_list.append(epoch_acc.item())
       else:
           val_loss_list.append(epoch_loss)
           val_acc_list.append(epoch_acc.item())
           # 保存验证精度最好的模型
           if epoch_acc > best_val_acc:
               best_val_acc = epoch_acc
               best_model_path = "/content/drive/MyDrive/mobilenet_emotion_model.pth"
               torch.save(model.state_dict(), best_model_path)
               print(f"保存最佳模型，Val Acc: {epoch_acc:.4f}")


   print("-" * 50)


# 可视化训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


print(f"\n训练完成，最佳模型保存在：{best_model_path}")
