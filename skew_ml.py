import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
    
    
from tqdm import tqdm
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkewDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.skew_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.skew_df)

    def __getitem__(self, idx):
        img_name = self.skew_df.iloc[idx, 0]
        angle = self.skew_df.iloc[idx, 1]
        img_path = f"{self.root_dir}/{img_name}"
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, angle
    
    
class ResNetRegression(nn.Module):
    def __init__(self, modelname='resnet18'):
        super(ResNetRegression, self).__init__()
        if modelname == 'resnet18':
            resnet = models.resnet18(weights= ResNet18_Weights.DEFAULT)
            self.features = nn.Sequential(*list(resnet.children())[:-1])  
            self.fc = nn.Linear(512, 1)  
            
        elif modelname == 'resnet50':
            resnet = models.resnet50(weights= ResNet50_Weights.DEFAULT)
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Linear(2048, 1)
        else:
            raise ValueError("modelname should be 'resnet18' or 'resnet50'")

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)
    
    
def train_model(model, criterion, optimizer, train_loader, valid_loader, start = 1, num_epochs=10, num_save = 5):
    tr_ls = []
    val_ls = []
    for epoch in range(start,num_epochs+1):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch}/{num_epochs}, ", end = " ")
        for images, angles in tqdm(train_loader, unit='it'):
            images = images.to(device)
            angles = angles.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, angles.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f" Loss: {epoch_loss:.4f}")
        tr_ls.append(epoch_loss)
    
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, angles in tqdm(valid_loader, unit='it'):
                images = images.to(device)
                angles = angles.to(device)
                outputs = model(images)
                loss = criterion(outputs, angles.float())
                valid_loss += loss.item() * images.size(0)
            valid_loss /= len(valid_loader.dataset)
            
        print(f"Val Loss: {valid_loss:.4f}")
        val_ls.append(valid_loss)
      

        
        if epoch % num_save or epoch == num_epochs:
            if not os.path.exists("weights"):
                os.makedirs("weights")
            checkpoint_path = f'weights/chkmodel{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                'tr_ls': tr_ls,
                'val_ls': val_ls    
            }, checkpoint_path)
    
        
    epochs = range(1, len(tr_ls) + 1)
    plt.plot(epochs, tr_ls, label='Training Loss')
    plt.plot(epochs, val_ls, label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True) 
    plt.show()
    
 
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


