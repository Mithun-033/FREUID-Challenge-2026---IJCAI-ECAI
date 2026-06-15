from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms


train_transform=transforms.Compose([
    ...,
    transforms.ToTensor()
])

val_transform=transforms.Compose([
    ...,
    transforms.ToTensor()
])


