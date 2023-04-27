import torchvision
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
from config import *
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def prepare_imgs(imgs):

  prep = torchvision.transforms.Compose([
        # smaller sized imagaes give faster results
      torchvision.transforms.Grayscale(),
      torchvision.transforms.ToTensor(),  # Trnsfor to tensor
      torchvision.transforms.Normalize((0.5) ,(0.5)) # Normalise to get data in range
  ])
  
  transformed_img = prep(imgs)

  return transformed_img

dataset =torchvision.datasets.ImageFolder(dataset_path, transform=prepare_imgs)

train_set, test_set= train_test_split(dataset, test_size=0.05)
print('Training set size: ', len(train_set))
print('test set size: ', len(test_set))


train_loader = DataLoader(train_set, batch_size, shuffle= True, num_workers= num_work)
test_loader = DataLoader(test_set, batch_size, shuffle= True, num_workers= num_work)


# for images, labels in train_loader:

#   fig, ax =plt.subplots(figsize = (15,15))
#   ax.set_xticks([])
#   ax.set_yticks([])

#   ax.imshow(make_grid(images, 4).clip(0,1).permute(1,2,0))

#   break