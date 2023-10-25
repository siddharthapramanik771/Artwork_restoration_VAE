import os
from torchvision import transforms

def get_clear_image_path(folder,filename,type='distorted'):
  return os.path.join(folder.replace(type,'clear'), filename.split(')')[0]+').jpg')

# Define the dataloader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None,type='distorted'):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)
        self.type = type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        # print(img_path)
        if type == 'clear':
          clear_image_path = img_path
        else:
          clear_image_path = get_clear_image_path(self.root_dir, self.images[idx],
                                                self.type)
        # print(clear_image_path)
        image = Image.open(img_path).convert('RGB')
        clear_image = Image.open(clear_image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            clear_image = self.transform(clear_image)
        return image,clear_image