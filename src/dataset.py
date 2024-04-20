from torch.utils.data import Dataset

class CustomFashionMNISTDataset(Dataset):
    def __init__(self, images, labels, image_shape, image_transform):
        self.images = images
        self.labels = labels
        self.image_shape = image_shape
        self.image_transform = image_transform


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        image = self.images[i].reshape(self.image_shape)
        label = self.labels[i]

        if self.image_transform:
            image = self.image_transform(image)
        
        return image, label