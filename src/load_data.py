import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_path="../data", batch_size=32, img_size=150):

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_dir = os.path.join(data_path, "Training")
    test_dir = os.path.join(data_path, "Testing")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = train_dataset.classes  # ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']

    return train_loader, test_loader, class_names
