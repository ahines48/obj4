import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch import optim
import numpy as np
import struct

if __name__ == '__main__':

    def read_idx(filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="CNN to classify Y and N characters.")
    parser.add_argument("--debug", type=bool, default=False, help="Enable debug mode.")
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Paths to the dataset
    train_images_path = '/Users/ameliahines/PycharmProjects/pythonProject2/emnist-bymerge-train-images-idx3-ubyte'
    train_labels_path = '/Users/ameliahines/PycharmProjects/pythonProject2/emnist-bymerge-train-labels-idx1-ubyte'
    test_images_path = '/Users/ameliahines/PycharmProjects/pythonProject2/emnist-bymerge-test-images-idx3-ubyte'
    test_labels_path = '/Users/ameliahines/PycharmProjects/pythonProject2/emnist-bymerge-test-labels-idx1-ubyte'

    # Reading the dataset
    train_images = read_idx(train_images_path)
    train_labels = read_idx(train_labels_path)
    test_images = read_idx(test_images_path)
    test_labels = read_idx(test_labels_path)


    def preprocess_emnist_data(train_images, test_images):
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0

        train_images = train_images.reshape(train_images.shape[0], 28 * 28)
        test_images = test_images.reshape(test_images.shape[0], 28 * 28)

        return train_images, test_images

    preprocess_emnist_data(train_images, test_images)

    # Filter the dataset for 'Y' and 'N' (Assuming 'Y' is label 34 and 'N' is label 23)
    train_mask = np.isin(train_labels, [34, 23])
    test_mask = np.isin(test_labels, [34, 23])

    #train_images_filtered = train_images[train_mask]
    #train_labels_filtered = train_labels[train_mask]
    #test_images_filtered = test_images[test_mask]
    #test_labels_filtered = test_labels[test_mask]

    # Convert labels to 0 and 1
    #train_labels_filtered = np.where(train_labels_filtered == 34, 1, 0)
    #test_labels_filtered = np.where(test_labels_filtered == 34, 1, 0)

    # Convert to PyTorch tensors
    train_images_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.int64)
    test_images_tensor = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1) / 255
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.int64)

    # Creating DataLoader
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
    loaders = {
        'train': DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=1),
        'test': DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)
    }


    # CNN Model Definition (Modified for binary classification)
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, 5, 1, 2),  # Convolution layer
                nn.ReLU(),  # Activation layer
                nn.MaxPool2d(kernel_size=2)  # Pooling layer
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, 1, 2),  # Another convolution layer
                nn.ReLU(),  # Activation layer
                nn.MaxPool2d(2)  # Another pooling layer
            )
            self.out = nn.Linear(32 * 7 * 7, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            output = self.out(x)
            return output


    # Instantiate model, loss function, and optimizer
    cnn = CNN().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)


    # Training Function
    def train(num_epochs, cnn, loaders, device=device):
        cnn.train()
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(loaders['train']):
                images, labels = images.to(device), labels.to(device)
                output = cnn(images)
                loss = loss_func(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                                                             len(loaders['train']), loss.item()))


    # Testing Function
    def test(cnn, loaders, device):
        cnn.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loaders['test']:
                images, labels = images.to(device), labels.to(device)
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('Test Accuracy of the model on the test images: {:.2f}%'.format(accuracy))


    # Run Training and Testing
    train(10, cnn, loaders, device)
    test(cnn, loaders, device)

    # Optional Debug Visualizations
    if args.debug:
        figure = plt.figure(figsize=(10, 8))
        for i in range(1, 26):
            img, label = train_dataset[i - 1]
            img = img.squeeze()  # Remove channel dimension for plotting
            figure.add_subplot(5, 5, i)
            plt.title(chr(label + 65))
            plt.axis('off')
            plt.imshow(img, cmap='gray')
        plt.show()
