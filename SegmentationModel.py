import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
from torchvision.datasets import FashionMNIST
import numpy as np
import pandas as pd
import argparse
from models import *
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import nibabel as nib

root = '/home/mist/Data/covid19/'

def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, image, mask, transform=None, target_transform=None, loader=default_loader):
        self.image = torch.from_numpy(nib.load(image).get_fdata())
        self.image = self.image.reshape(100, 1, 512, 512).float()
        self.image = self.image.to(torch.float)
        self.mask = torch.from_numpy(nib.load(mask).get_fdata())
        self.mask = self.mask.reshape(100, 1, 512, 512).float()
        self.mask = self.mask.to(torch.float)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        image = self.image[index, :, :, :]
        mask = self.mask[index, :, :, :]
        return image, mask

    def __len__(self):
        return self.image.size(0)


def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=5, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=4, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=4, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()
    torch.cuda.empty_cache()


class Solver(object):
    def __init__(self, config):
        self.model = resnet18()
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        train_transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2]),
        ])

        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2]),
        ])

        self.train_loader = torch.utils.data.DataLoader(
            MyDataset(image=root + 'tr_im.nii', mask=root + 'tr_mask.nii'),
            batch_size=self.train_batch_size,
            shuffle=True
        )

        self.test_loader = torch.utils.data.DataLoader(
            MyDataset(image=root + 'tr_im.nii', mask=root + 'tr_mask.nii'),
            batch_size=self.test_batch_size,
            shuffle=True
        )
        pass

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = UNet(1, 1).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5,
                                                              verbose=True)

        self.criterion = nn.MSELoss().to(self.device)

    def train(self):
        print("train:")
        print(f"Model:{type(self.model)}")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            binoutput = (output > 0.5).int()
            # for crt_batch in range(self.train_batch_size):
            #     for crt_line in range(output.size(1)):
            #         loss = self.criterion(output[crt_batch, crt_line, :], target[crt_batch, crt_line, :])
            #         loss.backward(retain_graph=True)
            #         self.optimizer.step()
            #         train_loss += loss.item()

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0) * output.size(2) * output.size(3)

            # train_correct incremented by one if predicted right
            # train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy
            train_correct += (binoutput == target.int()).sum().item()

            print(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data.float())
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0) * output.size(2) * output.size(3)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                print(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):

        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            self.scheduler.step(epoch)
            print(f"\n===> epoch: {epoch}/{self.epochs}")
            train_result = self.train()
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()


if __name__ == '__main__':
    main()

