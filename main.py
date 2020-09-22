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
import torch.nn.functional as F

root = '/tmp/pycharm_project_828/mydata'
# root = 'C:/Users/13377/Desktop/mydata'

def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader, is_smooth=False):
        fh = open(txt, 'r')
        fl = open('labels.txt', 'r')
        imgs = []
        i = 0
        # for line in fh:
        #     line = line.strip('\n')
        #     line = line.rstrip()
        #     words = line.split()
        #     label = [int(x == int(words[1])) for x in range(10)]
        #     label = torch.tensor(label, dtype=torch.float)
        #     label = [int(i), label]
        #     i = i + 1

        if is_smooth:
            for line in fl:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                label = []
                for i in range(1, 11):
                    label.append(float(words[i]))
                    i = i + 1
                label = torch.tensor(label, dtype=torch.float)

                imgs.append(('/tmp/pycharm_project_828/mydata/train/' + words[0], label))
        else:
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                label = [int(x == int(words[1])) for x in range(10)]
                label = torch.tensor(label, dtype=torch.float)
                imgs.append((root + words[0], label))

        fh.close()
        fl.close()
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)


# smooth_weight = {}

def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=128, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=128, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    parser.add_argument('-m', '--modelName', default='downsample1', type=str, help='name of saved model')
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()
    torch.cuda.empty_cache()

    # of = open('labels.txt', 'w')
    # for key in sorted(smooth_weight):
    #     of.write(f'{key}.jpg ')
    #     for weights in smooth_weight[key]:
    #         of.write(str(format(weights, '.3f')))
    #         of.write(' ')
    #     of.write('\n')
    # of.close()

class Solver(object):
    def __init__(self, config):
        self.model = VGG11()
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.model_name = config.modelName
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        # train_transformer = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        #     transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2]),
        # ])
        #
        # test_transformer = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        #     transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2]),
        # ])

        train_transformer =  transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ])
        test_transformer = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                ])

        self.train_loader = torch.utils.data.DataLoader(
            MyDataset(root + '/new_train.txt', transform=train_transformer, is_smooth=True),
            # FashionMNIST('tmp/pycharm_project_828', train=True, transform=train_transformer),
            batch_size = self.train_batch_size,
            shuffle = True
            )

        self.test_loader = torch.utils.data.DataLoader(
            MyDataset(root + '/new_test.txt', transform=test_transformer),
            # FashionMNIST('tmp/pycharm_project_828', train=False, transform=test_transformer),
            batch_size = self.test_batch_size,
            shuffle = True
        )
        pass

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.model = VGG11().to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5,verbose=True)

        self.criterion = nn.BCELoss().to(self.device)

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            data = F.interpolate(data, scale_factor=0.5)

            output = self.model(data)

            # for i in range(output.size(0)):
            #     smooth_weight[ori_target[0][i].item()] = output[i].cpu().clone().detach().numpy().tolist()
            #     i = i + 1

            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            ans = torch.max(target, 1)
            total += target.size(0)

            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == ans[1].cpu().numpy())

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

                data = F.interpolate(data, scale_factor=0.5)

                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                ans = torch.max(target, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == ans[1].cpu().numpy())

                print(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = self.model_name + '.pth'
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
    
