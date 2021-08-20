# -*- coding: utf-8 -*-
"""
@author: yjwu
"""

from MyLargeDataset import *
import torch, time, os

import torch.nn as nn
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
thresh = 0.3
lens = 0.25
decay = 0.3
num_classes = 10
batch_size = 40
num_epochs = 100
learning_rate = 1e-4
time_window = 15
names = 'nmnist_snn_5ms_new'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_path = r'./Dataset/NMNIST_train_data_5ms.mat'
test_path = r'./Dataset/NMNIST_test_data_5ms.mat'
# load datasets
train_dataset = MyDataset(train_path, 'nmnist_h', time_window)

test_dataset = MyDataset(test_path, 'nmnist_r')

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False, drop_last=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, drop_last=True)



class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens)


cfg_fc = [512, 512, 10]


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


best_acc = 0
acc_record = list([])


class SNN_Model(nn.Module):

    def __init__(self, num_classes=10):
        super(SNN_Model, self).__init__()

        self.fc1 = nn.Linear(34*34*2, cfg_fc[0], bias = True)
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[-1], bias = True)
        self.lateral1 = nn.Linear(cfg_fc[0], cfg_fc[0], bias = False)

    def forward(self, input, win=15):

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[-1], device=device)
        for step in range(win):

            x = input[:, :, :, :, step].view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike = h1_sumspike + h1_spike

            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike = h2_sumspike + h2_spike

        outputs = h2_sumspike / time_window
        # print(torch.mean(outputs,dim=0))
        return outputs

def mem_update(fc, x, mem, spike):
  
    mem = mem * decay * (1 - spike) + fc(x)
    spike = act_fun(mem)
    return mem, spike

snn = SNN_Model()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

act_fun = ActFun.apply

for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        outputs = snn(images, time_window)
        loss = criterion(outputs.cpu(), labels)
        running_loss = running_loss + loss.item()
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            # print( outputs.sum(dim=0) )
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, running_loss))
            running_loss = 0
            print('Time elasped:', time.time() - start_time)
    correct = 0
    total = 0

    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
    for images, labels in test_loader:
        images = images.float().to(device)
        outputs = snn(images, time_window)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted.cpu() == labels).sum()
    print('Iters:', epoch, '\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct.float() / total))

    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if epoch % 3 == 0:
        print(acc)
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
        best_acc = acc
