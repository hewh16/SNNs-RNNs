# -*- coding: utf-8 -*-

"""
@author: Weihua He
@email: hewh16@gmail.com
"""

# Please install dcll pkgs from below
# https://github.com/nmi-lab/dcll
# and then enjoy yourself.
# If there is any question please mail me.

from dcll.pytorch_libdcll import *
from dcll.experiment_tools import *
from dcll.load_dvsgestures_sparse import *
from tqdm import tqdm

import argparse, pickle, torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# For how many ms do we present a sample during classification
n_iters = 60
n_iters_test = 60 

# How epochs to run before testing 
# Here the epoch means batch, 1 epoch has 1176 samples, which is 49 batches
batch_size = 24
n_test_interval = batch_size

dt = 10000 #us
ds = 4
target_size = 11 # num_classes
n_epochs = 80 # 100
in_channels = 2 # Green and Red
thresh = 0.3
lens = 0.5
decay = 0.3
learning_rate = 1e-4
time_window = 60
im_dims = im_width, im_height = (128//ds, 128//ds)
names = 'dvsGesture_stbp_cnn_10_dropout'
parser = argparse.ArgumentParser(description='STDP for DVS gestures')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load data
gen_train, _ = create_data(
        batch_size = batch_size,
        chunk_size = n_iters,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)


_, gen_test = create_data(
        batch_size = batch_size,
        chunk_size = n_iters_test,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)

def generate_test(gen_test, n_test:int, offset=0):
    input_test, labels_test = gen_test.next(offset=offset)
    input_tests = []
    labels1h_tests = []
    n_test = min(n_test,int(np.ceil(input_test.shape[0]/batch_size)))
    for i in range(n_test):
        input_tests.append( torch.Tensor(input_test.swapaxes(0,1))[:,i*batch_size:(i+1)*batch_size].reshape(n_iters_test,-1,in_channels,im_width,im_height))
        labels1h_tests.append(torch.Tensor(labels_test[:,i*batch_size:(i+1)*batch_size]))
    return n_test, input_tests, labels1h_tests

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

cfg_cnn = [(2, 64, 1),
           (64, 128, 1),
           (128, 128, 1),
           ]

cfg_fc = [256, 11]


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


best_acc = 0
acc = 0
acc_record = list([])
Trainacc=0
Trainacc_record = list([])
class SNN_Model(nn.Module):

    def __init__(self, num_classes=10):
        super(SNN_Model, self).__init__()
        self.conv0 = nn.Conv2d(2, cfg_cnn[0][1], kernel_size=3, stride=1, padding=1)
        in_planes, out_planes, stride = cfg_cnn[1]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

        in_planes, out_planes, stride = cfg_cnn[2]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(8 * 8 * cfg_cnn[1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, input, win=15):
        c0_mem = c0_spike = torch.zeros(batch_size, cfg_cnn[0][1], 32, 32, device=device)
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[1][1], 32, 32, device=device)
        p1_mem = p1_spike = torch.zeros(batch_size, cfg_cnn[1][1], 16, 16, device=device)

        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[2][1], 16, 16, device=device)
        p2_mem = p2_spike = torch.zeros(batch_size, cfg_cnn[2][1], 8, 8, device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        #print(input.shape)
        for step in range(win):
            x = input[:, :, :, :, step].to(device)
            c0_mem, c0_spike = mem_update(self.conv0, x, c0_mem, c0_spike)
            c0_spike = F.dropout(c0_spike,p=0.5)
            torch.cuda.empty_cache()
            c1_mem, c1_spike = mem_update(self.conv1, c0_spike, c1_mem, c1_spike)
            p1_mem, p1_spike = mem_update_pool(F.avg_pool2d, c1_spike, p1_mem, p1_spike)
            p1_spike = F.dropout(p1_spike,p=0.5)
            c2_mem, c2_spike = mem_update(self.conv2, p1_spike, c2_mem, c2_spike)
            p2_mem, p2_spike = mem_update_pool(F.avg_pool2d, c2_spike, p2_mem, p2_spike)
            
            x = p2_spike.view(batch_size, -1)
            x = F.dropout(x,p=0.5)
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h1_spike = F.dropout(h1_spike,p=0.5)
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike
            del x
            torch.cuda.empty_cache()
        outputs = h2_sumspike / time_window
        #print(torch.mean(outputs,dim=0))
        return outputs

def mem_update(fc, x, mem, spike):
    torch.cuda.empty_cache()
    mem = mem * decay * (1 - spike) + fc(x)
    spike = act_fun(mem)
    return mem, spike

def mem_update_pool(opts, x, mem, spike, pool = 2):
    torch.cuda.empty_cache()
    mem = mem * decay * (1 - spike) + opts(x, pool)
    spike = act_fun(mem)
    return mem, spike

snn = nn.DataParallel(SNN_Model())
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(snn))
act_fun = ActFun.apply
print('Generating test...')
n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=100, offset = 0)
print('n_test %d' % (n_test))

input_extras = []
labels1h_extra = []

for i in range(8):
    input_extra, labels_extra = gen_train.next()
    n_extra = min(100,int(np.ceil(input_extra.shape[0]/batch_size)))
    for i in range(n_extra):
        input_extras.append( torch.Tensor(input_extra.swapaxes(0,1))[:,i*batch_size:(i+1)*batch_size].reshape(n_iters,-1,in_channels,im_width,im_height))
        labels1h_extra.append(torch.Tensor(labels_extra[:,i*batch_size:(i+1)*batch_size]))


for batch in range(n_epochs * batch_size):
    snn.train()
    snn.zero_grad()
    optimizer.zero_grad()
    
    running_loss = 0
    start_time = time.time()

    input, labels = gen_train.next()
    input = torch.Tensor(input.swapaxes(0,1)).reshape(n_iters,batch_size,in_channels,im_width,im_height)
    input = input.float()
    input = input.permute([1,2,3,4,0])
    labels = torch.from_numpy(labels).float()
    labels = labels[1, :, :]

    outputs = snn(input, time_window)

    loss = criterion(outputs.cpu(), labels)
    running_loss = running_loss + loss.item()
    loss.backward()
    optimizer.step()
    print('Batch [%d/%d], Loss:%.5f' % (batch + 1, n_epochs * batch_size, running_loss))

    if (batch + 1) % (1176//batch_size) == 0:
        snn.eval()
        correct,correctTrain = 0,0
        total,totalTrain = 0,0
        optimizer = lr_scheduler(optimizer, batch, learning_rate, 1000)
        for i in range(1176//batch_size):
            torch.cuda.empty_cache()
            inputs, labels = gen_train.next()
            inputs = torch.Tensor(inputs.swapaxes(0,1)).reshape(n_iters,batch_size,in_channels,im_width,im_height)
            inputs = inputs.float().permute([1,2,3,4,0])
            outputs = snn(inputs,time_window)
            _, predicted = torch.max(outputs.data, 1)
            _, labelTestTmp = torch.max(torch.Tensor(labels), 2)
            labelTest, _ = torch.max(labelTestTmp.data, 0)
            totalTrain = totalTrain + labelTest.size(0)
            correctTrain = correctTrain + (predicted.cpu() == labelTest).sum()
            del inputs, outputs, predicted, labelTestTmp, labelTest
        print('Train Accuracy of the model on the 288 train images: %.3f' % (100 * correctTrain.float() / totalTrain))
        print(totalTrain)
        print('total' )
        Trainacc = 100. * float(correctTrain) / float(totalTrain)
        Trainacc_record.append(Trainacc)
        for i in range(len(input_tests)):
            torch.cuda.empty_cache()
            inputTest = input_tests[i].float()
            inputTest = inputTest.permute([1,2,3,4,0])
            outputs = snn(inputTest, time_window)
            _, predicted = torch.max(outputs.data, 1)
            _, labelTestTmp = torch.max(labels1h_tests[i].data, 2)
            labelTest, _ = torch.max(labelTestTmp.data, 0)
            total = total + labelTest.size(0)
            correct = correct + (predicted.cpu() == labelTest).sum()
            del inputTest, outputs, predicted, labelTestTmp, labelTest

	 #######
	

        print('Test Accuracy of the model on the 288 test images: %.3f' % (100 * correct.float() / total))
        print(total)
        print('total' )
        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)

        print(acc)
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'batch': batch,
            'acc_record': acc_record,
        }
        #if not os.path.isdir('/home/student1/Documents/SCNN/dcll-master'):
            #os.mkdir('/home/student1/Documents/SCNN/dcll-master')
        torch.save(state, './' + names + '.t7')
        best_acc = acc

    
