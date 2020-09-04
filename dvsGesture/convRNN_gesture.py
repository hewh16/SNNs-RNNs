from dcll.pytorch_libdcll import *
from dcll.experiment_tools import *
from dcll.load_dvsgestures_sparse import *
from tqdm import tqdm

import argparse, pickle, torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from convRNN import ConvRNNCell

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# For how many ms do we present a sample during classification
n_iters = 60
n_iters_test = 60

# How epochs to run before testing
n_test_interval = 20

batch_size = 10
dt = 50000 #us
ds = 4
target_size = 11 # num_classes
n_epochs = 3500 # 4500
in_channels = 2 # Green and Red
thresh = 0.3
lens = 0.25
decay = 0.3
learning_rate = 1e-4
time_window = 60
im_dims = im_width, im_height = (128//ds, 128//ds)
names = 'dvsGesture_RNN_origcnn_50'

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

cfg_conv = [[in_channels,64],[64,128],[128,128]]
cfg_kernel = [(3,3),(3,3),(3,3)]
cfg_pool = [[2,2],[2,2]]
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

class ConvRNN(nn.Module):

    def __init__(self, input_size, conv_dim, pool_size, kernel_size, fc_size,
                 batch_first=False, bias=True):
        super(ConvRNN, self).__init__()
        # conv_dim: dims of all these three conv layers, in a shape of a 3x2 list
        # pool_size: sizes of all these pool layers, in a shape of a 2x2 list
        # kernel_size: sizes of kernels of conv layers, in a shape of a 3x2 list
        # fc_size: sizes of the ouput side of fc layers, 1x2 list
        self.height, self.width = input_size
        self.conv_dim = conv_dim
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.fc_size = fc_size
        self.batch_first = batch_first
        self.bias = bias
        self.conv1 = ConvRNNCell(input_size=(self.height, self.width),
                                  input_dim=conv_dim[0][0],
                                  hidden_dim=conv_dim[0][1],
                                  kernel_size=self.kernel_size[0],
                                  bias=self.bias)
        self.conv2 = ConvRNNCell(input_size=(self.height, self.width),
                                  input_dim=conv_dim[1][0],
                                  hidden_dim=conv_dim[1][1],
                                  kernel_size=self.kernel_size[1],
                                  bias=self.bias)
        self.pool1 = nn.AvgPool2d(self.pool_size[0][0],self.pool_size[0][1])
        conv3Width = self.width//self.pool_size[0][0]
        conv3Height = self.height//self.pool_size[0][0]
        self.conv3 = ConvRNNCell(input_size=(conv3Height, conv3Width),
                                  input_dim=conv_dim[2][0],
                                  hidden_dim=conv_dim[2][1],
                                  kernel_size=self.kernel_size[2],
                                  bias=self.bias)
        self.pool2 = nn.AvgPool2d(self.pool_size[0][0], self.pool_size[0][1])
        finalWidth = conv3Width//self.pool_size[1][0]
        finalHeight = conv3Width//self.pool_size[1][0]
        self.fc1 = nn.Linear(finalHeight*finalWidth*self.conv_dim[2][1],self.fc_size[0])
        self.fc2 = nn.Linear(self.fc_size[0],self.fc_size[1])

    def forward(self, input_tensor):
        if not self.batch_first: #batch first鍙傛暟锛歜atch鐨勫ぇ灏忔槸绗�涓€涓�缁村害
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        batch_size = input_tensor.size(0)
        seq_len = input_tensor.size(1)
        # convolution layer1
        h, c = self.conv1.init_hidden(batch_size)
        output_inner = []
        cur_layer_input = input_tensor
        for t in range(seq_len):
            h, c = self.conv1(input_tensor=cur_layer_input[:,t,:,:,:],
                              cur_state=[h, c])
            output_inner.append(h)

        layer_output = torch.stack(output_inner,dim=1)

        # convolution layer 2
        h, c = self.conv2.init_hidden(batch_size)
        output_inner = []
        cur_layer_input = layer_output
        for t in range(seq_len):
            h, c = self.conv2(input_tensor=cur_layer_input[:, t, :, :, :],
                              cur_state=[h, c])
            #if t%10==0:
            	#print(h.size(0))
            output_inner.append(self.pool1(h))

        layer_output = torch.stack(output_inner, dim=1)

        # pool layer 1
        # layer_output = self.pool1(layer_output)

        # convolution layer 3
        h, c = self.conv3.init_hidden(batch_size)
        output_inner = []
        cur_layer_input = layer_output
        for t in range(seq_len):
            h, c = self.conv3(input_tensor=cur_layer_input[:, t, :, :, :],
                              cur_state=[h, c])
            #if t%10==0:
            	#print(h.size())
            output_inner.append(self.pool2(h))
        #print('final output size:',output_inner.size())
        layer_output = output_inner[-1]
        #print('hidden shape: ',layer_output.size())
        # pool layer 2: extract the last layer
        # layer_output = self.pool2(layer_output)[:,-1,:,:,:]

        # Linear layers

        cur_layer_input = layer_output.view(batch_size,-1)
        #print('reshape result: ',cur_layer_input.size())
        layer_output = self.fc1(cur_layer_input)
        outputs = self.fc2(layer_output)
        return outputs

conv_rnn_model = ConvRNN(input_size=im_dims,
                          conv_dim=cfg_conv,
                          pool_size=cfg_pool,
                          kernel_size=cfg_kernel,
                          fc_size=cfg_fc,
                          batch_first=True,
                          bias=True)
conv_rnn_model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(conv_rnn_model.parameters(),lr=learning_rate)
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(conv_rnn_model))
act_fun = ActFun.apply
print('Generating test...')
n_test, input_tests, labels1h_tests = generate_test(gen_test, n_test=100, offset = 0)
print('n_test %d' % (n_test))

for epoch in range(n_epochs):
    conv_rnn_model.zero_grad()
    optimizer.zero_grad()

    running_loss = 0
    start_time = time.time()

    input, labels = gen_train.next()
    input = torch.Tensor(input.swapaxes(0,1)).reshape(n_iters,batch_size,in_channels,im_width,im_height)
    input = input.float().to(device)
    input = input.permute([1,0,2,3,4])
    #print(input.size())
    labels = torch.from_numpy(labels).float()
    labels = labels[1, :, :]
    #print(labels.size())
    outputs = conv_rnn_model(input)

    loss = criterion(outputs.cpu(), labels)
    running_loss = running_loss + loss.item()
    loss.backward()
    for name, parms in conv_rnn_model.named_parameters():	
        print(name, parms.grad.shape)
        #print('-->name:', name, '-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)
    optimizer.step()
    print('Epoch [%d/%d], Loss:%.5f' % (epoch + 1, n_epochs, running_loss))

    if (epoch + 1) % n_test_interval == 0:
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 4000)

        for i in range(len(input_tests)):
            torch.cuda.empty_cache()
            conv_rnn_model.zero_grad()
            optimizer.zero_grad()
            inputTest = input_tests[i].float().to(device)
            #print('test input size: ',inputTest.size())
            inputTest = inputTest.permute([1,0,2,3,4])
            outputs = conv_rnn_model(inputTest)
            #print('output size: ',outputs.size())
            _, predicted = torch.max(outputs.data, 1)
            _, labelTestTmp = torch.max(labels1h_tests[i].data, 2)
            labelTest, _ = torch.max(labelTestTmp.data, 0)
            total = total + labelTest.size(0)
            correct = correct + (predicted.cpu() == labelTest).sum()
            # inputTest, labelsTest = gen_train.next()
            # inputTest = inputTest.float().to(device)
            del inputTest, outputs, predicted, labelTestTmp, labelTest
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct.float() / total))

        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)

        print(acc)
        print('Saving..')
        state = {
            'net': conv_rnn_model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
        best_acc = acc
