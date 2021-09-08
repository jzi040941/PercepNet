import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import argparse
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils

plt.switch_backend('agg')
class h5Dataset(Dataset):

    def __init__(self, h5_filename="training.h5", window_size=500):
        self.window_size = window_size
        self.h5_filename = h5_filename
        self.x_dim = 70
        self.y_dim = 68

        #read h5file
        with h5py.File(self.h5_filename, 'r') as hf:
            all_data = hf['data'][:]
        
        self.nb_sequences = len(all_data)//window_size
        print(self.nb_sequences, ' sequences')
        x_train = all_data[:self.nb_sequences*self.window_size, :self.x_dim]
        self.x_train = np.reshape(x_train, (self.nb_sequences, self.window_size, self.x_dim))
        #pad 3 for each batch .. not sure it's right
        #self.x_train = np.pad(self.x_train,[(0,0),(3,3),(0,0)],'constant')

        y_train = np.copy(all_data[:self.nb_sequences*self.window_size, self.x_dim:self.x_dim+self.y_dim])
        self.y_train = np.reshape(y_train, (self.nb_sequences, self.window_size, self.y_dim))

    def __len__(self):
        return self.nb_sequences

    def __getitem__(self, index):
        return (self.x_train[index], self.y_train[index])

class PercepNet(nn.Module):
    def __init__(self, input_dim=70):
        super(PercepNet, self).__init__()
        #self.hidden_dim = hidden_dim
        #self.n_layers = n_layers
        
        self.fc = nn.Sequential(nn.Linear(input_dim, 128), nn.Sigmoid())
        self.conv1 = nn.Sequential(nn.Conv1d(128, 512, 5, stride=1, padding=4), nn.Sigmoid())#padding for align with c++ dnn
        self.conv2 = nn.Sequential(nn.Conv1d(512, 512, 3, stride=1, padding=2), nn.Sigmoid())
        #self.gru = nn.GRU(512, 512, 3, batch_first=True)
        self.gru1 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru2 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru3 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru_gb = nn.GRU(512, 512, 1, batch_first=True)
        self.gru_rb = nn.GRU(1024, 128, 1, batch_first=True)
        self.fc_gb = nn.Sequential(nn.Linear(512*5, 34), nn.Sigmoid())
        self.fc_rb = nn.Sequential(nn.Linear(128, 34), nn.Sigmoid())

    def forward(self, x):
        x = self.fc(x)
        x = x.permute([0,2,1]) # B, D, T
        x = self.conv1(x)
        x = x[:,:,:-4]
        convout = self.conv2(x)
        convout = convout[:,:,:-2]#align with c++ dnn
        convout = convout.permute([0,2,1]) # B, T, D
       
        gru1_out, gru1_state = self.gru1(convout)
        gru2_out, gru2_state = self.gru2(gru1_out)
        gru3_out, gru3_state = self.gru3(gru2_out)
        gru_gb_out, gru_gb_state = self.gru_gb(gru3_out)
        concat_gb_layer = torch.cat((convout,gru1_out,gru2_out,gru3_out,gru_gb_out),-1)
        gb = self.fc_gb(concat_gb_layer)

        #concat rb need fix
        concat_rb_layer = torch.cat((gru3_out,convout),-1)
        rnn_rb_out, gru_rb_state = self.gru_rb(concat_rb_layer)
        rb = self.fc_rb(rnn_rb_out)

        output = torch.cat((gb,rb),-1)
        return output

def test():
    model = PercepNet()
    x = torch.randn(20, 8, 70)
    out = model(x)
    print(out.shape)

class CustomLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        gamma = 0.5
        C4 = 10
        epsi = 1e-10
        gb_hat = outputs[:,:,:34]
        rb_hat = outputs[:,:,34:68]
        gb = targets[:,:,:34]
        rb = targets[:,:,34:68]
        
        '''
        total_loss=0
        for i in range(500):
            total_loss += (torch.sum(torch.pow((torch.pow(gb[:,i,:],gamma) - torch.pow(gb_hat[:,i,:],gamma)),2))) \
             + C4*torch.sum(torch.pow(torch.pow(gb[:,i,:],gamma) - torch.pow(gb_hat[:,i,:],gamma),4)) \
             + torch.sum(torch.pow(torch.pow((1-rb[:,i,:]),gamma)-torch.pow((1-rb_hat[:,i,:]),gamma),2))
        return total_loss
        '''
        return (torch.mean(torch.pow((torch.pow(gb,gamma) - torch.pow(gb_hat,gamma)),2))) \
             + C4*torch.mean(torch.pow(torch.pow(gb,gamma) - torch.pow(gb_hat,gamma),4)) \
             + torch.mean(torch.pow(torch.pow((1-rb),gamma)-torch.pow((1-rb_hat),gamma),2))

    

def train():
    parser = argparse.ArgumentParser()
    writer = SummaryWriter()

    UseCustomLoss = True
    dataset = h5Dataset("training.h5")
    trainset_ratio = 1 # 1 - validation set ration
    train_size = int(trainset_ratio * len(dataset))
    test_size = len(dataset) - train_size
    batch_size=10
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = PercepNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    if UseCustomLoss:
        #CustomLoss cause Nan error need fix
        criterion = CustomLoss()
    else:
        criterion = nn.MSELoss()
    num_epochs = 10000
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            #outputs = torch.cat(outputs,-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # for testing
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, loss.item()))

            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
       
        model.eval()
        tmp_output = model(torch.tensor(dataset[0][0]).unsqueeze(0))
        model.train()
        fig = plt.figure()
        plt.plot(tmp_output[0].squeeze(0).T.detach().numpy())
        writer.add_figure('output gb', fig, global_step=epoch)
        fig = plt.figure()
        plt.plot(dataset[0][1][:,:].T)
        writer.add_figure('target gb', fig, global_step=epoch)
        writer.add_scalar('loss', loss.item(), global_step=epoch)
    print('Finished Training')
    print('save model')
    writer.close()
    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    train()
