import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class h5Dataset(Dataset):

    def __init__(self, h5_filename="training.h5", window_size=2000):
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
        x_train = np.reshape(x_train, (self.nb_sequences, self.window_size, self.x_dim))

        y_train = np.copy(all_data[:self.nb_sequences*self.window_size, self.y_dim])
        y_train = np.reshape(y_train, (self.nb_sequences, self.window_size, self.y_dim))

    def __len__(self):
        return self.nb_sequences

    def __getitem__(self, index):
        return (x_train[index], y_train[index])

class PercepNet(nn.Module):
    def __init__(self, input_dim=70):
        super(PercepNet, self).__init__()
        #self.hidden_dim = hidden_dim
        #self.n_layers = n_layers
        
        self.fc = nn.Linear(input_dim, 128)
        self.conv1 = nn.Conv1d(128, 512, 5, stride=1)
        self.conv2 = nn.Conv1d(512, 512, 3, stride=1)
        self.gru = nn.GRU(512, 512, 3, batch_first=True)
        self.gru_gb = nn.GRU(512, 512, 1, batch_first=True)
        self.gru_rb = nn.GRU(512, 128, 1, batch_first=True)
        self.fc_gb = nn.Sequential(nn.Linear(512, 34), nn.Sigmoid())
        self.fc_rb = nn.Sequential(nn.Linear(128, 34), nn.Sigmoid())
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = x.permute([0,2,1]) # B, D, T
        x = self.conv1(x)
        convout = self.conv2(x)
        convout = convout.permute([0,2,1]) # B, T, D
        gruout, grustate = self.gru(convout)
        
        rnn_gb_out = self.gru_gb(gruout,grustate[3:])
        gb = self.fc_gb(self.rnn_gb_out)

        rnn_rb_out = self.gru_rb(gruout,grustate[3:])
        rb = self.fc_rb(self.rnn_rb_out)
        return gb,rb

def test():
    model = PercepNet()
    x = torch.randn(20, 8, 70)
    out = model(x)
    print(out.shape)

def train():
    dataset = h5Dataset("training.h5")
    trainset_ratio = 0.8 # 1 - validation set ration
    train_size = int(trainset_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    num_epochs = 2
    for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
test()