import torch
import torch.nn as nn

class PercepNet(nn.Module):
    def __init__(self, input_dim=70):
        super(PercepNet, self).__init__()
        #self.hidden_dim = hidden_dim
        #self.n_layers = n_layers
        
        self.fc = nn.Linear(input_dim, 128)
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=5,padding=2)
        self.conv2 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding=1)
        self.gru = nn.GRU(input_dim, 512, 3, batch_first=True)
        self.gru_gb = nn.GRU(128, 128, 1, batch_first=True)
        self.gru_rb = nn.GRU(128, 128, 1, batch_first=True)
        self.fc_gb = nn.Linear(512, 34)
        self.fc_rb = nn.Linear(128,34)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(1)  
        x = self.conv1(x)
        convout = self.conv2(x)

        gruout, grustate = self.gru(convout)
        
        rnn_rb_out = self.gru(x[3:],grustate[3:])
        gb = self.fc_rb(self.rnn_gb_out)

        rnn_rb_out = self.gru(x[3:],grustate[3:])
        rb = self.fc_rb(self.rnn_rb_out)
        return gb,rb

def test():
    model = PercepNet()
    x = torch.rand([4,70])
    out = model(x)
    print(out.shape)

test()