#!/usr/bin python3

# Copyright 2021 Seonghun Noh

import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
import argparse
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.utils.rnn as rnn_utils

from collections import defaultdict
import yaml
import glob
from tqdm import tqdm

plt.switch_backend('agg')
class h5DirDataset(Dataset):
    def __init__(self, h5_dir_path, train_length_size=500):
        self.train_length_size = train_length_size
        self.h5_dir_path = h5_dir_path
        self.x_dim = 70
        self.y_dim = 68

        h5_filelist = glob.glob(os.path.join(h5_dir_path, "*.h5"))
        self.nb_sequences = len(h5_filelist)
        all_data = []
        for filename in h5_filelist:
            with h5py.File(filename, 'r') as hf:
                all_data += [hf['data'][:]]

        all_data = np.vstack(tuple(all_data))
        print(self.nb_sequences, ' sequences')
        x_train = all_data[:self.nb_sequences*self.train_length_size, :self.x_dim]
        self.x_train = np.reshape(x_train, (self.nb_sequences, self.train_length_size, self.x_dim))

        y_train = np.copy(all_data[:self.nb_sequences*self.train_length_size, self.x_dim:self.x_dim+self.y_dim])
        self.y_train = np.reshape(y_train, (self.nb_sequences, self.train_length_size, self.y_dim))

    def __len__(self):
        return self.nb_sequences

    def __getitem__(self, index):
        return (self.x_train[index], self.y_train[index])

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

class Trainer(object):
    """Customized trainer module for PercepNet training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        criterion,
        optimizer,
        args,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.
        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (nn.Module): Model. Instance of nn.Module
            criterion (nn.Module): criterions.
            optimizer (torch.optim): optimizers.
            args (parser.parse_args()): Instance of argparse parse_args()
            device (torch.deive): Pytorch device instance.
        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.args = args
        self.optimizer = optimizer
        self.device = device
        self.config = config

        self.writer = SummaryWriter(config["out_dir"])
        self.finish_train = False


    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.args.train_max_steps, desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint."""
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.args.distributed:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
    
    def _train_step(self, batch):
        """Train model one step."""
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.model(inputs)
        #outputs = torch.cat(outputs,-1)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.args.rank == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["out_dir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            self.total_train_loss /= self.config["log_interval_steps"]
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
            )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True

def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train PercepNet (See detail in rnn_train.py)."
    )
    parser.add_argument(
        "--train_length_size",
        default=2000,
        type=int,
        help="RNN network train length size.",
    )
    parser.add_argument(
        "--train_max_steps",
        default=10000,
        type=int,
        help="max train steps.",
    )
    parser.add_argument(
        "--h5_train_dir",
        type=str,
        required=True,
        help="h5 train dataset directory.",
    )
    parser.add_argument(
        "--h5_dev_dir",
        type=str,
        required=True,
        help="h5 dev dataset directory.",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )

    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    with open(os.path.join(args.out_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    model = PercepNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = CustomLoss()

    train_dataset = h5DirDataset(
            args.h5_train_dir, train_length_size=args.train_length_size)
    dev_dataset = h5DirDataset(
            args.h5_dev_dir, train_length_size=args.train_length_size)

    logging.info(f"The number of training files = {len(train_dataset)}.")
    logging.info(f"The number of training files = {len(dev_dataset)}.")

    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    
    data_loader = {
        "train" : torch.utils.data.DataLoader(
            dataset["train"], 
            batch_size=config["batch_size"], 
            shuffle=True
        ),
        "dev": torch.utils.data.DataLoader(
            dataset["dev"], 
            batch_size=config["batch_size"], 
            shuffle=True
        )
    }
    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        args=args,
        sampler=sampler,
        device=device,
    )

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config["out_dir"], f"checkpoint-{trainer.steps}steps.pt")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")

if __name__ == '__main__':
    main()
    #train()
