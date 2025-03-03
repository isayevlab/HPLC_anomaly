import numpy as np
import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

class PhilCNNModelV3(pl.LightningModule):
    def __init__(self,
                 max_seq_len,
                 n_feature_layers = 3,
                 n_feature_layers_och = [2,4,8],
                 n_feature_layers_ich = [1,2,4],
                 n_feature_layers_ksize = [5,3,3],
                 n_feature_layers_stride = [3,3,3],

                 n_agg_layers = 2, #1
                 n_agg_layers_och = [64,32,],**kwargs):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.n_feature_layers = n_feature_layers
        self.n_feature_layers_och = n_feature_layers_och
        self.n_feature_layers_ich = n_feature_layers_ich
        self.n_feature_layers_ksize = n_feature_layers_ksize
        self.n_feature_layers_stride = n_feature_layers_stride

        self.n_agg_layers = n_agg_layers
        self.n_agg_layers_och = n_agg_layers_och
        self.lr = 1e-5

        self.__dict__.update(kwargs)
        
        self.criterion = nn.BCELoss()
        metrics = MetricCollection([BinaryF1Score(),BinaryAccuracy(),BinaryPrecision(),BinaryRecall()])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics  = metrics.clone(prefix='test_')
        in_size = self.max_seq_len
        feature_layers = []
        for i in range(self.n_feature_layers):
            feature_layers.append(nn.Conv1d(in_channels=self.n_feature_layers_ich[i],
                                            out_channels=self.n_feature_layers_och[i],
                                            kernel_size=self.n_feature_layers_ksize[i],
                                            stride=self.n_feature_layers_stride[i]))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.BatchNorm1d(self.n_feature_layers_och[i]))
        
        feature_layers.append(nn.Flatten())
        _testinput = torch.autograd.Variable(torch.rand(1, 1, in_size))
        feature_out_dim = nn.Sequential(*feature_layers)(_testinput).data.view(1, -1).size(1)
        
        agg_layers=[]
        for i in range(self.n_agg_layers):
            agg_layers.append(nn.Linear(in_features=feature_out_dim,out_features=self.n_agg_layers_och[i]))
            agg_layers.append(nn.ReLU())
            agg_layers.append(nn.Dropout1d(0.1))
            feature_out_dim = self.n_agg_layers_och[i]
        agg_layers.append(nn.Linear(self.n_agg_layers_och[-1],1))
        agg_layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*feature_layers,*agg_layers)
        
        self.example_input_array = torch.Tensor(10, 1, in_size)
        self.save_hyperparameters()
        
    def forward(self,x):
        return self.layers(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        
        output = self.train_metrics(y_hat, y)
        self.log_dict(output)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        output = self.valid_metrics(y_hat, y)
        self.log_dict(output)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        output = self.test_metrics(y_hat, y)
        self.log_dict(output)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_init)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=20)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'val_BinaryF1Score' }
        