import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

class HPLCTracesDatasetV2(Dataset):
    
    def __init__(self,df_pool_path,df_ids_path,MAX_SEQ_LEN,SKIP_FIRST_N):
        def pad_x_list_and_skip_nfirst(self,x):
            x = x[self.skip_first_n:]
            if len(x) >= self.max_seq_len:
                x = x[0:self.max_seq_len]
        
            if len(x) < self.max_seq_len:
                x=[0.0]*(self.max_seq_len - len(x))+x
            return np.array(x,dtype=float)
    
        self.max_seq_len = MAX_SEQ_LEN
        self.skip_first_n=SKIP_FIRST_N
        df_pool = pd.read_parquet(df_pool_path)
        df_pool["value"] = df_pool["value"].apply(lambda x: pad_x_list_and_skip_nfirst(self,x))
        df_ids  = pd.read_csv(df_ids_path)
        _ids = df_ids["ID"].values
        X = df_pool.loc[_ids]['value'].tolist()
        X = np.array(X,dtype=float)
        X = X.reshape(-1,1,self.max_seq_len)
        y = df_ids["Bubble"].values
        y = y.reshape(-1,1)
        X,y = torch.tensor(X).float(),torch.tensor(y).float()
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X,y

class HPLCTracesDatamoduleV2(pl.LightningDataModule):
    def __init__(self,max_seq_len,skip_first_n,batch_size,df_pool_path,df_ids_path_train,df_ids_path_test,df_ids_path_val):
        super().__init__()
        self.df_pool_path = df_pool_path
        self.df_ids_path_train= df_ids_path_train
        self.df_ids_path_test = df_ids_path_test
        self.df_ids_path_val  = df_ids_path_val
        self.max_seq_len      = max_seq_len
        self.skip_first_n     = skip_first_n

        self.batch_size       = batch_size
        self.num_workers      = 10
        self.save_hyperparameters()
    def setup(self,stage=None):
        self.ds_train = HPLCTracesDatasetV2(df_pool_path=self.df_pool_path,
                                          df_ids_path=self.df_ids_path_train,
                                          MAX_SEQ_LEN = self.max_seq_len,
                                          SKIP_FIRST_N= self.skip_first_n)
        
        self.ds_test = HPLCTracesDatasetV2(df_pool_path=self.df_pool_path,
                                          df_ids_path=self.df_ids_path_test,
                                          MAX_SEQ_LEN = self.max_seq_len,
                                          SKIP_FIRST_N= self.skip_first_n)
        
        self.ds_val  = HPLCTracesDatasetV2(df_pool_path=self.df_pool_path,
                                          df_ids_path=self.df_ids_path_val,
                                          MAX_SEQ_LEN = self.max_seq_len,
                                          SKIP_FIRST_N= self.skip_first_n)
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size,shuffle = True, num_workers = self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.ds_test,  batch_size=self.batch_size,shuffle = False, num_workers = self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.ds_val,   batch_size=self.batch_size,shuffle = False, num_workers = self.num_workers)