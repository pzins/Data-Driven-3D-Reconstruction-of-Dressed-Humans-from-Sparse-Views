import torch
from torch.utils.data import DataLoader
from utils.utils import *
from data.TrainDataset import *
from model.MVPIFuNet import *
import pytorch_lightning as pl



class MultiViewPIFu(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.netG = MVPIFuNet(self.cfg, cfg["exp"]["projection"])
    

    def prepare_data(self):
        train_dataset = TrainDataset(self.cfg, phase='train')
        test_dataset = TrainDataset(self.cfg, phase='test')
        
        self.train_loader = DataLoader(train_dataset,
                                    batch_size=self.cfg["training"]["batch_size"], shuffle=not self.cfg["training"]["serial_batches"],
                                    num_workers=self.cfg["training"]["num_threads"], pin_memory=self.cfg["training"]["pin_memory"])
        
        sampler = torch.utils.data.SubsetRandomSampler(range(len(test_dataset)))
        self.test_loader = DataLoader(test_dataset,
                                    batch_size=1, shuffle=False, sampler=sampler,
                                    num_workers=self.cfg["training"]["num_threads"], pin_memory=self.cfg["training"]["pin_memory"])

        self.test_tr_loader = DataLoader(train_dataset, sampler=sampler,
                                    batch_size=1, shuffle=False,
                                    num_workers=self.cfg["training"]["num_threads"], pin_memory=self.cfg["training"]["pin_memory"])
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return [self.test_tr_loader, self.test_loader]

    def forward(self, data):
        print("Inference is not done here")
        exit(0)

    def training_step(self, batch, batch_idx):
        image_tensor = batch['img']
        calibs_tensor = batch['calibs']
        sample_tensor = batch['samples']
        occ_tensor = batch['occ']

        res, error = self.netG.forward(image_tensor, sample_tensor, calibs_tensor, occ=occ_tensor)
        error = error.mean()
        self.log('my_loss', error, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return error


    def validation_step(self, batch, batch_idx, dataloader_idx):
        image_tensor = batch['img']
        calibs_tensor = batch['calibs']
        sample_tensor = batch['samples']
        occ_tensor = batch['occ']
        
        res, error = self.netG.forward(image_tensor, sample_tensor, calibs_tensor, occ=occ_tensor)
        error = error.mean()
        IOU, prec, recall = compute_acc(res, occ_tensor, self.cfg["exp"]["grid_res"])
        self.log_dict({'my_loss_val': error, 'IoU': IOU, 'Precision': prec, 'Recall': recall})
        return error

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.netG.parameters(), lr=self.cfg["training"]["learning_rate"], momentum=0, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
        return [optimizer], scheduler