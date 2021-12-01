import torch
from utils.options import BaseOptions
from model.MultiViewPIFu import MultiViewPIFu
from utils import config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import logging
logging.getLogger('pyembree').disabled = True

opt = BaseOptions().parse()

if __name__ == '__main__':

    if opt.load_checkpoint is not None:
        mvpifu = MultiViewPIFu.load_from_checkpoint(opt.load_checkpoint)
        cfg = mvpifu.cfg
        mvpifu.cfg = config.overwrite_options_resume_training(cfg, opt)
    if opt.resume_training is not None:
        mvpifu = MultiViewPIFu.load_from_checkpoint(opt.resume_training)
        cfg = mvpifu.cfg
        mvpifu.cfg = config.overwrite_options_resume_training(cfg, opt)
    else:
        cfg = config.load_config(opt.config)
        cfg = config.overwrite_options(cfg, opt)
        mvpifu = MultiViewPIFu(cfg)


    # mvpifu.prepare_data() # Automatically called by pl

    logger = TensorBoardLogger(cfg["exp"]["logs_path"], name=cfg["exp"]["name"])

    checkpoint_callback_iou = ModelCheckpoint(
        filename="{epoch}-{step}" + "-IoU_val",
        monitor='IoU/dataloader_idx_1',
        save_top_k=2,
        mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{step}" + "-my_loss_val",
        monitor='my_loss_val/dataloader_idx_1',
        save_top_k=2,
        mode='min'
    )
    checkpoint_callback_tr = ModelCheckpoint(
        filename="{epoch}-{step}" + "-my_loss",
        monitor='my_loss',
        save_top_k=2,
        mode='min'
    )

    trainer = pl.Trainer(
        progress_bar_refresh_rate=int(not cfg["training"]["no_print"]),
        callbacks=[checkpoint_callback, checkpoint_callback_iou, checkpoint_callback_tr], 
        default_root_dir="./", 
        gpus=cfg["training"]["num_gpu"],
        accelerator='dp',
        resume_from_checkpoint=opt.resume_training,
        num_sanity_val_steps=cfg["training"]["num_sanity"],
        logger=logger,
        max_epochs=cfg["training"]["num_epoch"],
        check_val_every_n_epoch=cfg["training"]["val_every_n_epoch"],
        log_every_n_steps=cfg["training"]["log_every_n_steps"]
    )
    
    trainer.fit(mvpifu)
