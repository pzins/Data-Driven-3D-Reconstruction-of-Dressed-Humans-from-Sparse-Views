import os
import torch
from utils.options import BaseOptions
from utils.utils import *
from model.MultiViewPIFu import MultiViewPIFu
from utils import config
from utils.evaluator import Evaluator
from data.TestDataset import *


# get options
opt = BaseOptions().parse()

if __name__ == '__main__':

    mvpifu = MultiViewPIFu.load_from_checkpoint(opt.load_checkpoint)
    mvpifu.cfg = config.save_opt_in_cfg(mvpifu.cfg, opt)
    mvpifu.netG.is_test = True
    test_dataset = TestDataset(mvpifu.cfg)
    evaluator = Evaluator(mvpifu.cfg, mvpifu.netG.cuda(), test_dataset)

    evaluator.evaluate()



