from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms
from PIL.ImageFilter import GaussianBlur



class BaseDataset(Dataset):
    '''
    This is the Base Datasets.
    Itself does nothing and is not runnable.
    Check self.get_item function to see what it should return.
    '''

    def __init__(self, cfg, phase='train'):
        self.cfg = cfg
        self.is_train = phase == 'train'
        self.projection_mode = cfg["exp"]["projection"]

    def __len__(self):
        return 0


    def load_data_persp_scene_inference(self, param_path):
        param = np.load(param_path, allow_pickle=True)

        proj = param.item().get('proj')

        factor = param.item().get('resize')
        factor = torch.Tensor(np.array([factor])).float()
        factor = factor.unsqueeze(0)

        view = param.item().get('Rt')

        calib = torch.Tensor(view).float()
        calib = calib.unsqueeze(0)
        proj = torch.Tensor(proj).float()
        proj = proj.unsqueeze(0)

        offset = param.item().get("offset")
        offset = torch.Tensor(offset).float()
        offset = offset.unsqueeze(0)

        translation = param.item().get("translation")
        translation[1] = 0

        translation = torch.Tensor(translation).float()
        translation = translation.unsqueeze(0)
        return calib, proj, offset, translation, factor


    def load_data_persp_scene(self, param_path):
        param = np.load(param_path, allow_pickle=True)
        if "proj" not in param.item() or param.item().get("proj") is None:
            proj = param.item().get('K')
        else:
            proj = param.item().get('proj')
        view = param.item().get('Rt')
        offset = param.item().get('offset')
        resize = param.item().get('resize')
        pos = param.item().get('pos')

        calib = torch.Tensor(view).float()
        proj = torch.Tensor(proj).float()
        offset = torch.Tensor(offset).float()
        resize = torch.Tensor(np.array([resize])).float()
        pos = torch.Tensor(pos).float()
        return calib, proj, offset, resize, pos

    def load_data_persp(self, param_path):
        # loading calibration data
        param = np.load(param_path, allow_pickle=True)
        
        norm = param.item().get('norm')
        proj = param.item().get('proj')
        view = param.item().get('view')
        rot = param.item().get('rot')

        rot = np.hstack((rot, np.zeros((3,1))))
        z = np.zeros((1, 4))
        z[0,3] = 1
        rot = np.vstack((rot, z))

        calib = torch.Tensor(view @ rot @ norm).float()
        proj = torch.Tensor(proj).float()
        
        norm = torch.Tensor(rot @ norm).float()
        return calib, proj, norm

    def load_data_ortho(self, param_path, render=None, mask=None):
        # loading calibration data
        param = np.load(param_path, allow_pickle=True)
        # pixel unit / world unit
        ortho_ratio = param.item().get('ortho_ratio')
        # world unit / model unit
        scale = param.item().get('scale')
        # camera center world coordinate
        center = param.item().get('center')
        # model rotation
        R = param.item().get('R')

        translate = -np.matmul(R, center).reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        # Match camera space to image pixel space
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = scale / ortho_ratio
        scale_intrinsic[1, 1] = -scale / ortho_ratio
        scale_intrinsic[2, 2] = scale / ortho_ratio
        # Match image pixel space to image uv space
        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(self.cfg["exp"]["loadSize"] // 2)
        uv_intrinsic[1, 1] = 1.0 / float(self.cfg["exp"]["loadSize"] // 2)
        uv_intrinsic[2, 2] = 1.0 / float(self.cfg["exp"]["loadSize"] // 2)
        # Transform under image pixel space
        trans_intrinsic = np.identity(4)

        if self.is_train:
            # Pad images
            pad_size = int(0.1 * self.load_size)
            render = ImageOps.expand(render, pad_size, fill=0)
            mask = ImageOps.expand(mask, pad_size, fill=0)

            w, h = render.size
            th, tw = self.load_size, self.load_size

            # random flip
            if self.cfg["data"]["random_flip"] and np.random.rand() > 0.5:
                scale_intrinsic[0, 0] *= -1
                render = transforms.RandomHorizontalFlip(p=1.0)(render)
                mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

            # random scale
            if self.cfg["data"]["random_scale"]:
                rand_scale = random.uniform(0.9, 1.1)
                w = int(rand_scale * w)
                h = int(rand_scale * h)
                render = render.resize((w, h), Image.BILINEAR)
                mask = mask.resize((w, h), Image.NEAREST)
                scale_intrinsic *= rand_scale
                scale_intrinsic[3, 3] = 1

            # random translate in the pixel space
            if self.cfg["data"]["random_trans"]:
                dx = random.randint(-int(round((w - tw) / 10.)),
                                    int(round((w - tw) / 10.)))
                dy = random.randint(-int(round((h - th) / 10.)),
                                    int(round((h - th) / 10.)))
            else:
                dx = 0
                dy = 0

            trans_intrinsic[0, 3] = -dx / float(self.cfg["exp"]["loadSize"] // 2)
            trans_intrinsic[1, 3] = -dy / float(self.cfg["exp"]["loadSize"] // 2)

            x1 = int(round((w - tw) / 2.)) + dx
            y1 = int(round((h - th) / 2.)) + dy

            render = render.crop((x1, y1, x1 + tw, y1 + th))
            mask = mask.crop((x1, y1, x1 + tw, y1 + th))

            render = self.aug_trans(render)

            # random blur
            if self.cfg["data"]["aug_blur"] > 0.00001:
                blur = GaussianBlur(np.random.uniform(0, self.cfg["data"]["aug_blur"]))
                render = render.filter(blur)

        intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
        return calib


    def get_item(self, index):
        print("No implemented")
        exit(0)

    def __getitem__(self, index):
        return self.get_item(index)
