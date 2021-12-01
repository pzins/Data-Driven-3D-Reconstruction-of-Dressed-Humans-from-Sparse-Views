import os
from data.BaseDataset import BaseDataset
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils import *
from utils.utils import *
from torch.utils.data import Dataset
import glob


class TestDataset(BaseDataset):
    def __init__(self, cfg, phase='test'):
        super(TestDataset, self).__init__(cfg=cfg, phase=phase)
        self.cfg = cfg
        self.projection_mode = self.cfg["exp"]["projection"]
        self.load_size = self.cfg["exp"]["loadSize"]
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.test_cases = [x[0] for x in os.walk(self.cfg["inference"]["infer_data"])][1:]

        if self.cfg["inference"]["skip_existing"]:
            new_test_cases = []
            for i in self.test_cases:
                tmp = i.split("/")[-1]
                name = os.path.join(self.cfg["inference"]["results_path"],
                                    self.cfg["exp"]["name"], "result_" + tmp + ".png")
                if os.path.isfile(name):
                    print("Skip", name)
                    continue
                new_test_cases.append(i)
            self.test_cases = new_test_cases

    def __len__(self):
        return len(self.test_cases)

    def get_calib(self, param_path):
        if os.path.isfile(param_path):
            if self.projection_mode == "ortho":
                B_MIN = np.array([-1, -1, -1]) * 1
                B_MAX = np.array([1, 1, 1]) * 1
                calib = self.load_data_ortho(param_path)
            elif self.projection_mode == "persp":
                B_MIN = np.array([-40, 0, -40])
                B_MAX = np.array([165, 205, 165])
                calib, proj, rotnorm = self.load_data_persp(param_path)
            elif self.projection_mode == "persp_scene":
                B_MIN = np.array([-1, -1, -1]) * 102.4
                B_MAX = np.array([1, 1, 1]) * 102.4
                calib, proj, offset, translation, factor = self.load_data_persp_scene_inference(param_path)

        elif os.path.isfile(param_path[:-4]+".npz"):
            param = np.load(param_path[:-4]+".npz", allow_pickle=True)
            
            calib = torch.Tensor(param["P"])
            rotnorm = torch.Tensor(param["Rt"].astype(np.float32))
            proj = None
        else:
            # No calibration => singleview ortho PIFu
            assert self.projection_mode == "ortho", "Only orthographic projection is supported without providing any calibration"
            projection_matrix = np.identity(4)
            projection_matrix[1, 1] = -1
            calib = torch.Tensor(projection_matrix).float()

        res = {
            'calib': calib.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX
        }
        if self.projection_mode == "persp":
            res["proj"] = proj.unsqueeze(0)
            res["norm"] = rotnorm.unsqueeze(0)
        elif self.projection_mode == "persp_scene":
            res["Rt"] = calib.unsqueeze(0)
            res["K"] = proj.unsqueeze(0)
            res["offset"] = offset.unsqueeze(0)
            res["translation"] = translation.unsqueeze(0)
            res["resize"] = factor.unsqueeze(0)

        return res

    def get_images(self, image_path, mask_path):
        # Mask
        mask = Image.open(mask_path).convert('L')
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        # image
        image = Image.open(image_path).convert('RGB')
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image
        return {
            'img': image.unsqueeze(0),
        }

    def get_item(self, index):
        test_case = self.test_cases[index]
        print(test_case)
        filenames = glob.glob(os.path.join(test_case, '*'))
        test_images = [f for f in filenames if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
        test_images = sorted(test_images)

        test_masks = [f[:-4]+'_mask.png' for f in test_images]
        test_calib = [f[:-4]+'.npy' for f in test_images]

        list_data = []
        for i in range(len(test_images)):
            data = self.get_images(test_images[i], test_masks[i])
            calib = self.get_calib(test_calib[i])
            data.update(calib)
            list_data.append(data)
        
        
        imgs = []
        calibs = []
        if self.projection_mode == "persp":
            projs = []
            norms = []
        elif self.projection_mode == "persp_scene":
            Rts = []
            Ks = []
            offsets = []
            translations = []
            resizes = []

        for i in list_data:
            imgs.append(i["img"])
            calibs.append(i["calib"])
            if self.projection_mode == "persp":
                if "proj" in i:
                    projs.append(i["proj"])
                if "norm" in i:
                    norms.append(i["norm"])
            elif self.projection_mode == "persp_scene":
                Rts.append(i["Rt"])
                Ks.append(i["K"])
                offsets.append(i["offset"])
                resizes.append(i["resize"])
                translations.append(i["translation"])

        data = {
            "subject": test_case.split("/")[-1],
            "img": torch.cat(imgs),
            "calib": torch.cat(calibs),
            "b_min": list_data[0]["b_min"],
            "b_max": list_data[0]["b_max"],
        }
        if self.projection_mode == "persp":
            data["proj"] = torch.cat(projs)
            data["norm"] = torch.cat(norms)
        elif self.projection_mode == "persp_scene":
            data["translation"] = torch.cat(translations)
            data["offset"] = torch.cat(offsets)
            data["resize"] = torch.cat(resizes)
            data["Rt"] = torch.cat(Rts)
            data["K"] = torch.cat(Ks)

        return data

    def __getitem__(self, index):
        return self.get_item(index)
    
