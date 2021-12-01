import numpy as np
import os
import random
import logging
import numpy as np
from PIL import Image, ImageOps
from PIL.ImageFilter import GaussianBlur
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import trimesh
import logging
import multiprocessing as mp
from multiprocessing import Pool
from utils.evaluator import create_grid
from data.BaseDataset import BaseDataset
import json

log = logging.getLogger('trimesh')
log.setLevel(40)

def load_trimesh(root_dir, used_subjects):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        if sub_name in used_subjects:
            meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_100k.obj' % sub_name))
    return meshs


class TrainDataset(BaseDataset):
    def __init__(self, cfg, phase='train'):
        super(TrainDataset, self).__init__(cfg=cfg, phase=phase)
        self.cfg = cfg

        # Path setup
        self.root = self.cfg["data"]["dataroot"]
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')
        self.DIST = os.path.join(self.root, 'DIST')
        self.SDF = os.path.join(self.root, 'SDF')
        self.SDFGEN = os.path.join(self.root, 'SDFGEN')


        self.B_MIN = np.array([-128, -28, -128])
        self.B_MAX = np.array([128, 228, 128])

        self.load_size = self.cfg["exp"]["loadSize"]

        self.num_views = self.cfg["exp"]["num_views"]

        self.num_sample_inout = self.cfg["exp"]["sampling"]["num_sample_inout"]

        self.yaw_list = list(range(0,360,45))
        # self.yaw_list = [0]


        self.pitch_list = [0]
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=cfg["data"]["aug_bri"], contrast=cfg["data"]["aug_con"], saturation=cfg["data"]["aug_sat"],
                                   hue=cfg["data"]["aug_hue"])
        ])

        # Single thread version
        if not self.cfg["exp"]["load_parallel"]:
            self.mesh_dic = load_trimesh(self.OBJ, self.subjects)
        else:
            # Parallel version
            num_cpu = mp.cpu_count()
            pool = Pool(num_cpu)
            mesh_dic = pool.map(self.load_trimesh, self.subjects)
            self.mesh_dic = {}
            for i in mesh_dic:
                self.mesh_dic[i[0]] = i[1]
        print("Dataset size :", len(self.mesh_dic))

    def load_trimesh(self, subject):
        ret = trimesh.load(os.path.join(self.OBJ, subject , '%s_100k.obj' % subject))
        return subject, ret


    def get_subjects(self):
        tr_file = self.cfg["data"]["train_subjects_list"]
        val_file = self.cfg["data"]["val_subjects_list"]

        if not os.path.isfile(tr_file) or not os.path.isfile(val_file):
            print("Files with list of training and validation subjects not found")
            exit(0)
        
        val_subjects = list(np.loadtxt(val_file, dtype=str))
        tr_subjects = list(np.loadtxt(tr_file, dtype=str))

        assert len(tr_subjects) > 0 and len(val_subjects) > 0

        if self.is_train:
            return sorted(tr_subjects)
        else:
            return sorted(val_subjects)

    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):

        pitch = self.pitch_list[pid]
        if self.cfg["exp"]["projection"] == "persp_scene":
            with open(os.path.join(self.RENDER, subject, "yaw_pitch.json"), 'r') as f:
                yaw_pitch = json.load(f)

        # The ids are an even distribution of num_views around view_id
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]

        # Add random offset
        for i in range(len(view_ids)):
            view_ids[i] = (np.random.randint(-20, 21, 1) + view_ids[i]) % 360
        
        calib_list = []
        render_list = []
        if self.projection_mode == "persp":
            proj_list = []
            norm_list = []
        elif self.projection_mode == "persp_scene":
            proj_list = []
            offset_list = []
            resize_list = []
            pos_list = []

        for vid in view_ids:
            if self.cfg["exp"]["projection"] == "persp_scene":
                pitch = yaw_pitch[str(vid[0])]

            param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0))
            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')
            
            if self.projection_mode == "ortho":
                calib = self.load_data_ortho(param_path, render, mask)
            elif self.projection_mode == "persp":
                calib, proj, norm = self.load_data_persp(param_path)
            elif self.projection_mode == "persp_scene":
                calib, proj, offset, resize, pos = self.load_data_persp_scene(param_path)

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            if self.projection_mode == "persp":
                proj_list.append(proj)
                norm_list.append(norm)
            elif self.projection_mode == "persp_scene":
                proj_list.append(proj)
                offset_list.append(offset)
                resize_list.append(resize)
                pos_list.append(pos)    

        if self.projection_mode == "persp":
            res = {
                'img': torch.stack(render_list, dim=0),
                'calibs': {
                    'calib': torch.stack(calib_list, dim=0),
                    'proj': torch.stack(proj_list, dim=0),
                    'norm': torch.stack(norm_list, dim=0)
                }
            }
        elif self.projection_mode == "persp_scene":
            res = {
                'img': torch.stack(render_list, dim=0),
                'calibs': {
                    'calib': torch.stack(calib_list, dim=0),
                    'proj': torch.stack(proj_list, dim=0),
                    'offset': torch.stack(offset_list, dim=0),
                    'resize': torch.stack(resize_list, dim=0),
                    'pos': torch.stack(pos_list, dim=0)
                }
            }

        elif self.projection_mode == "ortho":
            res = {
                'img': torch.stack(render_list, dim=0),
                'calibs': {
                    'calib': torch.stack(calib_list, dim=0)
                }
            }
        return res


    def select_sampling_method(self, subject, data, index):
        # if not self.is_train:
        #     random.seed(1991)
        #     np.random.seed(1991)
        #     torch.manual_seed(1991)

        # ======= Select equal number of points inside and outside following the same strategy as original PIFu =======
        mesh = self.mesh_dic[subject]
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
        
        sample_points = surface_points + np.random.normal(scale=self.cfg["exp"]["sampling"]["sigma"], size=surface_points.shape)


        # add random points within image space
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)

        # Prepare points inside and outside (num_sample_inout // 2)
        inside = mesh.contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                               :(self.num_sample_inout - nin)]
        
        sample_points = np.concatenate([inside_points, outside_points], 0).T

        # Get index of inside points in model coordinates
        inside = mesh.contains(sample_points.T)

        # Create occupancy labels
        occ = np.ones((sample_points.shape[1], 1), dtype=np.float32)
        occ[np.logical_not(inside)] = 0
        occ = occ.T
        occ = torch.Tensor(occ).float()

        samples = torch.Tensor(sample_points).float()

        res = {
            'samples': samples,
            'name': subject
        }
                
        res['occ'] = occ

        del mesh

        return res

    def get_item(self, index):
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)
        pid = tmp // len(self.yaw_list)

        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid)
        res.update(render_data)

        if self.cfg["exp"]["sampling"]["num_sample_inout"]:
            sample_data = self.select_sampling_method(subject, res, index)
            res.update(sample_data)

        return res

    def __getitem__(self, index):
        return self.get_item(index)
