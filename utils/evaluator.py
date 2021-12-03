import os
import numpy as np
import torch
from PIL import Image
from utils import *
from utils.utils import *


class Evaluator:
    def __init__(self, cfg, model, test_dataset):
        self.cfg = cfg
        self.projection_mode = self.cfg["exp"]["projection"]


        netG = model
        os.makedirs(cfg["inference"]["results_path"], exist_ok=True)
        os.makedirs('%s/%s' % (cfg["inference"]["results_path"], cfg["exp"]["name"]), exist_ok=True)

        self.netG = netG
        self.test_dataset = test_dataset

    def evaluate(self):
        for i, data in enumerate(self.test_dataset):
            self.eval_one_subject(data)

    def eval_one_subject(self, data):
        cfg = self.cfg
        with torch.no_grad():
            self.netG.eval()
            save_path = '%s/%s/result_%s.obj' % (cfg["inference"]["results_path"], cfg["exp"]["name"], data['subject'])
            image_tensor = data['img'].cuda()
            calib_tensor = data['calib'].cuda()
            if "proj" in data:
                proj_tensor = data["proj"].cuda()
            else:
                proj_tensor = None
            
            if "norm" in data:
                norm_tensor = data["norm"].cuda()
            else:
                norm_tensor = None
            
            if "Rt" in data:
                Rt_tensor = data["Rt"].cuda()
            else:
                Rt_tensor = None
            
            if "K" in data:
                K_tensor = data["K"].cuda()
            else:
                K_tensor = None
            
            if "resize" in data:
                resize_tensor = data["resize"].cuda()
            else:
                resize_tensor = None
            
            if "offset" in data:
                offset_tensor = data["offset"].cuda()
            else:
                offset_tensor = None
            
            if "translation" in data:
                translation_tensor = data["translation"].cuda()
            else:
                translation_tensor = None
            



            calibration = {
                "calib" : calib_tensor,
                "proj" : proj_tensor,
                "norm" : norm_tensor,
                "Rt" : Rt_tensor,
                "K": K_tensor,
                "resize": resize_tensor,
                "offset": offset_tensor,
                "translation": translation_tensor
            }

            self.netG.filter(image_tensor)
            b_min = data['b_min']
            b_max = data['b_max']

            try:
                save_img_path = save_path[:-4] + '.png'
                save_img_list = []
                for v in range(image_tensor.shape[0]):
                    save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
                    save_img_list.append(save_img)
                save_img = np.concatenate(save_img_list, axis=1)
                Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

                verts, faces, _, _, sdf = self.reconstruction(calibration, b_min, b_max, cfg, num_samples=cfg["inference"]["num_points"])

                if cfg["inference"]["save_npy"] == True:
                    np.save(save_path[:-4], sdf)

                if cfg["inference"]["color"] == True:
                    verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).cuda().float()
                    if cfg["exp"]["projection"] == "persp":
                        calib_tensor = proj_tensor @ calib_tensor
                    xyz_tensor = self.netG.projection(verts_tensor, calib_tensor[:1].cuda())
                    uv = xyz_tensor[:, :2, :]
                    color = project(image_tensor[:1], uv).detach().cpu().numpy()[0].T
                    color = color * 0.5 + 0.5
                    save_obj_mesh_with_color(save_path, verts, faces, color)
                else:
                    save_obj_mesh(save_path, verts, faces)
            except Exception as e:
                print(e)
                print('Can not create marching cubes at this time.')


    def eval_func(self, points, calibs):
        samples = torch.from_numpy(points).cuda().float()
        samples = samples.unsqueeze(0)
        samples = samples.permute(0,2,1)
        # calibs = calibs.unsqueeze(0).cuda()
        self.netG.query(samples, calibs)
        pred = self.netG.get_preds()[0][0]
        return pred.detach().cpu().numpy()





    def reconstruction(self, calibration, b_min, b_max, cfg,
                    num_samples=10000, transform=None, names=None):
        resolution = cfg["inference"]["resolution"]
        b_mins = [b_min]
        b_maxs = [b_max]

        coords_list = []
        for i in range(len(b_mins)):
            coords, mat = create_grid(resolution, resolution, resolution,
                                    b_mins[i], b_maxs[i], transform=transform)
            coords_list.append(coords)


        if cfg["inference"]["no_octree"]:
            sdf = self.eval_grid(coords, self.eval_func, calibration, cfg, num_samples=num_samples)
        else:
            sdfs = None
            for i in range(len(coords_list)):
                sdf = self.eval_grid_octree(coords_list[i], self.eval_func, calibration, cfg, num_samples=num_samples)
                if sdfs is None:
                    sdfs = sdf
                else:
                    sdfs += sdf
            sdf = sdfs / len(coords_list)
        sdf = sdf.astype(np.float16)

        try:
            thres = 0.5
            gdir = 'descent'
            verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, thres, gradient_direction=gdir)
            verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
            verts = verts.T

            if cfg["exp"]["projection"] == "persp_scene" and "translation" in calibration:
                verts += calibration["translation"][0,0,:3].T.cpu().numpy()

            return verts, faces, normals, values, sdf
        except:
            print('error cannot marching cubes')
            return -1




    def batch_eval(self, points, eval_func, calibration, cfg, num_samples=512 * 512 * 512):
        points = points.T
        num_pts = points.shape[0]
        sdf = np.zeros(num_pts)

        num_batches = num_pts // num_samples
        
        for i in range(num_batches):
            sample_points = points[i * num_samples:i * num_samples + num_samples,:]
            
            res = eval_func(sample_points, calibration)
            sdf[i * num_samples:i * num_samples + num_samples] = res


        if num_pts % num_samples:
            sample_points = points[num_batches * num_samples:, :]

            res = eval_func(sample_points, calibration)
            
            sdf[num_batches * num_samples:] = res

        return sdf

    def eval_grid(self, coords, eval_func, calibration, cfg, num_samples=512 * 512 * 512):
        resolution = coords.shape[1:4]
        coords = coords.reshape([3, -1])
        sdf = self.batch_eval(coords, eval_func, calibration, cfg, num_samples=num_samples)
        return sdf.reshape(resolution)


    def eval_grid_octree(self, coords, eval_func, calibration, cfg,
                        init_resolution=64, threshold=0.01,
                        num_samples=512 * 512 * 512):

        resolution = coords.shape[1:4]

        sdf = np.zeros(resolution)

        dirty = np.ones(resolution, dtype=np.bool)
        grid_mask = np.zeros(resolution, dtype=np.bool)

        reso = resolution[0] // init_resolution

        while reso > 0:
            # subdivide the grid
            grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
            # test samples in this iteration
            test_mask = np.logical_and(grid_mask, dirty)
            # print('step size:', reso, 'test sample size:', test_mask.sum())
            points = coords[:, test_mask]
            sdf[test_mask] = self.batch_eval(points, eval_func, calibration, cfg, num_samples=num_samples)
            dirty[test_mask] = False

            # do interpolation
            if reso <= 1:
                break
            for x in range(0, resolution[0] - reso, reso):
                for y in range(0, resolution[1] - reso, reso):
                    for z in range(0, resolution[2] - reso, reso):
                        # if center marked, return
                        if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                            continue
                        v0 = sdf[x, y, z]
                        v1 = sdf[x, y, z + reso]
                        v2 = sdf[x, y + reso, z]
                        v3 = sdf[x, y + reso, z + reso]
                        v4 = sdf[x + reso, y, z]
                        v5 = sdf[x + reso, y, z + reso]
                        v6 = sdf[x + reso, y + reso, z]
                        v7 = sdf[x + reso, y + reso, z + reso]
                        v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                        v_min = v.min()
                        v_max = v.max()
                        # this cell is all the same
                        if (v_max - v_min) < threshold:
                            sdf[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                            dirty[x:x + reso, y:y + reso, z:z + reso] = False
            reso //= 2

        return sdf.reshape(resolution)
