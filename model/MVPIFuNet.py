import torch
import torch.nn as nn
import torch.nn.functional as F
from .SHG import *
from .ResNet import *
from .Vgg16 import *
from .HRNet import *
from utils.utils import *
import matplotlib.pyplot as plt
from .MLP import *
from .UNet import UNet


class WeightedMSELoss(nn.Module):
    def __init__(self, gamma=None):
        super(WeightedMSELoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, gt, gamma, w=None):
        gamma = gamma[:,None,None] if self.gamma is None else self.gamma
        weight = gamma * gt + (1.0-gamma) * (1 - gt)
        loss = (weight * (pred - gt).pow(2)).mean()

        if w is not None:
            return (loss * w).mean()
        else:
            return loss.mean()


class MVPIFuNet(nn.Module):
    def __init__(self,
                 cfg,
                 error_term=nn.MSELoss(),
                 is_test=False
                 ):
        super(MVPIFuNet, self).__init__()
        

        self.debug_counter = 0
        self.error_term = error_term
        self.is_test = is_test
        self.preds = None
        self.labels = None
        self.sdf = None
        self.occ = None
        
        self.cfg = cfg
        self.num_views = self.cfg["exp"]["num_views"]
        self.encoder = self.cfg["model"]["encoder"]

        """
        # Force some parameter if not present in the params of an old checkpoint
        self.cfg["model"]["num_heads"] = 1
        # self.cfg["model"]["num_heads"] = 6
        self.cfg["model"]["use_token"] = False
        """

        if self.encoder == "shg":
            self.image_filter = HGFilter(cfg)
            input_feature = 257
            if self.cfg["exp"]["view_fusion"] == "attention" and self.cfg["model"]["num_heads"] == 6:
                input_feature = 258
        elif self.encoder == "hrnet":
            self.image_filter = HRNet(cfg)
            input_feature = 257
        elif self.encoder == "unet":
            self.image_filter = UNet(cfg, 3, 256)
            input_feature = 257
        elif self.encoder == "vgg":
            self.image_filter = Vgg16()
            input_feature = 1473
        elif self.encoder == "resnet":
            self.image_filter = ResNet()
            input_feature = 1025
        else:
            print("Feature Extractor not implemented :", self.encoder)
            exit(0)

        if self.cfg["model"]["skip_hourglass"]:
            input_feature += 64

        self.distance_classifier = MLP(
            filter_channels=[input_feature, 1024, 512, 256, 128, 1],
            num_views=self.cfg["exp"]["num_views"],
            view_fusion=self.cfg["exp"]["view_fusion"],
            grid_res=self.cfg["exp"]["grid_res"],
            grid_dim=self.cfg["exp"]["grid_dim"],
            partial_grid=self.cfg["exp"]["partial_grid"],
            no_residual=self.cfg["model"]["no_residual"]
        )

        self.predict_full = False

        if self.cfg["exp"]["loss"] == "mse":
            self.error_term_classif = nn.MSELoss()
        else:
            self.error_term_classif = nn.BCELoss()

        self.im_feat_list = []
        self.tmpx = None
        self.normx = None

        self.intermediate_preds_list = []

        if self.cfg["exp"]["view_fusion"] == "attention":
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_feature, nhead=self.cfg["model"]["num_heads"])
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
            if self.cfg["model"]["use_token"]:
                self.cls_token = nn.Parameter(torch.randn(1,1, input_feature))

        init_net(self)

    def normalize_depth(self, z):
        if self.cfg["exp"]["projection"] == "ortho":
            z_feat = z * (self.cfg["exp"]["loadSize"] // 2) / self.cfg["exp"]["z_size"]
        elif self.cfg["exp"]["projection"] == "persp" or self.cfg["exp"]["projection"] == "persp_scene":
            z_feat = z
        return z_feat


    def filter(self, images):
        '''
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        '''
        self.images = images.cpu()
        if self.encoder != "shg":
            self.im_feat_list = self.image_filter(images)
        else:
            self.im_feat_list, self.tmpx, self.normx = self.image_filter(images)
            # If it is not in training, only produce the last im_feat
            if not self.training:
                self.im_feat_list = [self.im_feat_list[-1]]

    def query(self, points, calibs, occ=None):
        if self.cfg["exp"]["projection"] == "ortho":
            if not self.is_test:
                points = self.world_to_view_ortho(points, calibs["calib"][...,0,:,:])
                points = self.create_local_grids(points)
                points = self.view_to_world_ortho(points, calibs["calib"][...,0,:,:])
                # self.debug_samples(points)
                points = self.world_to_view_ortho(points, calibs["calib"])
            else:
                # points = self.view_to_world_ortho(points, calibs["calib"][...,0,:,:])
                # points = self.world_to_view_ortho(points, calibs["calib"][...,0,:,:])
                points = self.create_local_grids(points)
                points = self.view_to_world_ortho(points, calibs["calib"][...,0,:,:])
                points = self.world_to_view_ortho(points, calibs["calib"])
        elif self.cfg["exp"]["projection"] == "persp":
            points = self.create_local_grids(points)
            points = self.world_to_view_persp(points, calibs)
        elif self.cfg["exp"]["projection"] == "persp_scene":
            if not self.is_test:
                points = self.create_local_grids(points)
                points = self.world_to_view_persp_scene(points, calibs)
            else:
                points = self.create_local_grids(points)
                points = self.world_to_view_persp_scene_inference(points, calibs)

        if not self.is_test:# and self.cfg["exp"]["num_views"] > 1:
            points = points.view(
                points.shape[0] * points.shape[1],
                points.shape[2],
                points.shape[3]
            )

        self.occ = occ
        xyz = points
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        
        if self.cfg["exp"]["partial_grid"]:
            idx = np.arange(
                    (self.cfg["exp"]["grid_res"]*self.cfg["exp"]["grid_dim"] - (self.cfg["exp"]["grid_dim"] - 1)) // 2,
                    in_img.shape[1],
                    self.cfg["exp"]["grid_res"]*self.cfg["exp"]["grid_dim"] - (self.cfg["exp"]["grid_dim"] - 1)
            )
        else:
            idx = np.arange(
                self.cfg["exp"]["grid_res"]**self.cfg["exp"]["grid_dim"] // 2,
                in_img.shape[1],
                self.cfg["exp"]["grid_res"]**self.cfg["exp"]["grid_dim"]
            )
        
        
        in_img = in_img[:, idx]

        if self.num_views > 1:
            in_img = in_img.view(-1, self.cfg["exp"]["num_views"], in_img.shape[1])
            in_img = torch.sum(in_img, 1)
            in_img[in_img < self.num_views] = 0
            in_img = in_img * 1 / self.num_views
        in_img.unsqueeze_(1)
        
        # Debug projections
        # self.debug_projections(xy)
        # Debug features
        # self.debug_features()
        
        z_feat = self.normalize_depth(z)

        if self.cfg["model"]["skip_hourglass"]:
            tmpx_local_feature = project(self.tmpx, xy)

        self.intermediate_preds_list = []
        for im_feat in self.im_feat_list:
            if self.cfg["model"]["num_heads"] == 1:
                point_local_feat_list = [project(im_feat, xy), z_feat]
            elif self.cfg["model"]["num_heads"] == 6:
                point_local_feat_list = [project(im_feat, xy), z_feat, z_feat]
            
            
            if self.cfg["model"]["skip_hourglass"]:
                point_local_feat_list.append(tmpx_local_feature)

            point_local_feat = torch.cat(point_local_feat_list, 1)
            
            if self.cfg["exp"]["view_fusion"] == "attention":
                bs = point_local_feat.shape[0] // self.cfg["exp"]["num_views"]
                point_local_feat = point_local_feat.reshape(-1, self.cfg["exp"]["num_views"], point_local_feat.shape[1], point_local_feat.shape[2])
                point_local_feat = point_local_feat.permute(1, 0, 3, 2)
                point_local_feat = point_local_feat.reshape(self.cfg["exp"]["num_views"], -1, point_local_feat.shape[3])
                
                if self.cfg["model"]["use_token"]:
                    cls_token = self.cls_token.repeat(1,point_local_feat.shape[1], 1)
                    point_local_feat = torch.cat([cls_token, point_local_feat], 0)
                
                tmp = self.transformer_encoder(point_local_feat)
                
                if self.cfg["model"]["use_token"]:
                    tmp = tmp[0]
                else:
                    tmp = torch.mean(tmp, 0)
                
                tmp = tmp.reshape(bs, -1, tmp.shape[1])
                point_local_feat = tmp.permute(0, 2, 1)
            res = self.distance_classifier(point_local_feat)
            if res.shape[2] != in_img[:,None].shape[2]:
                in_img = in_img[:,:res.shape[2]]
            pred = in_img.float() * res

            self.intermediate_preds_list.append(pred)
        self.preds = self.intermediate_preds_list[-1]

        


    def get_im_feat(self):
        return self.im_feat_list[-1]

    def get_error(self):
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.error_term_classif(preds, self.occ)
        error /= len(self.intermediate_preds_list)
        return error

    def get_preds(self):
        return self.preds
    
    
    def world_to_view_persp_scene_inference(self, points, calibs):
        points = torch.cat([points, torch.ones((points.shape[0], 1, points.shape[2])).cuda()], 1)
        sample_points_z = calibs["Rt"][...,:3] @ points[:,:3]
        sample_points_z /= 100
        
        calibs["translation"][0,0,3] = 0
        sample_points_world = points[0] + calibs["translation"][0,0]
        
        sample_points_view = calibs["K"] @ calibs["Rt"] @ sample_points_world
        sample_points_view[...,0,:] /= sample_points_view[...,-1,:]
        sample_points_view[...,1,:] /= -sample_points_view[...,-1,:]
        sample_points_view[...,2,:] /= sample_points_view[...,-1,:]
        sample_points_view = sample_points_view[...,:3,:]

        original_img_size = 2048
        for k in range(sample_points_view.shape[0]):
            sample_points_view[k,:,:2,:] = (sample_points_view[k,:,:2,:]+1) * (original_img_size/2) * calibs["resize"][k,...]
        sample_points_view[...,:2,:] -= calibs["offset"]
        sample_points_view = (sample_points_view / 256) - 1

        # For Z
        sample_points_view[...,2:3,:] = sample_points_z[...,2:3,:]
        sample_points_view = sample_points_view.permute(1,0,2,3)
        return sample_points_view[0]
    

    def world_to_view_persp_scene(self, points, calibs):
        points = torch.cat([points, torch.ones((points.shape[0], 1, points.shape[2])).cuda()], 1)
        sample_points_view = calibs["calib"] @ points

        z = calibs["calib"][...,:3,:3] @ (points[...,:3,:] - calibs["pos"][...,0,:].reshape(3,1))
        z /= 100

        sample_points_view = calibs["proj"] @ sample_points_view
        sample_points_view[...,0,:] /= sample_points_view[...,-1,:]
        sample_points_view[...,1,:] /= -sample_points_view[...,-1,:]
        sample_points_view[...,2,:] /= sample_points_view[...,-1,:]

        original_img_size = 2048
        for k in range(sample_points_view.shape[1]):
            sample_points_view[...,k,:2,:] = (sample_points_view[...,k,:2,:] + 1) * calibs["resize"][...,k,:] * original_img_size / 2

        sample_points_view[...,:2,:] -= calibs["offset"]
        sample_points_view = (sample_points_view / 256) - 1

        sample_points_view[...,2:3,:] = z[...,2:3,:]
        return sample_points_view

    def world_to_view_persp(self, points, calibs):
        points = torch.cat([points, torch.ones((points.shape[0], 1, points.shape[2])).cuda()], 1)
        
        sample_points_view = calibs["calib"] @ points

        z = calibs["norm"] @ points
        z = z[...,2:3,:]

        sample_points_view = calibs["proj"] @ sample_points_view
        sample_points_view[...,0,:] /= sample_points_view[...,3,:]
        sample_points_view[...,1,:] /= -sample_points_view[...,3,:]
        sample_points_view[...,2,:] /= sample_points_view[...,3,:]
        
        sample_points_view[...,2:3,:] = z
        return sample_points_view


    def world_to_view_ortho(self, points, calib):
        points = torch.cat([points, torch.ones((points.shape[0], 1, points.shape[2])).cuda()], 1)
        points_view = calib @ points
        points_view = points_view[...,:3,:]
        return points_view

    def view_to_world_ortho(self, points, calib):
        points = torch.cat([points, torch.ones((points.shape[0], 1, points.shape[2])).cuda()], 1)
        points = torch.inverse(calib) @ points
        points = points[...,:3,:]
        return points
    
    def create_local_grids(self, points):
        num_points = points.shape[-1]
        if self.cfg["exp"]["grid_res"] > 1:
            if self.cfg["exp"]["partial_grid"]:
                points = torch.repeat_interleave(points, self.cfg["exp"]["grid_res"]*self.cfg["exp"]["grid_dim"] - (self.cfg["exp"]["grid_dim"] - 1), 2)
            else:
                points = torch.repeat_interleave(points, self.cfg["exp"]["grid_res"]**self.cfg["exp"]["grid_dim"], 2)

            begin = -(self.cfg["exp"]["grid_res"] // 2)
            end = (self.cfg["exp"]["grid_res"] // 2)
            n = self.cfg["exp"]["grid_res"]
            r = np.linspace(begin, end, n)
            assert len(r) % 2 == 1, "Local grid resolution should be odd"

            tmp = np.empty((0,3))
            if self.cfg["exp"]["grid_dim"] == 3:
                if self.cfg["exp"]["partial_grid"]:
                    r = np.delete(r, len(r)//2)
                    for d in r:
                        tmp = np.vstack((tmp, np.array([0, 0, d])))
                    for h in r:
                        tmp = np.vstack((tmp, np.array([0, h, 0])))
                    for w in r:
                        tmp = np.vstack((tmp, np.array([w, 0, 0])))
                    tmp = np.insert(tmp, tmp.shape[0]//2, np.array([0, 0, 0]), 0)
                else:
                    for d in r:
                        for h in r:
                            for w in r:
                                tmp = np.vstack((tmp, np.array([w, h, d])))
            else:
                if self.cfg["exp"]["partial_grid"]:
                    r = np.delete(r, len(r)//2)
                    for h in r:
                        tmp = np.vstack((tmp, np.array([0, h, 0])))
                    for w in r:
                        tmp = np.vstack((tmp, np.array([w, 0, 0])))
                    tmp = np.insert(tmp, tmp.shape[0]//2, np.array([0, 0, 0]), 0)
                else:
                    for h in r:
                        for w in r:
                            tmp = np.vstack((tmp, np.array([w, h, 0])))
            tmp = tmp.astype(np.float32) * (1 / end) * self.cfg["exp"]["grid_size"]
            tmp = np.tile(tmp, (num_points, 1))
            tmp = torch.Tensor(tmp).cuda()
            points = points + tmp.T
        return points


    def forward(self, images, points, calibs, transforms=None, occ=None, masks=None):
        # Merge views and batch dimensions
        images = images.view(
            images.shape[0] * images.shape[1],
            images.shape[2],
            images.shape[3],
            images.shape[4]
        )
        
        if self.num_views == 1 and len(points.size()) > 3:
            points = points.view(points.size(0)*points.size(1), points.size(2), points.size(3))

        self.filter(images)
        self.query(points=points, occ=occ, calibs=calibs)
        res = self.get_preds()
        error = self.get_error()
        
        return res, error.unsqueeze(0)



    def debug_projections(self, xy):
        for i in range(self.cfg["exp"]["num_views"]):
            img = self.images[i]
            img = (img + 1) / 2
            resized_xy = (xy[i] + 1) * img.shape[1] * 0.5
            fig = plt.figure()
            img = np.transpose(img, (1,2,0))
            plt.imshow(img)
            plt.scatter(resized_xy[0, :].cpu(), resized_xy[1, :].cpu(), s=5, marker='o', c="red")
            filename = str(self.debug_counter) + '-img.png'
            plt.savefig(filename)
            print(filename)
            plt.close()
            self.debug_counter+=1
        exit(0)

    def debug_features(self):
        for i, v in enumerate(self.im_feat_list):
            print(v.shape)
            np.save("im_feat_" + str(i), v.cpu().detach().numpy())

    def debug_samples(self, points):
        data = np.hstack((np.array(["v"] * points.shape[2]).reshape(-1,1), points[0].permute(1,0).cpu().detach().numpy()))
        np.savetxt("samples.obj", data,  fmt="%s %s %s %s", delimiter=" ")
        exit(0)
