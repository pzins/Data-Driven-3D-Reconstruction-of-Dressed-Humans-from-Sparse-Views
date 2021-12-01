import argparse


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Not in config
        parser.add_argument('--config', type=str, default='./configs/ortho.yaml', help="Yaml configuration file")
        parser.add_argument('--resume_training', type=str, help="Path to the checkpoint to resume full training state")
        parser.add_argument('--load_checkpoint', type=str, help="Path to the checkpoint to load model, weights and hyperparams")

        # Special inference
        parser.add_argument('--infer_data', type=str, default="./sample_images", help='The folder of test image')
        parser.add_argument('--resolution', type=int, default=256, help='Resolution of the 3D grid at inference')
        parser.add_argument('--num_points', type=int, default=10000, help='# of points per batch at inference')
        parser.add_argument('--results_path', type=str, default="./results", help='Path to save results')
        parser.add_argument('--no_octree', action='store_true', help='Dont\'t use octree at inference')
        parser.add_argument('--color', action='store_true', help='Generate colored mesh using projection')
        parser.add_argument('--save_npy', action='store_true', help='Save network predictions')
        parser.add_argument('--skip_existing', action='store_true', help='Skip existing reconstructions')
        
        # Overwrite config if needed
        parser.add_argument('--name', type=str, help='Name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataroot', type=str, help='Path to images (data folder)')
        parser.add_argument('--view_fusion', type=str, help='Which method is used to fuse the information from the different views.')
        parser.add_argument('--num_views', type=int, help='How many views to use for multiview network.')
        parser.add_argument('--num_sanity', type=int, help='Number of sanity validation checks. -1 means use all validation data')
        parser.add_argument('--grid_res', type=int, help='Local grids resolution')
        parser.add_argument('--grid_size', type=float, help='Local grids size')
        parser.add_argument('--grid_dim', type=int, help='Local grids number of dimensions')
        parser.add_argument('--partial_grid', action='store_true', help='Use partial local grids (points only along cardinal axes)')
        parser.add_argument('--loss', type=str, help='Loss function to use ["mse" or "bce"]')
        parser.add_argument('--encoder', type=str, help='Feature Extractor to use ["shg", "vgg", "resnet", "hrnet]')
        parser.add_argument('--batch_size', type=int, help='Input batch size')
        parser.add_argument('--learning_rate', type=float, help='Learning rate')
        parser.add_argument('--num_epoch', type=int, help='Number epoch to train')
        parser.add_argument('--num_gpu', type=int, help='Number gpu to train')
        parser.add_argument('--num_sample_inout', type=int, help='Number of sampling points')
        parser.add_argument('--sigma', type=float, help='Displacement from surface for sampling')
        parser.add_argument('--recover_dim', action='store_true', help='Use additional high resolution layers for the SHG')
        parser.add_argument('--no_print', action='store_true', help='Disable progress bar')
        parser.add_argument('--load_parallel', action='store_true', help='Load meshes in parallel')
        parser.add_argument('--num_heads', type=int, help='Number of heads of the transformer')
        parser.add_argument('--use_token', type=int, help='Use extra token for MHA')
        parser.add_argument('--num_stack', type=int, help='Number of hourglass')
        parser.add_argument('--num_hourglass', type=int, help='Number of stacked layer of hourglass')
        parser.add_argument('--num_threads', type=int, help='Number of dataloader workers')
        parser.add_argument('--projection', type=str, help='Specify the projection used (ortho, persp, persp_scene)')
        

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
