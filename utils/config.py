import yaml


# General config
def load_config(path):
    ''' Loads config file.

    Args:  
        path (str): path to config file
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
    
def overwrite_options(cfg, opt):
    if opt.dataroot is not None:
        cfg["data"]["dataroot"] = opt.dataroot

    if opt.batch_size is not None:
        cfg["training"]["batch_size"] = opt.batch_size
    if opt.learning_rate is not None:
        cfg["training"]["learning_rate"] = opt.learning_rate
    if opt.num_epoch is not None:
        cfg["training"]["num_epoch"] = opt.num_epoch
    if opt.num_sanity is not None:
        cfg["training"]["num_sanity"] = opt.num_sanity
    if opt.num_gpu is not None:
        cfg["training"]["num_gpu"] = opt.num_gpu
    if opt.no_print is not None:
        cfg["training"]["no_print"] = opt.no_print
    if opt.num_threads is not None:
        cfg["training"]["num_threads"] = opt.num_threads
    
    if opt.name is not None:
        cfg["exp"]["name"] = opt.name
    if opt.view_fusion is not None:
        cfg["exp"]["view_fusion"] = opt.view_fusion
    if opt.num_views is not None:
        cfg["exp"]["num_views"] = opt.num_views
    if opt.load_parallel is not None:
        cfg["exp"]["load_parallel"] = opt.load_parallel
    if opt.grid_res is not None:
        cfg["exp"]["grid_res"] = opt.grid_res
    if opt.grid_size is not None:
        cfg["exp"]["grid_size"] = opt.grid_size
    if opt.grid_dim is not None:
        cfg["exp"]["grid_dim"] = opt.grid_dim
    if opt.partial_grid is not None:
        cfg["exp"]["partial_grid"] = opt.partial_grid
    if opt.loss is not None:
        cfg["exp"]["loss"] = opt.loss
    if opt.num_sample_inout is not None:
        cfg["exp"]["sampling"]["num_sample_inout"] = opt.num_sample_inout
    if opt.sigma is not None:                                                                                                                                                                                                           
        cfg["exp"]["sampling"]["sigma"] = opt.sigma

    if opt.recover_dim is not None:
        cfg["model"]["recover_dim"] = opt.recover_dim
    if opt.encoder is not None:
        cfg["model"]["encoder"] = opt.encoder
    if opt.num_heads is not None:
        cfg["model"]["num_heads"] = opt.num_heads
    if opt.use_token is not None:
        cfg["model"]["use_token"] = opt.use_token
    if opt.num_stack is not None:
        cfg["model"]["num_stack"] = opt.num_stack
    if opt.num_hourglass is not None:
        cfg["model"]["num_hourglass"] = opt.num_hourglass
    return cfg

def overwrite_options_resume_training(cfg, opt):
    if opt.num_gpu is not None:
        cfg["training"]["num_gpu"] = opt.num_gpu
    if opt.dataroot is not None:
        cfg["data"]["dataroot"] = opt.dataroot
    if opt.batch_size is not None:
        cfg["training"]["batch_size"] = opt.batch_size
    if opt.no_print is not None:
        cfg["training"]["no_print"] = opt.no_print
    if opt.num_threads is not None:
        cfg["training"]["num_threads"] = opt.num_threads
    if opt.num_epoch is not None:
        cfg["training"]["num_epoch"] = opt.num_epoch
    if opt.name is not None:
        cfg["exp"]["name"] = opt.name
    return cfg

def save_opt_in_cfg(cfg, opt):
    if "inference" not in cfg:
        cfg["inference"] = {}
    if opt.grid_size is not None:
        cfg["exp"]["grid_size"] = opt.grid_size
    if opt.grid_res is not None:
        cfg["exp"]["grid_res"] = opt.grid_res
    if opt.infer_data is not None:
        cfg["inference"]["infer_data"] = opt.infer_data
    if opt.resolution is not None:
        cfg["inference"]["resolution"] = opt.resolution
    if opt.results_path is not None:
        cfg["inference"]["results_path"] = opt.results_path
    if opt.num_points is not None:
        cfg["inference"]["num_points"] = opt.num_points
    
    if opt.no_octree is not None:
        cfg["inference"]["no_octree"] = opt.no_octree
    else:
        cfg["inference"]["no_octree"] = False
    
    if opt.color is not None:
        cfg["inference"]["color"] = opt.color
    else:
        cfg["inference"]["color"] = False
    
    if opt.save_npy is not None:
        cfg["inference"]["save_npy"] = opt.save_npy
    else:
        cfg["inference"]["save_npy"] = False
    
    if opt.skip_existing is not None:
        cfg["inference"]["skip_existing"] = opt.skip_existing
    else:
        cfg["inference"]["skip_existing"] = False
    
    if opt.projection is not None:
        cfg["exp"]["projection"] = opt.projection
    
    return cfg
