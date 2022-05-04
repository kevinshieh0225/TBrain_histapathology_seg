import os
import yaml
import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger


def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='config.yaml', 
        help='Path to option YAML file.'
        )

    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
    opts_dict['model']['classes'] = len(opts_dict['classes'])

    expname_base = opts_dict['expname']
    expname = searchnewname(expname_base)
    opts_dict['expname'] = expname
    opts_dict['savepath'] = os.path.join('./result', expname)

    return opts_dict

def load_wdb_config(
        cfgpath='./result/dlv3p-Unet_resize-sweep-2/expconfig.yaml',
        inference='/work/u7085556/TBrain_histapathology_seg'
        ):
    with open(cfgpath, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
    unflatten_json(opts_dict)
    opts_dict['expname'] = cfgpath.split('/')[-2]
    opts_dict['inference'] = inference

    norm = norm_config('./cfg/normalize.yaml')
    opts_dict.update(norm)

    return opts_dict

def wandb_config(project, name, cfg='cfg/wandbcfg.yaml'):
    expname = searchnewname(name)
    wandb_logger = WandbLogger(project=project, name=expname, config=cfg)
    # wandb_logger.experiment (the wandb run) is only initialized on rank0,
    # but we need every proc to get the wandb sweep config, which happens on .init
    # so we have to call .init on non rank0 procs, but we disable creating a new run
    if not isinstance(wandb_logger.experiment.config, wandb.sdk.wandb_config.Config):
        wandb.init(config=cfg, mode="disabled")
        opts_dict = wandb.config.as_dict()
    else:
        opts_dict = wandb_logger.experiment.config.as_dict()
        save_path = os.path.join('./result', expname)
        os.makedirs(save_path, exist_ok=True)
        ymlsavepath = os.path.join(save_path, 'expconfig.yaml')
        with open(ymlsavepath, 'w') as yaml_file:
            yaml.dump(opts_dict, yaml_file, default_flow_style=False)

    unflatten_json(opts_dict)

    opts_dict['project'] = project
    opts_dict['expname'] = expname
    opts_dict['savepath'] = os.path.join('./result', expname)
    opts_dict['model']['classes'] = len(opts_dict['classes'])

    norm = norm_config('./cfg/normalize.yaml')
    opts_dict.update(norm)

    return opts_dict, wandb_logger

def searchnewname(expname_base):
    expname = expname_base
    num = 0
    while(os.path.isdir(os.path.join('./result', expname))):
        num += 1
        expname = f'{expname_base}-{num}'
    return expname

def norm_config(cfgpath):
    with open(cfgpath, 'r') as fp:
        norm = yaml.load(fp, Loader=yaml.FullLoader)
    return norm

def flatten_json(json):
    if type(json) == dict:
        for k, v in list(json.items()):
            if type(v) == dict:
                flatten_json(v)
                json.pop(k)
                for k2, v2 in v.items():
                    json[k+"."+k2] = v2

                    
def unflatten_json(json):
    if type(json) == dict:
        for k in sorted(json.keys(), reverse=True):
            if "." in k:
                key_parts = k.split(".")
                json1 = json
                for i in range(0, len(key_parts)-1):
                    k1 = key_parts[i]
                    if k1 in json1:
                        json1 = json1[k1]
                        if type(json1) != dict:
                            conflicting_key = ".".join(key_parts[0:i+1])
                            raise Exception('Key "{}" conflicts with key "{}"'.format(
                                k, conflicting_key))
                    else:
                        json2 = dict()
                        json1[k1] = json2
                        json1 = json2
                if type(json1) == dict:
                    v = json.pop(k)
                    json1[key_parts[-1]] = v