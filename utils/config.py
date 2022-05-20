import os, torch
import yaml
import wandb
from utils.network import Litsmp
from pytorch_lightning.loggers import WandbLogger

def loadmodel(pretrain_path, load_last = False, config_path = 'None' ):
    for pth in os.listdir(pretrain_path):
        if '.ckpt' in pth:
            if 'last' in pth and load_last == True:
                weight = os.path.join(pretrain_path, pth)
                print(weight)
                break
            elif 'last' not in pth and load_last == False:
                weight = os.path.join(pretrain_path, pth)
                print(weight)
                break
    checkpoint_dict = torch.load(weight)
    # loading your own config (soup)
    if config_path != 'None':
        opts_dict = load_wdb_config(config_path)
        model = Litsmp.load_from_checkpoint(weight, opts_dict=opts_dict)
    # loading predefined config from the chekpoint
    elif 'hyper_parameters' in checkpoint_dict:
        opts_dict = checkpoint_dict['hyper_parameters']
        model = Litsmp.load_from_checkpoint(weight)
    
    return opts_dict, model

def load_wdb_config(
        cfgpath='./result/Unet_efnb4_nonorm/expconfig.yaml',
        ):
    with open(cfgpath, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
    unflatten_json(opts_dict)
    opts_dict['expname'] = cfgpath.split('/')[-2]

    return opts_dict

def load_setting(cfgpath = './cfg/setting.yaml'):
    with open(cfgpath, 'r') as fp:
        ds_dict = yaml.load(fp, Loader=yaml.FullLoader)
    ds_dict['dataset_root'] = os.path.join(ds_dict['root'], ds_dict['dataset_root'])
    ds_dict['crop_dataset_root'] = os.path.join(ds_dict['root'], ds_dict['crop_dataset_root'])
    ds_dict['train_valid_list'] = os.path.join(ds_dict['listroot'], ds_dict['train_valid_list'])
    ds_dict['public_root'] = os.path.join(ds_dict['root'], ds_dict['public_root'])
    ds_dict['inference_root'] = os.path.join(ds_dict['root'], ds_dict['inference_root'])
    ds_dict['crop_public_root'] = os.path.join(ds_dict['root'], ds_dict['crop_public_root'])

    return ds_dict


def wandb_config(project, name, cfg='cfg/wandbcfg.yaml'):
    expname = searchnewname(name)
    wandb_logger = WandbLogger(project=project,
                               entity="aicup2022",
                               name=expname,
                               config=cfg,
                               reinit=True)
    # wandb_logger.experiment (the wandb run) is only initialized on rank0,
    # but we need every proc to get the wandb sweep config, which happens on .init
    # so we have to call .init on non rank0 procs, but we disable creating a new run
    if not isinstance(wandb_logger.experiment.config, wandb.sdk.wandb_config.Config):
        wandb.init(config=cfg, mode="disabled", reinit=True)
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

    return opts_dict, wandb_logger

def searchnewname(expname_base, root='./result'):
    expname = expname_base
    num = 0
    while(os.path.isdir(os.path.join(root, expname))):
        num += 1
        expname = f'{expname_base}-{num}'
    return expname


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
