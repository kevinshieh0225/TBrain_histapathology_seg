import os
import yaml
import argparse

def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='config.yml', 
        help='Path to option YAML file.'
        )

    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
    opts_dict['model']['classes'] = len(opts_dict['classes'])

    expname_base = opts_dict['expname']
    expname = expname_base
    num = 0
    while(os.path.isdir(os.path.join('./result', expname))):
        num += 1
        expname = f'{expname_base}-{num}'
    opts_dict['expname'] = expname
    opts_dict['savepath'] = os.path.join('./result', expname)

    return opts_dict