from src.config import Config_remote
from src.train import train_net, get_device
import os

if __name__=='__main__':

    device = get_device()
    print(f"Device for current machine is: {device}")

    cur_wdir = os.getcwd()

    config_path = ''.join([cur_wdir, '/', 'CNN1D_LINPOOL/Configs/exp_6.json'])
    train_dir = ''
    eval_dir = ''
    epochs_wanted = 100

    config = Config_remote(config_path, train_dir, eval_dir)

    train_net(config, device, epochs_wanted)