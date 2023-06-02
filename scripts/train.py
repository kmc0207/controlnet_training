import os
import sys
sys.path.append("./")
sys.path.append('./lib/discriminators/')
from lib import utils
from MyModel.model import MyModel
import torch
import wandb
from distutils import dir_util
import warnings
warnings.filterwarnings('ignore')

def train(gpu, CONFIG): 
    torch.cuda.set_device(gpu)
    sys.path.append(CONFIG['BASE']['PACKAGES_PATH'])

    CONFIG['BASE']['GPU_ID'] = gpu
    CONFIG['BASE']['GLOBAL_STEP'] = 0

    model = MyModel(CONFIG)
    # Initialize wandb to gather and display loss on dashboard 
    if CONFIG['BASE']['IS_MASTER'] and CONFIG['WANDB']['TURN_ON']:
        wandb.init(project=CONFIG['BASE']['MODEL_ID'], name=CONFIG['BASE']['RUN_ID'])

    # Training loop
    while CONFIG['BASE']['GLOBAL_STEP'] < CONFIG['BASE']['MAX_STEP']:

        # go one step
        model.go_step()

        if CONFIG['BASE']['IS_MASTER']:
            
            # Save and print loss
            if CONFIG['BASE']['GLOBAL_STEP'] % CONFIG['CYCLE']['LOSS'] == 0:
                model.loss_collector.print_loss()
                model.loss_collector.log_wandb()
                
            # Save image
            if CONFIG['BASE']['GLOBAL_STEP'] % CONFIG['CYCLE']['IMAGE'] == 0:
                utils.save_grid_image(f"{CONFIG['BASE']['SAVE_ROOT_IMGS']}/{str(CONFIG['BASE']['GLOBAL_STEP']).zfill(8)}_train.png", model.train_images)
                utils.save_grid_image(f"{CONFIG['BASE']['SAVE_ROOT_IMGS']}/_latest_train_result.png", model.train_images)

            # Save checkpoint parameters 
            if CONFIG['BASE']['GLOBAL_STEP'] % CONFIG['CYCLE']['CKPT'] == 0:
                model.save_checkpoint()

            if CONFIG['BASE']['GLOBAL_STEP'] % CONFIG['CYCLE']['VALID'] == 0:
                if CONFIG['BASE']['DO_VALID']:
                    model.do_validation()
                    utils.save_grid_image(f"{CONFIG['BASE']['SAVE_ROOT_IMGS']}/{str(CONFIG['BASE']['GLOBAL_STEP']).zfill(8)}_valid.png", model.valid_images)
                    utils.save_grid_image(f"{CONFIG['BASE']['SAVE_ROOT_IMGS']}/_latest_valid_result.png", model.valid_images)

            if CONFIG['BASE']['GLOBAL_STEP'] % CONFIG['CYCLE']['TEST'] == 0:
                if CONFIG['BASE']['DO_TEST']:
                    model.do_test()
                    utils.save_grid_image(f"{CONFIG['BASE']['SAVE_ROOT_IMGS']}/{str(CONFIG['BASE']['GLOBAL_STEP']).zfill(8)}_test.png", model.test_images)
                    utils.save_grid_image(f"{CONFIG['BASE']['SAVE_ROOT_IMGS']}/_latest_test_result.png", model.test_images)

        CONFIG['BASE']['GLOBAL_STEP'] += 1


if __name__ == "__main__":

    # load config
    # CONFIG = utils.load_yaml("./configs.yaml")
    CONFIG = utils.load_jsonnet("./configs.jsonnet")
    sys.path.append(CONFIG['BASE']['PACKAGES_PATH']) 

    # update configs
    CONFIG['BASE']['RUN_ID'] = sys.argv[1] # command line: python train.py {run_id}
    CONFIG['BASE']['GPU_NUM'] = torch.cuda.device_count()

    # save config
    utils.make_dirs(CONFIG)
    utils.print_dict(CONFIG)
    utils.save_json(f"{CONFIG['BASE']['SAVE_ROOT_RUN']}/config_{CONFIG['BASE']['RUN_ID']}", CONFIG)
    dir_util.copy_tree("./MyModel", CONFIG['BASE']['SAVE_ROOT_CODE'])

    # Set up multi-GPU training
    if CONFIG['BASE']['USE_MULTI_GPU']:
        torch.multiprocessing.spawn(train, nprocs=CONFIG['BASE']['GPU_NUM'], args=(CONFIG, ))

    # Set up single GPU training
    else:
        train(0, CONFIG)
