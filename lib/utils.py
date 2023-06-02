import torch
import torch.nn as nn
import torchvision
import cv2
import os
import glob

import os, yaml, json
import _jsonnet

def print_dict(dict):
    print(json.dumps(dict, sort_keys=True, indent=4))

def load_jsonnet(load_path):
    return json.loads(_jsonnet.evaluate_file(load_path))

def save_json(save_path, dict):
    with open(save_path+'.jsonnet', 'w') as f:
        json.dump(dict, f, indent=4, sort_keys=True)
        
def load_yaml(load_path):
    with open(load_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def save_yaml(save_path, dict):
    with open(save_path, 'w') as f:
        yaml.dump(dict, f)

def make_dirs(CONFIG):
    os.makedirs(CONFIG['BASE']['SAVE_ROOT'], exist_ok=True)

    train_result_dirs = os.listdir(CONFIG['BASE']['SAVE_ROOT'])
    if train_result_dirs:
        last_train_index = sorted(train_result_dirs)[-1][:3]
        CONFIG['BASE']['RUN_ID'] = f"{str(int(last_train_index)+1).zfill(3)}_{CONFIG['BASE']['RUN_ID']}"
    
    else:
        CONFIG['BASE']['RUN_ID'] = f"000_{CONFIG['BASE']['RUN_ID']}"    

    CONFIG['BASE']['SAVE_ROOT_RUN'] = f"{CONFIG['BASE']['SAVE_ROOT']}/{CONFIG['BASE']['RUN_ID']}"
    os.makedirs(CONFIG['BASE']['SAVE_ROOT_RUN'], exist_ok=True)

    CONFIG['BASE']['SAVE_ROOT_CKPT'] = f"{CONFIG['BASE']['SAVE_ROOT_RUN']}/ckpt"
    CONFIG['BASE']['SAVE_ROOT_IMGS'] = f"{CONFIG['BASE']['SAVE_ROOT_RUN']}/imgs"
    CONFIG['BASE']['SAVE_ROOT_CODE'] = f"{CONFIG['BASE']['SAVE_ROOT_RUN']}/code"
    os.makedirs(CONFIG['BASE']['SAVE_ROOT_CKPT'], exist_ok=True)
    os.makedirs(CONFIG['BASE']['SAVE_ROOT_IMGS'], exist_ok=True)
    os.makedirs(CONFIG['BASE']['SAVE_ROOT_CODE'], exist_ok=True)

def get_all_images(dataset_root_list):
    image_paths = []

    for dataset_root in dataset_root_list:
        image_paths += glob.glob(f"{dataset_root}/*.*g")
        for root, dirs, _ in os.walk(dataset_root):
            for dir in dirs:
                image_paths += glob.glob(f"{root}/{dir}/*.*g")

    return sorted(image_paths)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
        
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)

def update_net(model, optimizer, loss, use_mGPU=False):
    optimizer.zero_grad()  
    loss.backward()   
    if use_mGPU:
        size = float(torch.distributed.get_world_size())
        for param in model.parameters():
            if param.grad == None:
                continue            
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= size
    optimizer.step()  

def setup_ddp(gpu, ngpus_per_node, PORT):
    torch.distributed.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:{PORT}',
            world_size=ngpus_per_node,
            rank=gpu)

def save_grid_image(img_path, images_list):
    grid_rows = []

    for images in images_list:
        # images = images[:8] # Drop images if there are more than 8 images in the list
        grid_row = torchvision.utils.make_grid(images, nrow=images.shape[0]) * 0.5 + 0.5
        grid_rows.append(grid_row)

    grid = torch.cat(grid_rows, dim=1)
    grid = grid.detach().cpu().numpy().transpose([1,2,0]) * 255

    cv2.imwrite(img_path, grid[:,:,::-1])
    
def stack_image_grid(batch_data_items : list, target_image : list):
    column = []
    for item in batch_data_items:
        column.append(item)
    target_image.append(torch.cat(column, dim=-2))

def load_checkpoint(CONFIG, model, optimizer, type):

    FLAG = False 
    for run_id in os.listdir('./train_results'):
        if int(run_id[:3]) == CONFIG['CKPT']['ID_NUM']:
            CONFIG['CKPT']['ID'] = run_id
            FLAG = True

    assert FLAG, 'IN_NUM is wrong'

    ckpt_step = "latest" if CONFIG['CKPT']['STEP'] is None else CONFIG['CKPT']['STEP']
    ckpt_path = f"{CONFIG['BASE']['SAVE_ROOT']}/{CONFIG['CKPT']['ID']}/ckpt/{type}_{ckpt_step}.pt"
    
    ckpt_dict = torch.load(ckpt_path, map_location=torch.device('cuda'))
    model.load_state_dict(ckpt_dict['model'], strict=False)
    optimizer.load_state_dict(ckpt_dict['optimizer'])

    return ckpt_dict['global_step']

def save_checkpoint(CONFIG, model, optimizer, type):
    
    ckpt_dict = {}
    ckpt_dict['global_step'] = CONFIG['BASE']['GLOBAL_STEP']
    ckpt_dict['model'] = model.state_dict()
    ckpt_dict['optimizer'] = optimizer.state_dict()

    torch.save(ckpt_dict, f"{CONFIG['BASE']['SAVE_ROOT_CKPT']}/{type}_{CONFIG['BASE']['GLOBAL_STEP']}.pt")
    torch.save(ckpt_dict, f"{CONFIG['BASE']['SAVE_ROOT_CKPT']}/{type}_latest.pt")
        
