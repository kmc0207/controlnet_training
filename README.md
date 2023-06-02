# INVZ Code Template

```
python scripts/train.py {run_id}
```

## Release Note

### 22.11.16.
[x] Replace config class with dictionary  
[x] Make the code to copy "MyModel" to "train_result"  
[x] Combine "imgs_train" and "imgs_valid"  
[x] Rename {model, loss}_interface to {model, loss}  
[x] Divide "lib/dataset.py" to "MyModel/dataset.py" and "lib/dataset.py"  
[x] Remove "packages" directory  
[x] Add automatic numbering for training runs, starting from 000


### 22.11.19.
[x] Add test mode code  
[x] Automatically divide train/valid dataset from whole dataset.   
[x] When you want to continue finished train before, you only write training number in "config.yaml" file    
[x] Add train/eval mode select code  
[x] Fix Lpips checkpoint path  
[x] Modify minor things in "lib/nets.py" - By 1zong2

### 22.12.3

[x] lib/dataset.py line 25: train_dataset_dict >> test_dataset_dict  
[x] train_dataset_dict does not defined when DO_VALID is False  
[x] valid loss should be initialized to 0 when the do_validation function is called  
[x] set F.interpolate to bilinear mode  
[x] set beta of Adam optimizer [0, 0.999] as default  

### 22.12.25

[x] set_networks_test_mode > set_networks_eval_mode  
[x] add ID loss  
[x] sort data paths list  
[x] add an example of the package importing

### 23.01.13
[x] delete sampler of valid dataloader   
[x] sys.path.append(CONFIG['BASE']['PACKAGES_PATH']) @scripts/trian.py 

### 23.03.12
[x] error occurs when the Dataset returns only one variable.     
[x] replace config with jsonnet  
[x] add ddp port num to config file  
[x] fix a bug in the calculation of valid loss  

## Request 
[ ]

## Issues
### Error #1
AttributeError: 'EfficientNet' object has no attribute 'act1'   
If you face an error above, do this >> pip install timm==0.5.4     
ref: https://github.com/autonomousvision/projected_gan/issues/88    
