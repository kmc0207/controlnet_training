import abc
import torch
from torch.utils.data import DataLoader
from MyModel.dataset import divide_datasets, MyDataset 
from lib import utils
import numpy as np
# from packages import Ranger
import glob

class ModelInterface(metaclass=abc.ABCMeta):
    """
    Base class for face GAN models. This base class can also be used 
    for neural network models with different purposes if some of concrete methods 
    are overrided appropriately. Exceptions will be raised when subclass is being 
    instantiated but abstract methods were not implemented. 
    """

    def __init__(self, CONFIG):
        """
        When overrided, super call is required.
        """
        self.G = None
        self.D = None
        
        self.CONFIG = CONFIG
        self.train_dict = {}
        self.valid_dict = {}
        self.test_dict = {}
        
        self.SetupModel()

    def SetupModel(self):
        
        self.CONFIG['BASE']['IS_MASTER'] = self.CONFIG['BASE']['GPU_ID'] == 0
        self.RandomGenerator = np.random.RandomState(42)
        self.declare_networks()
        self.set_optimizers()

        if self.CONFIG['BASE']['USE_MULTI_GPU']:
            self.set_multi_GPU()

        if self.CONFIG['CKPT']['TURN_ON']:
            self.load_checkpoint()

        divide_datasets(self, self.CONFIG)
        self.set_datasets()
        self.set_loss_collector()

        if self.CONFIG['BASE']['IS_MASTER']:
            print(f"Model {self.CONFIG['BASE']['MODEL_ID']} has successively created")
            
    def load_next_batch(self, dataloader, iterator, mode):
        """
        Load next batch of source image, target image, and boolean values that denote 
        if source and target are identical.
        """
        try:
            batch_data = next(iterator)
            batch_data = batch_data[0].cuda() if len(batch_data) == 1 else [data.cuda() for data in batch_data]

        except StopIteration:
            self.__setattr__(mode+'_iterator', iter(dataloader))
            batch_data = next(self.__getattribute__(mode+'_iterator'))
            batch_data = batch_data[0].cuda() if len(batch_data) == 1 else [data.cuda() for data in batch_data]

        return batch_data

    def set_datasets(self):
        """
        Initialize dataset using the dataset paths specified in the command line arguments.
        """
        if self.CONFIG['BASE']['DO_TRAIN']:
            self.train_dataset = MyDataset(self.CONFIG, 'TRAIN', self.train_dataset_dict)
            self.set_train_data_iterator()
            
        if self.CONFIG['BASE']['DO_VALID']:
            self.valid_dataset = MyDataset(self.CONFIG, 'VALID', self.valid_dataset_dict)
            self.set_valid_data_iterator()
            
        if self.CONFIG['BASE']['DO_TEST']:
            self.test_dataset = MyDataset(self.CONFIG, 'TEST', self.test_dataset_dict)
            self.set_test_data_iterator()
            
    def set_train_data_iterator(self):
        """
        Construct sampler according to number of GPUs it is utilizing.
        Using self.dataset and sampler, construct dataloader.
        Store Iterator from dataloader as a member variable.
        """
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.CONFIG['BASE']['USE_MULTI_GPU'] else None
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.CONFIG['BASE']['BATCH_PER_GPU'], pin_memory=True, sampler=sampler, num_workers=8, drop_last=True)
        self.train_iterator = iter(self.train_dataloader)

    def set_valid_data_iterator(self):
        """
        Predefine test images only if args.valid_dataset_root is specified.
        These images are anchored for checking the improvement of the model.
        """
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=1, pin_memory=True, drop_last=True)
        self.valid_iterator = iter(self.valid_dataloader)
        
    def set_test_data_iterator(self):
        """
        Predefine test images only if args.test_dataset_root is specified.
        These images are anchored for checking the improvement of the model.
        """
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, pin_memory=True, drop_last=True)
        self.test_iterator = iter(self.test_dataloader)

    @abc.abstractmethod
    def declare_networks(self):
        """
        Construct networks, send it to GPU, and set training mode.
        Networks should be assigned to member variables.

        eg. self.D = Discriminator(input_nc=3).cuda(self.gpu).train() 
        """
        pass

    def set_multi_GPU(self):
        utils.setup_ddp(self.CONFIG['BASE']['GPU_ID'], self.CONFIG['BASE']['GPU_NUM'], self.CONFIG['BASE']['PORT'])

        # Data parallelism is required to use multi-GPU
        self.G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[self.CONFIG['BASE']['GPU_ID']], broadcast_buffers=False, find_unused_parameters=True).module
        if self.D : self.D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[self.CONFIG['BASE']['GPU_ID']]).module

    def save_checkpoint(self):
        """
        Save model and optimizer parameters.
        """
        utils.save_checkpoint(self.CONFIG, self.G, self.opt_G, type='G')
        if self.D : utils.save_checkpoint(self.CONFIG, self.D, self.opt_D, type='D')
        
        if self.CONFIG['BASE']['IS_MASTER']:
            print(f"\nCheckpoints are succesively saved in {self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['BASE']['RUN_ID']}/ckpt/\n")
    
    def load_checkpoint(self):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """
        self.CONFIG['BASE']['GLOBAL_STEP'] = \
        utils.load_checkpoint(self.CONFIG, self.G, self.opt_G, type="G")
        if self.D : utils.load_checkpoint(self.CONFIG, self.D, self.opt_D, type="D")

        if self.CONFIG['BASE']['IS_MASTER']:
            print(f"Pretrained parameters are succesively loaded from {self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['CKPT']['ID']}/ckpt/")

    def set_optimizers(self):
        if self.CONFIG['OPTIMIZER']['TYPE'] == "Adam":
            self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.CONFIG['OPTIMIZER']['LR_G'], betas=self.CONFIG['OPTIMIZER']['BETA'])
            if self.D : self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.CONFIG['OPTIMIZER']['LR_D'], betas=self.CONFIG['OPTIMIZER']['BETA'])
            
        if self.CONFIG['OPTIMIZER']['TYPE'] == "Ranger":
            self.opt_G = Ranger(self.G.parameters(), lr=self.CONFIG['OPTIMIZER']['LR_G'], betas=self.CONFIG['OPTIMIZER']['BETA'])
            if self.D : self.opt_D = Ranger(self.D.parameters(), lr=self.CONFIG['OPTIMIZER']['LR_D'], betas=self.CONFIG['OPTIMIZER']['BETA'])

    @abc.abstractmethod
    def set_loss_collector(self):
        """
        Set self.loss_collector as an implementation of lib.loss.LossInterface.
        """
        pass

    @property
    @abc.abstractmethod
    def loss_collector(self):
        """
        loss_collector should be an implementation of lib.loss.LossInterface.
        This property should be assigned in self.set_loss_collector.
        """
        pass

    @abc.abstractmethod
    def go_step(self):
        """
        Implement a single iteration of training. This will be called repeatedly in a loop. 
        This method should return list of images that was created during training.
        Returned images are passed to self.save_image and self.save_image is called in the 
        training loop preiodically.
        """
        pass

    @abc.abstractmethod
    def do_validation(self):
        """
        Test the model using a predefined valid set.
        This method includes util.save_image and returns nothing.
        """
        pass