from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import utils
from lib.model import ModelInterface
from lib.discriminators import ProjectedDiscriminator
from MyModel.loss import MyModelLoss
from MyModel.nets import MyGenerator

class MyModel(ModelInterface):
    def declare_networks(self):
        self.G = MyGenerator().cuda()
        self.D = ProjectedDiscriminator().cuda()
        
        self.set_networks_train_mode()

        # PACKAGES
        from id_extractor import IdExtractor
        self.IE = IdExtractor()

    def set_networks_train_mode(self):
        self.G.train()
        self.D.train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)
        
    def set_networks_eval_mode(self):
        self.G.eval()
        self.D.eval()

    def go_step(self):
        source, GT = self.load_next_batch(self.train_dataloader, self.train_iterator, 'train')
        
        self.train_dict["source"] = source
        self.train_dict["GT"] = GT

        # run G
        self.run_G(self.train_dict)

        # update G
        loss_G = self.loss_collector.get_loss_G(self.train_dict)
        utils.update_net(self.G, self.opt_G, loss_G, self.CONFIG['BASE']['USE_MULTI_GPU'])

        # run D
        self.run_D(self.train_dict)

        # update D
        loss_D = self.loss_collector.get_loss_D(self.train_dict)
        utils.update_net(self.D, self.opt_D, loss_D, self.CONFIG['BASE']['USE_MULTI_GPU'])
        
        # print images
        self.train_images = [
            self.train_dict["source"],
            self.train_dict["output"],
            self.train_dict["GT"],
            ]

    def run_G(self, run_dict):
        # with torch.no_grad():
        run_dict['output'] = self.G(run_dict['source'])
        g_pred_fake, feat_fake = self.D(run_dict["output"], None)
        feat_real = self.D.get_feature(run_dict["source"])

        run_dict['g_feat_fake'] = feat_fake
        run_dict['g_feat_real'] = feat_real
        run_dict["g_pred_fake"] = g_pred_fake

    def run_D(self, run_dict):
        d_pred_real, _  = self.D(run_dict['source'], None)
        d_pred_fake, _  = self.D(run_dict['output'].detach(), None)
        
        run_dict["d_pred_real"] = d_pred_real
        run_dict["d_pred_fake"] = d_pred_fake

    def do_validation(self):
        self.valid_images = []
        self.set_networks_eval_mode()

        self.loss_collector.loss_dict["valid_L_G"],  self.loss_collector.loss_dict["valid_L_D"] = 0., 0.
        pbar = tqdm(range(len(self.valid_dataloader)), desc='Run validate..')
        for _ in pbar:
            source, GT = self.load_next_batch(self.valid_dataloader, self.valid_iterator, 'valid')
            
            self.valid_dict["source"] = source
            self.valid_dict["GT"] = GT

            with torch.no_grad():
                self.run_G(self.valid_dict)
                self.run_D(self.valid_dict)
                self.loss_collector.get_loss_G(self.valid_dict, valid=True)
                self.loss_collector.get_loss_D(self.valid_dict, valid=True)
                            
            if len(self.valid_images) < 8 : utils.stack_image_grid([self.valid_dict["source"], self.valid_dict["output"], self.valid_dict["GT"]], self.valid_images)
            
        self.loss_collector.loss_dict["valid_L_G"] /= len(self.valid_dataloader)
        self.loss_collector.loss_dict["valid_L_D"] /= len(self.valid_dataloader)
        self.loss_collector.val_print_loss()
        
        self.valid_images = torch.cat(self.valid_images, dim=-1)

        self.set_networks_train_mode()
        
    def do_test(self):
        self.test_images = []
        self.set_networks_eval_mode()
        
        pbar = tqdm(range(len(self.test_dataloader)), desc='Run test...')
        for _ in pbar:
            source, GT = self.load_next_batch(self.test_dataloader, self.test_iterator, 'test')
            
            self.test_dict["source"] = source
            self.test_dict["GT"] = GT

            with torch.no_grad():
                self.run_G(self.test_dict)
                self.run_D(self.test_dict)

            utils.stack_image_grid([self.test_dict["source"], self.test_dict["output"], self.test_dict["GT"]], self.test_images)
        
        self.test_images = torch.cat(self.test_images, dim=-1)

        self.set_networks_train_mode()

    @property
    def loss_collector(self):
        return self._loss_collector


    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.CONFIG)        
