import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
import wandb
import torchvision

class LossInterface(metaclass=abc.ABCMeta):
    """
    Base class for loss of GAN model. Exceptions will be raised when subclass is being 
    instantiated but abstract methods were not implemented. Concrete methods can be 
    overrided as well if needed.
    """

    def __init__(self, CONFIG):
        """
        When overrided, super call is required.
        """
        self.CONFIG = CONFIG
        self.start_time = time.time()
        self.loss_dict = {}
        self.loss_dict['L_G'] = .0
        self.loss_dict['L_D'] = .0
        self.loss_dict['valid_L_G'] = .0
        self.loss_dict['valid_L_D'] = .0
        

    def log_wandb(self):
        if self.CONFIG['WANDB']['TURN_ON']:
            wandb.log(self.loss_dict)

    def print_loss(self):
        """
        Print discriminator and generator loss and formatted elapsed time.
        """
        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f"steps: {self.CONFIG['BASE']['GLOBAL_STEP']:06} / {self.CONFIG['BASE']['MAX_STEP']}")
        print(f"lossD: {self.loss_dict['L_D']} | lossG: {self.loss_dict['L_G']}")
    
    def val_print_loss(self):
        """
        Print discriminator and generator loss and formatted elapsed time.
        """
        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f"steps: {self.CONFIG['BASE']['GLOBAL_STEP']:06} / {self.CONFIG['BASE']['MAX_STEP']}")
        print(f"val_lossD: {round(self.loss_dict['valid_L_D'], 4)} | val_lossG: {round(self.loss_dict['valid_L_G'], 4)}")


    @abc.abstractmethod
    def get_loss_G(self):
        """
        Caculate generator loss.
        Once loss values are saved in self.loss_dict, they can be uploaded on the 
        dashboard via wandb or printed in self.print_loss. self.print_loss can be 
        overrided as needed.
        """
        pass

    @abc.abstractmethod
    def get_loss_D(self):
        """
        Caculate discriminator loss.
        Once loss values are saved in self.loss_dict, they can be uploaded on the 
        dashboard via wandb or printed in self.print_loss. self.print_loss can be 
        overrided as needed.
        """
        pass

class Loss:
    """
    Provide various losses such as LPIPS, L1, L2, BCE and so on.
    """
    
    L1 = torch.nn.L1Loss().to("cuda")
    L2 = torch.nn.MSELoss().to("cuda")

    def get_id_loss(a, b):
        return (1 - torch.cosine_similarity(a, b, dim=1)).mean()

    @classmethod
    def get_lpips_loss(cls, a, b):
        if not hasattr(cls, 'lpips'):
            cls.lpips = LPIPS().eval().to("cuda")
        return cls.lpips(a, b)

    @classmethod
    def get_vgg_loss(cls, a, b):
        if not hasattr(cls, 'vgg'):
            cls.vgg = VGGLoss().eval().to("cuda")
        return cls.vgg(a, b)

    @classmethod
    def get_L1_loss(cls, a, b):   
        return cls.L1(a, b)

    @classmethod
    def get_L2_loss(cls, a, b):
        return cls.L2(a, b)

    def get_attr_loss(a, b, batch_size):
        L_attr = 0
        for i in range(len(a)):
            L_attr += torch.mean(torch.pow((a[i] - b[i]), 2).reshape(batch_size, -1), dim=1).mean()
        L_attr /= 2.0

        return L_attr

    def softplus_loss(logit, isReal=True):
        if isReal:
            return F.softplus(-logit).mean()
        else:
            return F.softplus(logit).mean()

    @classmethod
    def get_softplus_loss(cls, Di, label):
        L_adv = 0
        for di in Di:
            L_adv += cls.softplus_loss(di[0], label)
        return L_adv

    def hinge_loss(logit, positive=True):
        if positive:
            return torch.relu(1-logit).mean()
        else:
            return torch.relu(logit+1).mean()

    @classmethod
    def get_hinge_loss(cls, Di, label):
        L_adv = 0
        for di in Di:
            L_adv += cls.hinge_loss(di[0], label)
        return L_adv

    def get_BCE_loss(logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    def get_r1_reg(d_out, x_in):
        # zero-centered gradient penalty for real images
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg

    def get_adversarial_loss(logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()
        # Imagenet normalization for range [-1, 1]
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).cuda()
        self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).cuda()

    def _load_lpips_weights(self, ckpt_path='lib/ckpt/lpips_weights.ckpt'):
        own_state_dict = self.state_dict()
        state_dict = torch.load(ckpt_path, map_location=torch.device('cuda'))
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

    def forward(self, x, y):
        # x = (x - self.mu) / self.sigma
        # y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))

        return lpips_value


def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torchvision.models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)