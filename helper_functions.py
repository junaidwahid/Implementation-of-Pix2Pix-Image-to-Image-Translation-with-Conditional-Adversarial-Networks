from torchvision.utils import make_grid
from torch import nn
import matplotlib.pyplot as plt
import torch


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

    
def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    fake_img = gen(condition)
    fake_disc = disc(fake_img, condition)
    recon_loss = recon_criterion(real, fake_img)
    adv_loss = adv_criterion(fake_disc, torch.ones_like(fake_disc))
    gen_loss = adv_loss + lambda_recon * recon_loss
    return gen_loss