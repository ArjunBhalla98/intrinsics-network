import sys, torch, torch.nn as nn, torch.nn.functional as F, pdb
from Decomposer import Decomposer
from Shader import Shader
from torch.autograd import Variable
from primitives import *


class Composer(nn.Module):
    def __init__(self, decomposer, shader):
        super(Composer, self).__init__()

        self.decomposer = decomposer
        self.shader = shader

    def forward(self, inp, mask):
        reflectance, depth, shape, lights = self.decomposer(inp, mask)
        # print(reflectance.size(), lights.size())
        shading = self.shader(shape, lights)
        shading_rep = shading.repeat(1, 3, 1, 1)
        # print(shading.size())
        reconstruction = reflectance * shading_rep
        return reconstruction, reflectance, depth, shape, lights, shading


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    # import models

    from PIL import Image, ImageOps
    import imageio
    import torchvision.transforms

    # img = Image.open("/phoenix/S3/ab2383/data/train_imgs/00277_0005.png")
    # mask = Image.open("/phoenix/S3/ab2383/data/TikTok_dataset/00277/masks/0005.png")
    img = Image.open(
        "/home/ab2383/intrinsics-network/dataset/output/motorbike_test/2_composite.png"
    )
    mask = Image.open(
        "/home/ab2383/intrinsics-network/dataset/output/motorbike_test/2_mask.png"
    )
    w, h = img.size  # why is this wxh??
    if w != h:
        new_size = max(w, h)
        side_padding = (h - w) // 2
        padding = (side_padding, 0, side_padding, 0)
        img = ImageOps.expand(img, padding)
        mask = ImageOps.expand(mask, padding)
        # print(img.size)

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize([256, 256]),
                torchvision.transforms.ToTensor(),
            ]
        )
        inp = Variable(transform(img).unsqueeze(0).type("torch.FloatTensor"))

        mask = Variable(
            transform(mask).expand(3, 256, 256).unsqueeze(0).type("torch.FloatTensor")
        )

        inp = inp.cuda()
        mask = mask.cuda()
    else:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize([256, 256]),
                torchvision.transforms.ToTensor(),
            ]
        )
        inp = Variable(transform(img).type("torch.FloatTensor"))
        # print(inp.size())
        inp = (inp[:3, :, :] * inp[3, :, :]).unsqueeze(0)
        mask = Variable(transform(mask).type("torch.FloatTensor"))
        mask = (mask[:3, :, :] * mask[3, :, :]).unsqueeze(0)
        inp = inp.cuda()
        mask = mask.cuda()
    # print(inp.size())
    # print(mask.size())

    # decomposer_path = "../logs/separated_decomp_0.01lr_0.1lights/model.t7"
    # shader_path = "../logs/separated_shader_0.01/model.t7"
    # decomposer = torch.load(decomposer_path)
    # shader = torch.load(shader_path)
    # composer = Composer(decomposer, shader).cuda()
    decomposer = Decomposer()
    # decomposer.load_state_dict(torch.load("saved/decomposer/state.t7"))
    shader = Shader()
    # shader.load_state_dict(torch.load("saved/shader/state.pth"))
    composer_path = "/home/ab2383/intrinsics-network/saved/composer/state.t7"
    composer = Composer(decomposer, shader)
    composer = composer.cuda()
    composer.load_state_dict(torch.load(composer_path))
    print(composer)
    # pdb.set_trace()
    # inp = Variable(torch.randn(5, 3, 256, 256).cuda())
    # mask = Variable(torch.randn(5, 3, 256, 256).cuda())

    out = composer.forward(inp, mask)

    print([i.size() for i in out])
    imageio.imsave(
        "recons_composer.png",
        out.squeeze().detach().numpy().transpose(1, 2, 0).clip(0, 1),
    )

    # pdb.set_trace()
