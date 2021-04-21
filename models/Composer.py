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


def process_img(img, mask):
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

    return inp, mask


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    # import models

    from PIL import Image, ImageOps
    import imageio
    import torchvision.transforms
    import os

    decomposer = Decomposer()
    decomposer.load_state_dict(torch.load("saved/decomposer/state.t7"))
    shader = Shader()
    shader.load_state_dict(torch.load("saved/shader/state.pth"))
    composer_path = "/home/ab2383/intrinsics-network/saved/composer/state.t7"
    composer = Composer(decomposer, shader)
    composer = composer.cuda()
    composer.load_state_dict(torch.load(composer_path))
    print("Composer Built")

    folders = ["00277", "00339", "00267"]
    base_path = "/phoenix/S3/ab2383/data/TikTok_dataset/"
    save_path = "/home/ab2383/intrinsics-network/"

    for folder in folders:
        imgs_path = base_path + folder + "/images/"
        masks_path = base_path + folder + "/masks/"
        for img in os.listdir(imgs_path):
            inp, mask = process_img(
                Image.open(imgs_path + img), Image.open(masks_path + img)
            )
            out_recons = composer.forward(inp, mask)[0]
            image = (
                out_recons.cpu()
                .detach()
                .numpy()
                .reshape(img.shape[1], 256, 256)
                .transpose(1, 2, 0)
                .clip(0, 1)
            )
            imageio.imsave(save_path + f"/{folder}/{img}")

    # out = composer.forward(inp, mask)
    # output_labels = ["reflectance", "depth", "normals", "lights"]
    # outputs = []

    # for i, img in enumerate(out):
    #     if i != 4:  # lights
    #         image = (
    #             img.cpu()
    #             .detach()
    #             .numpy()
    #             .reshape(img.shape[1], 256, 256)
    #             .transpose(1, 2, 0)
    #             .clip(0, 1)
    #         )
    #         imageio.imsave(str(i) + ".png", image)
    #     else:
    #         image = img
    #     outputs.append(image)

    # print([i.size() for i in out])
    print("Process Complete")
    # imageio.imsave(
    #     "recons_composer.png",
    #     out.squeeze().detach().numpy().transpose(1, 2, 0).clip(0, 1),
    # )

    # pdb.set_trace()
