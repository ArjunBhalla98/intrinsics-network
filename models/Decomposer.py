import sys, torch, torch.nn as nn, torch.nn.functional as F
from Shader import Shader
from torch.autograd import Variable
from primitives import *

"""
Predicts reflectance, shape, and lighting conditions given an image

Reflectance and shape are 3-channel images of the 
same dimensionality as input (expects 256x256). 
Lights have dimensionality lights_dim. By default,
they are represented as [x, y, z, energy].
"""


class Decomposer(nn.Module):
    def __init__(self, lights_dim=4):
        super(Decomposer, self).__init__()

        #######################
        #### shape encoder ####
        #######################
        ## there is a single shared convolutional encoder
        ## for all intrinsic images
        channels = [3, 16, 32, 64, 128, 256, 256]
        kernel_size = 3
        padding = 1
        ## stride of 1 on first layer and 2 everywhere else
        stride_fn = lambda ind: 1 if ind == 0 else 2
        sys.stdout.write("<Decomposer> Building Encoder")
        self.encoder = build_encoder(channels, kernel_size, padding, stride_fn)

        #######################
        #### shape decoder ####
        #######################
        ## link encoder and decoder
        channels.append(channels[-1])
        ## reverse channel order for decoder
        channels = list(reversed(channels))
        stride_fn = lambda ind: 1
        sys.stdout.write("<Decomposer> Building Decoder")
        ## separate reflectance and normals decoders.
        ## mult = 2 because the skip layer concatenates
        ## an encoder layer with the decoder layer,
        ## so the number of input channels in each layer is doubled.
        self.decoder_reflectance = build_encoder(
            channels, kernel_size, padding, stride_fn, mult=2
        )
        self.decoder_normals = build_encoder(
            channels, kernel_size, padding, stride_fn, mult=2
        )
        channels[-1] = 1
        self.decoder_depth = build_encoder(
            channels, kernel_size, padding, stride_fn, mult=2
        )
        self.upsampler = nn.UpsamplingNearest2d(scale_factor=2)

        #### lights encoder
        lights_channels = [256, 128, 64]
        stride_fn = lambda ind: 2
        sys.stdout.write("<Decomposer> Lights Encoder  ")
        self.decoder_lights = build_encoder(
            lights_channels, kernel_size, padding, stride_fn
        )
        lights_encoded_dim = 2

        self.lights_fc1 = nn.Linear(lights_channels[-1] * (lights_encoded_dim ** 2), 32)
        self.lights_fc2 = nn.Linear(32, lights_dim)

    def __decode(self, decoder, encoded, inp):
        x = inp
        for ind in range(len(decoder) - 1):
            x = decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            x = join(1)(x, encoded[-(ind + 1)])
            x = F.leaky_relu(x)

        x = decoder[-1](x)
        return x

    def forward(self, inp, mask):
        ## shared encoder
        x = inp
        encoded = []
        for ind in range(len(self.encoder)):
            x = self.encoder[ind](x)
            x = F.leaky_relu(x)
            encoded.append(x)

        ## decode lights
        lights = x
        for ind in range(len(self.decoder_lights)):
            lights = self.decoder_lights[ind](lights)
            lights = F.leaky_relu(lights)
        lights = lights.view(lights.size(0), -1)
        lights = F.leaky_relu(self.lights_fc1(lights))
        lights = self.lights_fc2(lights)

        ## separate decoders
        reflectance = self.__decode(self.decoder_reflectance, encoded, x)
        normals = self.__decode(self.decoder_normals, encoded, x)
        depth = self.__decode(self.decoder_depth, encoded, x)

        ## R, G in [-1,1]
        rg = torch.clamp(normals[:, :-1, :, :], -1, 1)
        ## B in [0,1]
        b = torch.clamp(normals[:, -1, :, :].unsqueeze(1), 0, 1)
        clamped = torch.cat((rg, b), 1)
        ## normals are unit vector
        normed = normalize(clamped)

        ## turn float mask into bool array
        print(mask.shape)
        mask = mask < 0.25
        ## set background pixels to 0 so
        ## we don't count them in error
        reflectance[mask] = 0
        normed[mask] = 0
        # depth[mask[:, 0]] = 0
        depth[mask[:, 0].unsqueeze(1)] = 0

        return reflectance, depth, normed, lights


if __name__ == "__main__":
    from PIL import Image, ImageOps
    import imageio
    import torchvision.transforms

    # img = Image.open("/phoenix/S3/ab2383/data/train_imgs/00277_0005.png")
    # mask = Image.open("/phoenix/S3/ab2383/data/TikTok_dataset/00277/masks/0005.png")
    img = Image.open(
        "/home/ab2383/intrinsics-network/dataset/output/motorbike_test/1360_composite.png"
    )
    mask = Image.open(
        "/home/ab2383/intrinsics-network/dataset/output/motorbike_test/1360_mask.png"
    )
    w, h = img.size  # why is this wxh??
    if w != h:
        new_size = max(w, h)
        side_padding = (h - w) // 2
        padding = (side_padding, 0, side_padding, 0)
        img = ImageOps.expand(img, padding)
        mask = ImageOps.expand(mask, padding)
        print(img.size)

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
    else:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize([256, 256]),
                torchvision.transforms.ToTensor(),
            ]
        )
        inp = Variable(transform(img).type("torch.FloatTensor"))
        inp = (inp[:, :, :3] * inp[:, :, 3]).unsqueeze(0)
        mask = Variable(transform(mask).type("torch.FloatTensor"))
        mask = (mask[:, :, :3] * mask[:, :, 3]).unsqueeze(0)
    print(inp.size())
    print(mask.size())

    decomposer = Decomposer()
    decomposer.load_state_dict(torch.load("saved/decomposer/state.t7"))
    shader = Shader()
    shader.load_state_dict(torch.load("saved/shader/state.pth"))
    out = decomposer.forward(inp, mask)

    output_labels = ["reflectance", "depth", "normals", "lights"]
    outputs = []

    for i, img in enumerate(out):
        if i != 3:
            image = (
                img.cpu()
                .detach()
                .numpy()
                .reshape(img.shape[1], 256, 256)
                .transpose(1, 2, 0)
                .clip(0, 1)
            )
            imageio.imsave(output_labels[i] + ".png", image)
        else:
            image = img
        outputs.append(image)

    shading = shader(out[2], out[3])
    shading_rep = shading.repeat(1, 3, 1, 1)
    print(shading_rep.size(), out[0].size())
    imageio.imsave(
        "shading.png",
        shading_rep.squeeze().detach().numpy().transpose(1, 2, 0).clip(0, 1),
    )
    recons = out[0] * shading_rep
    imageio.imsave(
        "reconstructed.png",
        recons.squeeze().detach().numpy().transpose(1, 2, 0).clip(0, 1),
    )

    print([i.size() for i in out])
