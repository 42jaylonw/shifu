from shifu.models.module import Module

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from typing import List, Tuple


def linear_multilayer(
        input_dim: int,
        hidden_dims: [List, Tuple],
        activation: nn.Module = nn.ReLU(inplace=True)
):
    modules = []
    for h_dim in hidden_dims:
        modules.append(nn.Sequential(
            nn.Linear(input_dim, h_dim),
            activation
        ))
        input_dim = h_dim
    multilayer = nn.Sequential(*modules)
    return multilayer


def conv_encoder(
        in_channels: int,
        hidden_dims: [List, Tuple] = (16, 32, 64, 128, 256, 512),
        activation: nn.Module = nn.ReLU(inplace=True)
):
    modules = []
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels=h_dim,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=True),
                nn.BatchNorm2d(h_dim),
                activation)
        )
        in_channels = h_dim

    encoder = nn.Sequential(*modules)
    return encoder


def make_conv_layers(
        in_channels: int,
        hidden_dims: [List, Tuple] = (64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
        activation: nn.Module = nn.ReLU(inplace=True),
        batch_norm: bool = True
):
    layers = []
    for v in hidden_dims:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activation]
            else:
                layers += [conv2d, activation]
            in_channels = v
    encoder = nn.Sequential(*layers)
    return encoder


def conv_decoder(
        out_channels: int,
        hidden_dims: [List, Tuple] = (512, 256, 128, 64, 32, 16),
        activation: nn.Module = nn.ReLU(inplace=True)
):
    modules = []
    for i in range(len(hidden_dims) - 1):
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i],
                                   hidden_dims[i + 1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=True),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                activation)
        )
    # output layer
    modules.append(
        nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=True),
        ))

    decoder = nn.Sequential(*modules)
    return decoder


def re_param(mu, log_var, training):
    if training:
        std = log_var.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    else:  # return mean during inference
        return mu


def product_of_experts(x_mu, x_log_var, eps=1e-8):
    pr_size = (1, x_mu.shape[0], x_mu.shape[1])
    pr_mu = torch.zeros(pr_size)
    pr_log_var = torch.log(torch.ones(pr_size))
    pri_mu = torch.cat((pr_mu, x_mu.unsqueeze(0)), dim=0)
    pri_log_var = torch.cat((pr_log_var, x_log_var.unsqueeze(0)), dim=0)

    var = torch.exp(pri_log_var) + eps
    T = 1 / (var + eps)  # precision of i-th Gaussian expert at point x
    pd_mu = torch.sum(pri_mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1 / torch.sum(T, dim=0)
    pd_log_var = torch.log(pd_var + eps)

    return pd_mu, pd_log_var


def reconstruction_loss_func(task):
    if 'seg' in task:
        return torch.nn.CrossEntropyLoss()
    else:
        return torch.nn.MSELoss()


class Encoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: [List, Tuple] = (512, 256, 128, 64, 32),
            variational: bool = False,
            activation=nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = output_dim
        self.activation = activation
        self.variational = variational

        self.feature_extractor = self._build_feature_extractor()
        self.num_out_features = 2 * output_dim if variational else output_dim

        if variational:
            self.fc_mu = nn.Linear(self.get_middle_dim(), output_dim)
            self.fc_var = nn.Linear(self.get_middle_dim(), output_dim)
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.get_middle_dim(), 512),
                activation,
                nn.Linear(512, output_dim * 2 if variational else output_dim))

    def get_middle_dim(self):
        return self.hidden_dims[-1]

    def _build_feature_extractor(self):
        return linear_multilayer(self.input_dim, self.hidden_dims, self.activation)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        if self.variational:
            x_mu = self.fc_mu(x)
            x_log_var = self.fc_var(x)
            return x_mu, x_log_var
        else:
            x = self.fc(x)
            return x


class ConvEncoder(Encoder):
    def __init__(
            self,
            in_channels: int,
            latent_dim: int,
            hidden_dims: [List, Tuple] = (16, 32, 64, 128, 256, 512),
            variational: bool = False,
            activation=nn.ReLU(inplace=True),
    ):
        self.in_channels = in_channels
        super().__init__(-1, latent_dim, hidden_dims, variational, activation)

    def get_middle_dim(self):
        return self.hidden_dims[-1] * 4

    def _build_feature_extractor(self):
        return conv_encoder(self.in_channels, self.hidden_dims, self.activation)


class VGGEncoder(ConvEncoder):
    def __init__(
            self,
            in_channels: int,
            latent_dim: int,
            hidden_dims: [List, Tuple] = (64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
            variational: bool = False,
            activation=nn.ReLU(inplace=True),
    ):
        super().__init__(in_channels, latent_dim, hidden_dims, variational, activation)

    def get_middle_dim(self):
        return 512 * 4 * 4

    def _build_feature_extractor(self):
        return make_conv_layers(self.in_channels, self.hidden_dims, self.activation)


class Decoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: [List, Tuple] = (32, 64, 128, 256, 512),
            activation=nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.lin_decoder = linear_multilayer(input_dim, hidden_dims, activation)
        self.num_out_features = output_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.num_out_features)
        )

    def forward(self, x):
        x = self.lin_decoder(x)
        x = self.fc(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(
            self,
            out_channels: int,
            latent_dim: int,
            middle_dim: int,
            hidden_dims: [List, Tuple] = (512, 256, 128, 64, 32, 16),
            activation=nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.lin_decoder = nn.Sequential(
            nn.Linear(latent_dim, middle_dim),
            activation,
        )
        self.conv_decoder = conv_decoder(out_channels=out_channels, hidden_dims=hidden_dims, activation=activation)

    def forward(self, x):
        x = self.lin_decoder(x)
        x = x.view(-1, self.hidden_dims[0], 2, 2)
        x = self.conv_decoder(x)
        return x


class MultimodalAE(Module):
    def __init__(
            self,
            encoders: [Encoder, ConvEncoder],  # modal_name: Encoder
            decoders: [Decoder, ConvDecoder],  # modal_name: Decoder
            latent_dim: int,
            device='cuda:0'
    ):
        super().__init__(device=device)
        self.latent_dim = latent_dim
        self.device = self.device
        self.encoders = encoders
        self.decoders = decoders
        self.fusion_module = nn.Linear(len(self.encoders) * latent_dim, latent_dim)
        self.position_dict = {k: i for i, k in enumerate(self.encoders.keys())}

        self.fusion_module.to(device)
        for i, (name, encoder) in enumerate(self.encoders.items()):
            self.add_module(name, encoder)
            encoder.to(device)
        for name, decoder in self.decoders.items():
            self.add_module(name, decoder)
            decoder.to(device)

    def loss_func(self, pred, label):
        for name in label.keys():
            assert name in self.decoders.keys(), f"{name} must be in decoders"
        total_loss = 0
        loss_log = {}
        for k in label.keys():
            loss_func = reconstruction_loss_func(k)
            x_loss = loss_func(pred[k], label[k])
            total_loss += x_loss
            loss_log[k] = x_loss
        return total_loss, loss_log

    def cross_modal_encode(self, x_dict):
        modals_stack = torch.zeros(list(x_dict.values())[0].size(0),
                                   len(self.encoders) * self.latent_dim, device=self.device)

        for name, x in x_dict.items():
            encoder = self.encoders[name]
            z_x = encoder(x_dict[name])
            p = self.position_dict[name]
            modals_stack[:, p * self.latent_dim: (p + 1) * self.latent_dim] = z_x

        z = self.fusion_module(modals_stack)
        return z

    def cross_modal_decode(self, z):
        decode_dict = {}
        for name, decoder in self.decoders.items():
            decode_dict[name] = decoder(z)
        return decode_dict

    def forward(self, joint_dict):
        for name in joint_dict.keys():
            assert name in self.encoders.keys(), f"{name} must be in encoders"
        joint_encode = self.cross_modal_encode(joint_dict)
        decode_dict = self.cross_modal_decode(joint_encode)
        return decode_dict


if __name__ == '__main__':
    # encoder = ConvEncoder(in_channels=1, latent_dim=32)
    # decoder = ConvDecoder(out_channels=1, latent_dim=32, middle_dim=encoder.middle_dim)
    # z, m, lv = encoder(torch.randn(8, 1, 128, 128))
    # rec = decoder(z)
    # print({
    #     'z': z.shape,
    #     'rec': rec.shape
    # })
    x_rgb = torch.randn(8, 3, 128, 128)
    x_depth = torch.randn(8, 1, 128, 128)
    m_latent = 128
    b_size = 64
    device = 'cuda:0'

    vgg_rgb_encoder = VGGEncoder(in_channels=3, latent_dim=m_latent, variational=False)
    vgg_depth_encoder = VGGEncoder(in_channels=1, latent_dim=m_latent, variational=False)
    z_rgb = vgg_rgb_encoder(x_rgb)
    z_depth = vgg_depth_encoder(x_depth)

    # rgb_encoder = ConvEncoder(m_latent, 3)
    # rgb_decoder = ConvDecoder(m_latent, 3, rgb_encoder.middle_dim)
    # depth_encoder = ConvEncoder(m_latent, 1)
    # depth_decoder = ConvDecoder(m_latent, 1, depth_encoder.middle_dim)
    # mvae = MultimodalVAE(
    #     encoders={'rgb': rgb_encoder, 'depth': depth_encoder},
    #     decoders={'rgb': rgb_decoder, 'depth': depth_decoder},
    #     latent_dim=m_latent
    # )
    #
    # data_dict = {'rgb': torch.randn(b_size, 3, 128, 128, device=device),
    #              'depth': torch.randn(b_size, 1, 128, 128, device=device)}
    # label_dict = {'rgb': torch.randn(b_size, 3, 128, 128, device=device),
    #               'depth': torch.randn(b_size, 1, 128, 128, device=device)}
    #
    # recon_dict = mvae(data_dict)
    # loss, loss_log = mvae.loss_func(recon_dict, label_dict)
    print()
