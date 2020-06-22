import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from utils.model_blocks import Mapping2Dto3D, Identity, get_template

dim_template_dict = {"SQUARE": 2, "SPHERE": 3}


class args():
    def __init__(self):
        self.SVR = False
        self.number_points = 4096
        self.number_points_eval = 4096
        self.num_layers = 2
        self.hidden_neurons = 512
        self.nb_primitives = 4
        self.template_type = 'SPHERE'
        self.bottleneck_size = 512
        self.activation = 'relu'
        self.device = 'cuda'
        self.remove_all_batchNorms = False
        self.dim_template = dim_template_dict[self.template_type]


class AtlasDec(nn.Module):

    def __init__(self):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        :param opt: 
        """
        super(AtlasDec, self).__init__()
        opt = args()
        self.opt = opt
        # self.device = opt.device

        # Define number of points per primitives
        self.nb_pts_in_primitive = opt.number_points // opt.nb_primitives
        # self.nb_pts_in_primitive_eval = opt.number_points_eval // opt.nb_primitives

        if opt.remove_all_batchNorms:
            torch.nn.BatchNorm1d = Identity
            print("Replacing all batchnorms by identities.")

        # Initialize templates
        self.template = [get_template(opt.template_type, device=opt.device) for i in range(0, opt.nb_primitives)]

        # Intialize deformation networks
        self.decoder = nn.ModuleList([Mapping2Dto3D(opt) for i in range(0, opt.nb_primitives)])

    def forward(self, latent_vector, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an opt.bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud of size : batch, 3, nb_pts_in_primitive * nb_ptimitives
        """
        # Sample points in the patches
        # input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive,
        #                                                     device=latent_vector.device)
        #                 for i in range(self.opt.nb_primitives)]
        input_points = [self.template[i].get_random_points(
            torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)),
            latent_vector.device) for i in range(self.opt.nb_primitives)]

        # Deform each patch
        output_points = torch.cat([self.decoder[i](input_points[i], latent_vector.unsqueeze(2)) for i in
                                   range(0, self.opt.nb_primitives)], dim=-1)

        # Return the deformed pointcloud
        return output_points.contiguous()  # (B, 3, nb_pts_in_primitive * nb_ptimitives)
