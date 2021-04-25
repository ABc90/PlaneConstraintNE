from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import utils

class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim*self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.identity(self.dim, 'float32')).clone()).view(1, self.dim*self.dim).repeat(batchsize, 1)

        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x

class QSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales, 1))
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = Variable(torch.FloatTensor([1, 0, 0, 0]))
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden

        # convert quaternion to rotation matrix
        if x.is_cuda:
            trans = Variable(torch.cuda.FloatTensor(batchsize, 3, 3))
        else:
            trans = Variable(torch.FloatTensor(batchsize, 3, 3))
        x = utils.batch_quat_to_rotmat(x, trans)

        return x


class AttenLayer(nn.Module):
    def __init__(self, in_channels=64, mode='gaussian', inter_channels=None, dimension=1,bn_layer=False):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
             args:
                 in_channels: original channel size (1024 in the paper)
                 inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
                 mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
                 bn_layer: whether to add batch norm
                 https://github.com/tea1528/Non-Local-NN-Pytorch/blob/master/models/non_local.py
             """
        super(AttenLayer, self).__init__()

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = torch.nn.Conv3d.Conv3d
            max_pool_layer = torch.nn.Conv3d.MaxPool3d(kernel_size=(1, 2, 2))
            bn = torch.nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = torch.nn.Conv2d
            max_pool_layer = torch.nn.MaxPool2d(kernel_size=(2, 2))
            bn = torch.nn.BatchNorm2d
        else:
            conv_nd = torch.nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = torch.nn.BatchNorm1d

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)


        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = torch.nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = torch.nn.functional.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


class MultiscaleMask(nn.Module):
    def __init__(self,R = [0.03,0.05,0.07],num_points=500):
        pass

class PointNetfeat(nn.Module):
    def __init__(self, num_scales=1, num_points=500, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=True, point_tuple=1):
        super(PointNetfeat, self).__init__()
        self.num_points = num_points
        self.num_scales = num_scales
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals
        self.point_tuple = point_tuple

        if self.use_point_stn:
            # self.stn1 = STN(num_scales=self.num_scales, num_points=num_points, dim=3, sym_op=self.sym_op)
            self.stn1 = QSTN(num_scales=self.num_scales, num_points=num_points*self.point_tuple, dim=3, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        # self.atten1 = AttenLayer(in_channels=64)
        # self.atten2 = AttenLayer(in_channels=128)
        # self.atten3 = AttenLayer(in_channels=1024)


        self.conv0a = torch.nn.Conv1d(3*self.point_tuple, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)
        self.bn0a = nn.BatchNorm1d(64)
        self.bn0b = nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1_1 = torch.nn.Conv1d(2048, 128, 1)
        self.conv1_2 = torch.nn.Conv1d(2048, 64, 1)
        self.conv1_3 = torch.nn.Conv1d(64, 1, 1)
        self.bn1_1 = nn.BatchNorm1d(128)
        self.bn2_1 = nn.BatchNorm1d(64)

        if self.num_scales > 1:
            self.conv4 = torch.nn.Conv1d(1024, 1024*self.num_scales, 1)
            self.bn4 = nn.BatchNorm1d(1024*self.num_scales)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.mp1 = None
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

    def forward(self, x):

        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3*self.point_tuple, -1)
        else:
            trans = None

        # mlp (64,64)
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        # mlp (64,128,1024)
        x = F.relu(self.bn1(self.conv1(x)))
        # x = self.atten1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        # x = self.atten2(x)

        x = self.bn3(self.conv3(x))
        # x = self.atten3(x)


        # mlp (1024,1024*num_scales)
        if self.num_scales > 1:
            x = self.bn4(self.conv4(F.relu(x)))

        if self.get_pointfvals:
            pointfvals = x
        else:
            pointfvals = None # so the intermediate result can be forgotten if it is not needed


        # symmetric max operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024*self.num_scales**2, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024*self.num_scales**2, 1))
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*1024:(s+1)*self.num_scales*1024, :] = torch.sum(x[:, :, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = x_scales

        x = x.view(-1, 1024*self.num_scales**2)
        xxx = x.view(-1, 1024, 1).repeat(1, 1, 500)
        return torch.cat([xxx, pointfvals], 1), trans, trans2, pointfvals


class PCPNet(nn.Module):
    def __init__(self, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=True, point_tuple=1):
        super(PCPNet, self).__init__()
        self.num_points = num_points

        self.feat = PointNetfeat(
            num_points=num_points,
            num_scales=1,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)

        self.conv1_1 = torch.nn.Conv1d(2048, 256, 1)
        self.conv1_2 = torch.nn.Conv1d(256, 64, 1)
        self.conv1_3 = torch.nn.Conv1d(64, 64, 1)
        self.conv1_4 = torch.nn.Conv1d(64, 2, 1)

        self.bn1_1 = nn.BatchNorm1d(256)
        self.bn1_2 = nn.BatchNorm1d(64)
        self.bn1_3 = nn.BatchNorm1d(64)

        self.conv2_1 = torch.nn.Conv1d(2048, 1024, 1)
        self.conv2_2 = torch.nn.Conv1d(1024, 1024, 1)

        # self.conv3_1 = torch.nn.Conv1d(1024+2048, 1024, 1)
        # self.conv3_2 = torch.nn.Conv1d(1024, 1024, 1)
        #
        # self.bn3_1 = nn.BatchNorm1d(1024)
        # self.bn3_2 = nn.BatchNorm1d(1024)


        self.bn2_1 = nn.BatchNorm1d(1024)
        self.bn2_2 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout(p=0.3)
        self.do2 = nn.Dropout(p=0.3)
    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        xx = x
        x_o = x
        # calculate the mask
        xx = F.relu(self.bn1_1(self.conv1_1(xx)))
        xx = F.relu(self.bn1_2(self.conv1_2(xx)))
        xx = F.relu(self.bn1_3(self.conv1_3(xx)))
        xx = self.conv1_4(xx)
        xx = xx.transpose(2,1).contiguous()
        xx = xx.view(-1, 2)
        xx = F.log_softmax(xx, dim=-1)
        xxx = F.softmax(xx, dim=-1)
        xx = xx.view(-1, 500, 2)
        xxx = xxx.view(-1, 500, 2)
        xxx = 1/(1+torch.exp(-100*(xxx-0.5)))

        # x_mask1 = torch.mul(x, xxx[:,:,1].view(-1, 1, 500).repeat(1, 2048, 1))
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        # x_mask2 = torch.mul(x, xxx[:,:,1].view(-1, 1, 500).repeat(1, 1024, 1))
        x = F.relu(self.bn2_2(self.conv2_2(x)))

        x_mask3 = torch.mul(x, xxx[:,:,1].view(-1, 1, 500).repeat(1, 1024, 1))
        x = torch.sum(x, 2, keepdim=True)

        #x = torch.cat([x+x_mask3, x_o],dim=1)

        # x = torch.sum(x, 2, keepdim=True)
        # x = F.relu(self.bn3_1(self.conv3_1(x)))
        # x = F.relu(self.bn3_2(self.conv3_2(x)))

        x = x.view(-1, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.do1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do2(x)
        x = self.fc3(x)

        return x, xx, xxx, trans, trans2, pointfvals

class MSPCPNet(nn.Module):
    def __init__(self, num_scales=2, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max', get_pointfvals=True, point_tuple=1):
        super(MSPCPNet, self).__init__()
        self.num_points = num_points

        self.feat1 = PCPNet(
            num_points=num_points,
            output_dim=output_dim,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)

        self.feat2 = PCPNet(
            num_points=num_points,
            output_dim=output_dim,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)

        self.feat3 = PCPNet(
            num_points=num_points,
            output_dim=output_dim,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)

        self.fc0_1 = nn.Linear(1024*3, 1024)
        self.fc1_1 = nn.Linear(1024, 512)
        self.fc2_1 = nn.Linear(512, 256)
        self.fc3_1 = nn.Linear(256, 3)
        self.bn0_1 = nn.BatchNorm1d(1024)
        self.bn1_1 = nn.BatchNorm1d(512)
        self.bn2_1 = nn.BatchNorm1d(256)

        # self.fc0_2 = nn.Linear(1024, 1024)
        # self.fc1_2 = nn.Linear(1024, 512)
        # self.fc2_2 = nn.Linear(512, 256)
        # self.fc3_2 = nn.Linear(256, 1)
        # self.bn0_2 = nn.BatchNorm1d(1024)
        # self.bn1_2 = nn.BatchNorm1d(512)
        # self.bn2_2 = nn.BatchNorm1d(256)
        #
        # self.fc0_3 = nn.Linear(1024, 1024)
        # self.fc1_3 = nn.Linear(1024, 512)
        # self.fc2_3 = nn.Linear(512, 256)
        # self.fc3_3 = nn.Linear(256, 1)
        # self.bn0_3 = nn.BatchNorm1d(1024)
        # self.bn1_3 = nn.BatchNorm1d(512)
        # self.bn2_3 = nn.BatchNorm1d(256)
        self.sm1 = torch.nn.Softmax()
        # self.sm2 = torch.nn.Sigmoid()
        # self.sm3 = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = x[...,0:500]
        x2 = x[...,500:1000]
        x3 = x[...,1000:1500]
        x1, xx1, xxx1, trans1, trans2_1, pointfvals1 = self.feat1(x1)
        x2, xx2, xxx2, trans2, trans2_2, pointfvals2 = self.feat2(x2)
        x3, xx3, xxx3, trans3, trans2_3, pointfvals3 = self.feat3(x3)

        pointfvals1 = torch.sum(pointfvals1, 2, keepdim=True)
        pointfvals1 = pointfvals1.view(-1, 1024)
        pointfvals2 = torch.sum(pointfvals2, 2, keepdim=True)
        pointfvals2 = pointfvals2.view(-1, 1024)
        pointfvals3 = torch.sum(pointfvals3, 2, keepdim=True)
        pointfvals3 = pointfvals3.view(-1, 1024)
        f = torch.cat([pointfvals1,pointfvals2,pointfvals3],-1)

        v1 = F.relu(self.bn0_1(self.fc0_1(f)))
        v1 = F.relu(self.bn1_1(self.fc1_1(v1)))
        v1 = F.relu(self.bn2_1(self.fc2_1(v1)))
        v1 = self.fc3_1(v1)
        v1 = self.sm1(v1)


        # v2= F.relu(self.bn0_2(self.fc0_2(pointfvals2)))
        # v2= F.relu(self.bn1_2(self.fc1_2(v2)))
        # v2 = F.relu(self.bn2_2(self.fc2_2(v2)))
        # v2 = self.fc3_1(v2)
        # v2 =  self.sm2(v2)
        #
        #
        # v3 = F.relu(self.bn0_3(self.fc0_3(pointfvals3)))
        # v3 = F.relu(self.bn1_3(self.fc1_3(v3)))
        # v3 = F.relu(self.bn2_3(self.fc2_3(v3)))
        # v3 = self.fc3_3(v3)
        # v3 = self.sm3(v3)

        return x1, x2, x3, xx1, xx2, xx3, trans1,trans2,trans3 ,v1, xxx1, xxx2, xxx3
