from __future__ import print_function

import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import utils
from dataset2 import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from pcpnet_patch_selective_multi import PCPNet, MSPCPNet
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import entr

def plot_pcd(ax, pcd,cc):
    ax.scatter(pcd[0,:], pcd[1, :], pcd[2,:], zdir='y', c=cc, s=0.5, cmap='RdBu', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--name', type=str, default='my_single_scale_normal_with_mask_res_models_0.5_0.5__multi2', help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./mask_res_models_0.5_0.5_multi2', help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='./logs', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name') #trainingset_whitenoise
    parser.add_argument('--testset', type=str, default='validationset_whitenoise.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='10', help='save model each n epochs')
    parser.add_argument('--refine', type=str, default='', help='refine model at this path')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=4000, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.03, 0.05, 0.01], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate') #0.0001
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--use_pca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')

    return parser.parse_args()

def train_pcpnet(opt):

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % (opt.name))
    # if os.path.exists(log_dirname) or os.path.exists(model_filename):
    #     response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (opt.name))
    #     if response == 'y':
    #         if os.path.exists(log_dirname):
    #             shutil.rmtree(os.path.join(opt.logdir, opt.name))
    #
    #     else:
    #         sys.exit()

    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
        else:
            raise ValueError('Unknown output: %s' % (o))

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')


    # create model
    if len(opt.patch_radius) == 1:
        pcpnet = PCPNet(
            num_points=opt.points_per_patch,
            output_dim=pred_dim,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)
    else:
        pcpnet = MSPCPNet(
            num_scales=len(opt.patch_radius),
            num_points=opt.points_per_patch,
            output_dim=pred_dim,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)

    if opt.refine != '':
        pcpnet.load_state_dict(torch.load(opt.refine))

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # create train and test dataset loaders
    train_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    if opt.training_order == 'random':
        train_datasampler = RandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            train_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    test_dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.testset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    if opt.training_order == 'random':
        test_datasampler = RandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        test_datasampler = SequentialShapeRandomPointcloudPatchSampler(
            test_dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    print('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass


    optimizer = optim.SGD(pcpnet.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=1e-4) #, weight_decay=1e-4
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1) # milestones in number of optimizer iterations
    pcpnet.cuda()

    train_num_batch = len(train_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    for epoch in range(1800, opt.nepoch):

        train_batchind = -1
        train_fraction_done = 0.0
        train_enum = enumerate(train_dataloader, 0)

        for train_batchind, data in train_enum:

            # update learning rate
            scheduler.step(epoch * train_num_batch + train_batchind)

            # set to training mode
            pcpnet.train()

            # get trainingset batch, convert to variables and upload to GPU
            points = data[0]
            target = data[1:3]
            s = data[3]

            s1 = s[:,0:500]
            s2 = s[:,500:1000]
            s3 = s[:,1000:1500]

            s_t1 = s1.clone()
            s_t2 = s2.clone()
            s_t3 = s3.clone()

            # s_t = s_t-v_repeat
            s1[s_t1<0.8]=1
            s1[s_t1>=0.8]=0
            s1 = s1.contiguous().view(-1, 1)[:,0].long().cuda()

            s2[s_t2<0.6]=1
            s2[s_t2>=0.6]=0
            s2 = s2.contiguous().view(-1, 1)[:,0].long().cuda()

            s3[s_t3<0.9]=1
            s3[s_t3>=0.9]=0
            s3 = s3.contiguous().view(-1, 1)[:,0].long().cuda()


            points = Variable(points)
            points = points.transpose(2, 1)
            points = points.cuda()

            s_v = s.cpu().numpy()

            target = tuple(Variable(t) for t in target)
            target = tuple(t.cuda() for t in target)

            # zero gradients
            optimizer.zero_grad()

            # forward pass
            x1, x2, x3, xx1, xx2, xx3,trans1,trans2,trans3, v,_,_,_ = pcpnet(points)
            pred = torch.cat([x1,x2,x3],-1)
            s = s.view(-1, 1)[:,0].long().cuda()
            xx1 = xx1.view(-1, 2)
            xx2 = xx2.view(-1, 2)
            xx3 = xx3.view(-1, 2)


            loss, loss1, loss2 = compute_loss(
                pred1=x1, pred2=x2, pred3=x3,
                target=target,
                mask_gt1=s1,mask1 = xx1,
                mask_gt2=s2, mask2=xx2,
                mask_gt3=s3, mask3=xx3,
                outputs=opt.outputs,
                output_pred_ind=output_pred_ind,
                output_target_ind=output_target_ind,
                output_loss_weight=v,
                patch_rot1=trans1 if opt.use_point_stn else None,
                patch_rot2=trans2 if opt.use_point_stn else None,
                patch_rot3=trans3 if opt.use_point_stn else None,
                normal_loss=opt.normal_loss)

            # backpropagate through entire network to compute gradients of loss w.r.t. parameters
            loss.backward()

            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind+1) / train_num_batch

            # print info and update log file
            print('[%s %d: %d/%d] %s loss: [%f, %f, %f]' % (opt.name, epoch, train_batchind, train_num_batch-1, green('train'), loss.item(),loss1.item(),loss2.item()))

        # save model, overwriting the old model
        if epoch % opt.saveinterval == 0 or epoch == opt.nepoch-1:
            torch.save(pcpnet.state_dict(), model_filename)

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        # if epoch % (5 * 10**math.floor(math.log10(max(2, epoch-1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch-1:
        if epoch % (5 * 10 ** math.floor( math.log10(max(2, epoch - 1)))) == 0 or epoch % 100 == 0 or epoch ==1930 or epoch==1950 or epoch==1970 or epoch==1990 or epoch == opt.nepoch - 1:
            torch.save(pcpnet.state_dict(), os.path.join(opt.outdir, '%s_model_%d.pth' % (opt.name, epoch)))


def compute_loss(pred1,pred2,pred3, target, mask_gt1, mask1, mask_gt2, mask2,mask_gt3, mask3, outputs, output_pred_ind,
                 output_target_ind, output_loss_weight, patch_rot1, patch_rot2, patch_rot3, normal_loss):

    loss = 0

    loss1_1 = torch.nn.functional.nll_loss(mask1, mask_gt1)
    loss1_2 = torch.nn.functional.nll_loss(mask2, mask_gt2)
    loss1_3 = torch.nn.functional.nll_loss(mask3, mask_gt3)

    loss1 =(loss1_1+loss1_2+loss1_3)/3
    for oi, o in enumerate(outputs):
        if o == 'unoriented_normals' or o == 'oriented_normals':
            o_pred1 = pred1[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            o_pred2 = pred2[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
            o_pred3 = pred3[:, output_pred_ind[oi]:output_pred_ind[oi]+3]

            o_target = target[output_target_ind[oi]]

            if patch_rot1 is not None:
                # transform predictions with inverse transform
                # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                o_pred1 = torch.bmm(o_pred1.unsqueeze(1), patch_rot1.transpose(2, 1)).squeeze(1)
                o_pred2 = torch.bmm(o_pred2.unsqueeze(1), patch_rot2.transpose(2, 1)).squeeze(1)
                o_pred3 = torch.bmm(o_pred3.unsqueeze(1), patch_rot3.transpose(2, 1)).squeeze(1)

            if o == 'unoriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss2_1= torch.min((o_pred1-o_target).pow(2).sum(1), (o_pred1+o_target).pow(2).sum(1))
                    loss2_2= torch.min((o_pred2-o_target).pow(2).sum(1), (o_pred2+o_target).pow(2).sum(1))
                    loss2_3= torch.min((o_pred3-o_target).pow(2).sum(1), (o_pred3+o_target).pow(2).sum(1))

                    tt = torch.mul(loss2_1, output_loss_weight[:,0])+\
                            torch.mul(loss2_2, output_loss_weight[:, 1])+\
                            torch.mul(loss2_3, output_loss_weight[:, 2])

                    loss += tt.mean()


                    # loss += torch.min((o_pred-o_target).pow(2).sum(1), (o_pred+o_target).pow(2).sum(1)).mean() * output_loss_weight[oi]
                elif normal_loss == 'ms_oneminuscos':
                    loss += (1-torch.abs(utils.cos_angle(o_pred, o_target))).pow(2).mean() * output_loss_weight[oi]
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            elif o == 'oriented_normals':
                if normal_loss == 'ms_euclidean':
                    loss += (o_pred-o_target).pow(2).sum(1).mean() * output_loss_weight[oi]
                elif normal_loss == 'ms_oneminuscos':
                    loss += (1-utils.cos_angle(o_pred, o_target)).pow(2).mean() * output_loss_weight[oi]
                else:
                    raise ValueError('Unsupported loss type: %s' % (normal_loss))
            else:
                raise ValueError('Unsupported output type: %s' % (o))

        elif o == 'max_curvature' or o == 'min_curvature':
            o_pred = pred1[:, output_pred_ind[oi]:output_pred_ind[oi]+1]
            o_target = target[output_target_ind[oi]]

            # Rectified mse loss: mean square of (pred - gt) / max(1, |gt|)
            normalized_diff = (o_pred - o_target) / torch.clamp(torch.abs(o_target), min=1)
            loss += normalized_diff.pow(2).mean() * output_loss_weight[oi]

        else:
            raise ValueError('Unsupported output type: %s' % (o))

    return 0.5*loss+0.5*loss1, loss, loss1

if __name__ == '__main__':
    train_opt = parse_arguments()
    train_pcpnet(train_opt)
