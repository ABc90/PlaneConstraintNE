from __future__ import print_function
import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset2 import PointcloudPatchDataset, SequentialPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from pcpnet_patch_selective_multi import PCPNet, MSPCPNet
# from mayavi import mlab
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils


def plot_pcd(ax, pcd,cc):
    ax.scatter(pcd[0,:], pcd[1, :], pcd[2,:], zdir='y', c=cc, s=0.5, cmap='RdBu', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./results_1', help='output folder (estimated point cloud properties)')
    parser.add_argument('--dataset', type=str, default='testset_no_noise.txt', help='shape set file name')
    parser.add_argument('--modeldir', type=str, default='./mask_res_models_0.5_0.5__multi2', help='model folder')
    parser.add_argument('--models', type=str, default='my_single_scale_normal_with_mask_res_models_0.5_0.5_multi2', help='names of trained models, can evaluate multiple models')
    parser.add_argument('--modelpostfix', type=str, default='_model_2100.pth', help='model file postfix')
    parser.add_argument('--parmpostfix', type=str, default='_params.pth', help='parameter file postfix')
    parser.add_argument('--sparse_patches', type=int, default=False, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=0, help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')

    return parser.parse_args()

def eval_pcpnet(opt):

    opt.models = opt.models.split()

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    for model_name in opt.models:

        print("Random Seed: %d" % (opt.seed))
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

        model_filename = os.path.join(opt.modeldir, model_name+opt.modelpostfix)
        param_filename = os.path.join(opt.modeldir, model_name+opt.parmpostfix)

        # load model and training parameters
        trainopt = torch.load(param_filename)
        trainopt.batchSize = 16
        if opt.batchSize == 0:
            model_batchSize = trainopt.batchSize
        else:
            model_batchSize = opt.batchSize

        # get indices in targets and predictions corresponding to each output
        pred_dim = 0
        output_pred_ind = []
        for o in trainopt.outputs:
            if o == 'unoriented_normals' or o == 'oriented_normals':
                output_pred_ind.append(pred_dim)
                pred_dim += 3
            elif o == 'max_curvature' or o == 'min_curvature':
                output_pred_ind.append(pred_dim)
                pred_dim += 1
            else:
                raise ValueError('Unknown output: %s' % (o))

        dataset = PointcloudPatchDataset(
            root=opt.indir, shape_list_filename=opt.dataset,
            patch_radius=trainopt.patch_radius,
            points_per_patch=trainopt.points_per_patch,
            patch_features=[],
            seed=opt.seed,
            use_pca=trainopt.use_pca,
            center=trainopt.patch_center,
            point_tuple=trainopt.point_tuple,
            sparse_patches=opt.sparse_patches,
            cache_capacity=opt.cache_capacity)

        if opt.sampling == 'full':
            datasampler = SequentialPointcloudPatchSampler(dataset)
        elif opt.sampling == 'sequential_shapes_random_patches':
            datasampler = SequentialShapeRandomPointcloudPatchSampler(
                dataset,
                patches_per_shape=opt.patches_per_shape,
                seed=opt.seed,
                sequential_shapes=True,
                identical_epochs=False)
        else:
            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=datasampler,
            batch_size=model_batchSize,
            num_workers=int(opt.workers))

        if len(trainopt.patch_radius) == 1:
            regressor = PCPNet(
                num_points=trainopt.points_per_patch,
                output_dim=pred_dim,
                use_point_stn=trainopt.use_point_stn,
                use_feat_stn=trainopt.use_feat_stn,
                sym_op=trainopt.sym_op,
                point_tuple=trainopt.point_tuple)
        else:
            regressor = MSPCPNet(
                num_scales=len(trainopt.patch_radius),
                num_points=trainopt.points_per_patch,
                output_dim=pred_dim,
                use_point_stn=trainopt.use_point_stn,
                use_feat_stn=trainopt.use_feat_stn,
                sym_op=trainopt.sym_op,
                point_tuple=trainopt.point_tuple)

        regressor.load_state_dict(torch.load(model_filename))
        regressor.cuda()
        regressor.eval()

        shape_ind = 0
        shape_patch_offset = 0
        if opt.sampling == 'full':
            shape_patch_count = dataset.shape_patch_count[shape_ind]
        elif opt.sampling == 'sequential_shapes_random_patches':
            shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
        else:
            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
        shape_properties1 = torch.FloatTensor(shape_patch_count, pred_dim).zero_()
        shape_properties2 = torch.FloatTensor(shape_patch_count, pred_dim).zero_()
        shape_properties3 = torch.FloatTensor(shape_patch_count, pred_dim).zero_()
        shape_max_v = torch.FloatTensor(shape_patch_count, 3).zero_()
        # append model name to output directory and create directory if necessary
        model_outdir = os.path.join(opt.outdir, model_name)
        if not os.path.exists(model_outdir):
            os.makedirs(model_outdir)

        num_batch = len(dataloader)
        batch_enum = enumerate(dataloader, 0)
        for batchind, data in batch_enum:

            # get batch, convert to variables and upload to GPU
            # points, data_trans = data


            points = data[0]
            target = data[1:3]
            # s = data[3]
            # hist = data[4]
            # curvs = data[5]

            points = data[0]
            data_trans = data[1]
            # target = data[2]
            # s = data[3]
            # s_t = s.clone()
            # s[s_t < 0.6] = 1
            # s[s_t >= 0.6] = 0


            # points = Variable(points, volatile=True)
            # points = points.transpose(2, 1)
            # points = points.cuda()

            points = Variable(points)
            points = points.transpose(2, 1)
            points = points.cuda()
            data_trans = data_trans.cuda()


            x1, x2, x3, xx1, xx2, xx3,trans1,trans2,trans3,v,xxx1,xxx2,xxx3 = regressor(points)
            # pred = torch.cat([x1.view(-1,3,1),x2.view(-1,3,1),x3.view(-1,3,1)],-1)
            pred1 =x1
            pred2 = x2
            pred3 = x3

            trans = trans1
            # trans = torch.cat([trans1.view(-1,3,3,1),trans2.view(-1,3,3,1),trans3.view(-1,3,3,1)],-1)
            # print(v)


            # points_v = points.cpu().numpy()
            # xxx_v = xxx3.detach().cpu().numpy()
            #
            # fig = plt.figure(figsize=(4, 4))
            # cont=1
            # for ii in range(0, 16, 1):
            #     ax = fig.add_subplot(4, 4, int(cont), projection='3d')
            #     plot_pcd(ax, points_v[ii,:,1000:1500], xxx_v[ii,:,0])
            #     cont +=1
            #     # ax.set_title('Output')
            # plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
            # plt.show()



            # don't need to work with autograd variables anymore
            pred1 = pred1.data
            pred2 = pred2.data
            pred3 = pred3.data
            v = v.data
            if trans is not None:
                trans1 = trans1.data
                trans2 = trans2.data
                trans3 = trans3.data

            # post-processing of the prediction
            for oi, o in enumerate(trainopt.outputs):
                if o == 'unoriented_normals' or o == 'oriented_normals':
                    o_pred1 = pred1[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
                    o_pred2 = pred2[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
                    o_pred3 = pred3[:, output_pred_ind[oi]:output_pred_ind[oi]+3]

                    if trainopt.use_point_stn:
                        # transform predictions with inverse transform
                        # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                        o_pred1[:, :] = torch.bmm(o_pred1.unsqueeze(1), trans1.transpose(2, 1)).squeeze(1)
                        o_pred2[:, :] = torch.bmm(o_pred2.unsqueeze(1), trans2.transpose(2, 1)).squeeze(1)
                        o_pred3[:, :] = torch.bmm(o_pred3.unsqueeze(1), trans3.transpose(2, 1)).squeeze(1)

                    if trainopt.use_pca:
                        # transform predictions with inverse pca rotation (back to world space)
                        o_pred1[:, :] = torch.bmm(o_pred1.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(1)

                    # normalize normals
                    o_pred_len1 = torch.max(torch.cuda.FloatTensor([sys.float_info.epsilon*100]), o_pred1.norm(p=2, dim=1, keepdim=True))
                    o_pred1 = o_pred1 / o_pred_len1
                    o_pred_len2 = torch.max(torch.cuda.FloatTensor([sys.float_info.epsilon*100]), o_pred2.norm(p=2, dim=1, keepdim=True))
                    o_pred2 = o_pred2/ o_pred_len2
                    o_pred_len3 = torch.max(torch.cuda.FloatTensor([sys.float_info.epsilon*100]), o_pred3.norm(p=2, dim=1, keepdim=True))
                    o_pred3 = o_pred3 / o_pred_len3
                elif o == 'max_curvature' or o == 'min_curvature':
                    o_pred1 = pred1[:, output_pred_ind[oi]:output_pred_ind[oi]+1]

                    # undo patch size normalization:
                    o_pred1[:, :] = o_pred1 / dataset.patch_radius_absolute[shape_ind][0]

                else:
                    raise ValueError('Unsupported output type: %s' % (o))

            print('[%s %d/%d] shape %s' % (model_name, batchind, num_batch-1, dataset.shape_names[shape_ind]))

            batch_offset = 0
            while batch_offset < pred1.size(0):

                shape_patches_remaining = shape_patch_count-shape_patch_offset
                batch_patches_remaining = pred1.size(0)-batch_offset

                # append estimated patch properties batch to properties for the current shape on the CPU
                shape_properties1[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining, batch_patches_remaining), :] =o_pred1[
                    batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]
                shape_properties2[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining, batch_patches_remaining), :] = o_pred2[
                    batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]
                shape_properties3[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining, batch_patches_remaining), :] = o_pred3[
                    batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]
                shape_max_v[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining, batch_patches_remaining), :] = v[
                    batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]


                batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
                shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

                if shape_patches_remaining <= batch_patches_remaining:

                    # save shape properties to disk
                    prop_saved = [False]*len(trainopt.outputs)

                    # save normals
                    oi = [i for i, o in enumerate(trainopt.outputs) if o in ['unoriented_normals', 'oriented_normals']]
                    if len(oi) > 1:
                        raise ValueError('Duplicate normal output.')
                    elif len(oi) == 1:
                        oi = oi[0]
                        normal_prop1 = shape_properties1[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'_scale1.normals'), normal_prop1.numpy())
                        normal_prop2 = shape_properties2[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'_scale2.normals'), normal_prop2.numpy())
                        normal_prop3 = shape_properties3[:, output_pred_ind[oi]:output_pred_ind[oi]+3]
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'_scale3.normals'), normal_prop3.numpy())
                        v_prop = shape_max_v[:, 0:3]
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'_v.normals'), v_prop.numpy())
                        prop_saved[oi] = True

                    # save curvatures
                    oi1 = [i for i, o in enumerate(trainopt.outputs) if o == 'max_curvature']
                    oi2 = [i for i, o in enumerate(trainopt.outputs) if o == 'min_curvature']
                    if len(oi1) > 1 or len(oi2) > 1:
                        raise ValueError('Duplicate minimum or maximum curvature output.')
                    elif len(oi1) == 1 or len(oi2) == 1:
                        curv_prop = torch.FloatTensor(shape_properties.size(0), 2).zero_()
                        if len(oi1) == 1:
                            oi1 = oi1[0]
                            curv_prop[:, 0] = shape_properties[:, output_pred_ind[oi1]:output_pred_ind[oi1]+1]
                            prop_saved[oi1] = True
                        if len(oi2) == 1:
                            oi2 = oi2[0]
                            curv_prop[:, 1] = shape_properties[:, output_pred_ind[oi2]:output_pred_ind[oi2]+1]
                            prop_saved[oi2] = True
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'.curv'), curv_prop.numpy())

                    if not all(prop_saved):
                        raise ValueError('Not all shape properties were saved, some of them seem to be unsupported.')

                    # save point indices
                    if opt.sampling != 'full':
                        np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'.idx'), datasampler.shape_patch_inds[shape_ind], fmt='%d')

                    # start new shape
                    if shape_ind + 1 < len(dataset.shape_names):
                        shape_patch_offset = 0
                        shape_ind = shape_ind + 1
                        if opt.sampling == 'full':
                            shape_patch_count = dataset.shape_patch_count[shape_ind]
                        elif opt.sampling == 'sequential_shapes_random_patches':
                            # shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
                            shape_patch_count = len(datasampler.shape_patch_inds[shape_ind])
                        else:
                            raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
                        shape_properties1 = torch.FloatTensor(shape_patch_count, pred_dim).zero_()
                        shape_properties2 = torch.FloatTensor(shape_patch_count, pred_dim).zero_()
                        shape_properties3 = torch.FloatTensor(shape_patch_count, pred_dim).zero_()
                        shape_max_v = torch.FloatTensor(shape_patch_count, 3).zero_()

def compute_loss(pred1,pred2,pred3, target, output_loss_weight, patch_rot1, patch_rot2,patch_rot3):

    loss = 0


    o_pred1 = pred1
    o_pred2 = pred2
    o_pred3 = pred3

    o_target = target[2]

    if patch_rot1 is not None:
        # transform predictions with inverse transform
        # since we know the transform to be a rotation (QSTN), the transpose is the inverse
        o_pred1 = torch.bmm(o_pred1.unsqueeze(1), patch_rot1.transpose(2, 1)).squeeze(1)
        o_pred2 = torch.bmm(o_pred2.unsqueeze(1), patch_rot2.transpose(2, 1)).squeeze(1)
        o_pred3 = torch.bmm(o_pred3.unsqueeze(1), patch_rot3.transpose(2, 1)).squeeze(1)


        loss2_1= torch.min((o_pred1-o_target).pow(2).sum(1), (o_pred1+o_target).pow(2).sum(1))
        loss2_2= torch.min((o_pred2-o_target).pow(2).sum(1), (o_pred2+o_target).pow(2).sum(1))
        loss2_3= torch.min((o_pred3-o_target).pow(2).sum(1), (o_pred3+o_target).pow(2).sum(1))

        tt = torch.mul(loss2_1, output_loss_weight[:,0])+\
                torch.mul(loss2_2, output_loss_weight[:, 1])+\
                torch.mul(loss2_3, output_loss_weight[:, 2])
        loss += tt.mean()


    return loss

if __name__ == '__main__':
    eval_opt = parse_arguments()
    eval_pcpnet(eval_opt)
