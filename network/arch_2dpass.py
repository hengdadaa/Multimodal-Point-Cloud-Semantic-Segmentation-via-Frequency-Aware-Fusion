import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from network.basic_block import Lovasz_loss
from network.spvcnn import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block import ResNetFCN
from network.fam import FAM





class Attention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(Attention, self).__init__()
        self.pseudo_in, self.valid_in = channels
       
        middle = self.valid_in  

        
        self.fc1 = nn.Linear(self.pseudo_in, middle)  
        self.fc2 = nn.Linear(self.valid_in, middle)
        self.fc3 = nn.Linear(2 * middle, 2)  

        # Define convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.pseudo_in, self.valid_in, 1),
            nn.BatchNorm1d(self.valid_in),
            nn.ReLU()
        ) 
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.valid_in, self.valid_in, 1),
            nn.BatchNorm1d(self.valid_in),
            nn.ReLU()
        )

        # block = FAM(feature_dim=feature_dim, N=num_elements).to('cuda')

        # Channel and Spatial Attention
        # self.channel_attention = ChannelAttention(self.valid_in, reduction)
        # self.spatial_attention = SpatialAttention()
        self.down_up_sample = DownUpSample(in_channels=64, out_channels=64)

        #################################test
        # self.agent_attention = None
        # self.reduce_conv = nn.Conv1d(64, 64, kernel_size=16, stride=2, padding=0)
        # self.restore_conv = nn.ConvTranspose1d(64, 64, kernel_size=2, stride=2, padding=0)

    def forward(self, pseudo_feas, valid_feas):
        # pfeas = []  （n，c）
        # ifeas = []
        # leng = 0
        # for i in range(len(list)):
        #     if i == 0:
        #         pfeas.append(pseudo_feas[leng:(leng + list[i])])
        #         ifeas.append(valid_feas[leng:(leng + list[i])])
        #         leng += list[i]
        #     else:
        #         pfeas.append(pseudo_feas[leng:(leng + list[i])])
        #         ifeas.append(valid_feas[leng:(leng + list[i])])
        #         leng += list[i]
        pseudo_feas = torch.unsqueeze(pseudo_feas, dim=0).permute(0, 2, 1)
        # 在第0加一个维度 变为 (1, N, C) 最终变为(1, C, N)
        valid_feas = torch.unsqueeze(valid_feas, dim=0).permute(0, 2, 1)
        # 同理
        batch = pseudo_feas.size(0)
        # batch=1

        pseudo_feas_f = pseudo_feas.transpose(1, 2).contiguous().view(-1, self.pseudo_in)
        # 交换位置 最终变为（n，64） (N, C)      其中contiguous() 确保张量在内存中是连续的，便于后续的 view 操作。
        # 就是 变成 n 64    用semantickitti来举例
        valid_feas_f = valid_feas.transpose(1, 2).contiguous().view(-1, self.valid_in)

        pseudo_feas_f_ = self.fc1(pseudo_feas_f)  # 通过全连接层 让维度变化 (N, middle)
        valid_feas_f_ = self.fc2(valid_feas_f)  # 同理
        # pseudo_valid_feas_f = pseudo_feas_f_ + valid_feas_f_
        # print(pseudo_valid_feas_f.shape)
        pseudo_feas_f_ = pseudo_feas_f_.view(1, 64, -1)
        valid_feas_f_ = valid_feas_f_.view(1, 64, -1)
        # 拼接后的结果中，前middle列是pseudo_feas_f_的内容，后middle列是valid_feas_f_的内容  维度（n，2*middle）
        # print(pseudo_valid_feas_f.shape)   # torch.Size([88171, 32])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pseudo_feas_f_ = pseudo_feas_f_.to(device)
        valid_feas_f_ = valid_feas_f_.to(device)

        pseudo_valid_feas_f = self.down_up_sample(pseudo_feas_f_, valid_feas_f_)
        # print('222222222222', valid_features_att.shape)
        pseudo_valid_feas_f = pseudo_valid_feas_f.view(-1, 64)
        # print(pseudo_valid_feas_f.shape)

        ################2

        # valid_features_att = self.reduce_conv(valid_features_att)
        # print(f"Reduced shape: {valid_features_att.shape}")

        # _, num_patches, dim = valid_features_att.shape
        # H = W = int(np.sqrt(num_patches))
        # pts_feat = AgentAttention(dim=dim, num_patches=num_patches)(pts_feat, H, W)
        # print(f"Restored shape: {valid_features_att.shape}")

        #
        # valid_features_att = self.restore_conv(valid_features_att)
        # print(f"Restored shape: {valid_features_att.shape}")
        ########2
        #         print('111111111111111', pseudo_features_att.shape)
        #             # print('222222222222222',pts_pred.shape)

        #         pseudo_features_att = self.reduce_conv(pseudo_features_att)
        #         print(f"Reduced shape: {pseudo_features_att.shape}")

        #         _, num_patches, dim = pseudo_features_att.shape
        #         H = W = int(np.sqrt(num_patches))
        #         #pts_feat = AgentAttention(dim=dim, num_patches=num_patches)(pts_feat, H, W)
        #         print(f"Restored shape: {pseudo_features_att.shape}")

        #     
        #         pseudo_features_att = self.restore_conv(pseudo_features_att)
        #         print(f"Restored shape: {pseudo_features_att.shape}")

        return pseudo_valid_feas_f


# Example usage:
# model = Attention(channels=(pseudo_in_channels, valid_in_channels))
# output = model(pseudo_feas, valid_feas)


class xModalKD(nn.Module):
    def __init__(self, config):
        super(xModalKD, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.leaners = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.leaners.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs1.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))

        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None
        self.attention = Attention(channels=[64, 64])
        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights,
                                           ignore_index=config['dataset_params']['ignore_label'])
        self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

        self.reduce_conv = nn.Conv1d(64, 64, kernel_size=10, stride=10, padding=0)
        self.restore_conv = nn.ConvTranspose1d(64, 64, kernel_size=10, stride=10, padding=0)

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        # pts_fea 处理后的点云数据
        # p2i就是找到与图像对应的点云索引
        # 总的图片对应的有效值
        img_feat = []
        for b in range(batch_idx.max() + 1):
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
        labels = unq_lbxyz[:, 0][label_ind]
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    def forward(self, data_dict):
        loss = 0
        img_seg_feat = []
        batch_idx = data_dict['batch_idx']
        point2img_index = data_dict['point2img_index']

        for idx, scale in enumerate(self.scale_list):
            img_feat = data_dict['img_scale{}'.format(scale)]
            pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
            coors_inv = data_dict['scale_{}'.format(scale)]['coors_inv']
            # 映射

            # 3D prediction
            pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)

            # correspondence
            pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}'.format(idx)]['full_coors'])
            pts_feat = self.p2img_mapping(pts_feat[coors_inv], point2img_index, batch_idx)  # 这个就是另一个模态
            pts_pred = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx)
            # 

            #             print('111111111111111', pts_feat.shape)
            #             # print('222222222222222',pts_pred.shape)
            #             pts_feat = torch.unsqueeze(pts_feat, dim=0).permute(0, 2, 1)

            #             pts_feat = self.reduce_conv(pts_feat)
            #             print(f"Reduced shape: {pts_feat.shape}")

            #             _, num_patches, dim = pts_feat.shape
            #             H = W = int(np.sqrt(num_patches))
            #             pts_feat = AgentAttention(dim=dim, num_patches=num_patches)(pts_feat, H, W)
            #             print(f"Restored shape: {pts_feat.shape}")

            #           
            #             pseudo_features_restored = self.restore_conv(pseudo_features_reduced)
            #             print(f"Restored shape: {pseudo_features_restored.shape}")

            #             pts_feat = pts_feat.squeeze(dim=0).permute(1, 0)
            #             print('2', pts_feat.shape)

            fuse_pred = self.attention(pts_feat, img_feat)

            img_seg_feat.append(fuse_pred)
            fuse_pred = self.multihead_fuse_classifier[idx](fuse_pred)

            # Segmentation Loss
            seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)
            seg_loss_2d = self.seg_loss(fuse_pred, data_dict['img_label'])
            loss += seg_loss_3d + seg_loss_2d * self.lambda_seg2d / self.num_scales

            # KL divergence
            xm_loss = F.kl_div(
                F.log_softmax(pts_pred, dim=1),
                F.softmax(fuse_pred, dim=1),
            )
            loss += xm_loss * self.lambda_xm / self.num_scales

        img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
        loss += self.seg_loss(img_seg_logits, data_dict['img_label'])
        data_dict['loss'] += loss

        return data_dict


class DownUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownUpSample, self).__init__()
        self.reduce_conv = nn.Conv1d(in_channels, out_channels, kernel_size=16, stride=16, padding=0)
        
        self.restore_conv = nn.ConvTranspose1d(out_channels, in_channels, kernel_size=16, stride=16, padding=0)

    def forward(self, x, y):

        reduced_x = self.reduce_conv(x)
       
        reduced_y = self.reduce_conv(y)
        

        _, num_patches, dim = reduced_x.shape


        feature_dim = dim
        N = num_patches

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pseudo_valid_feas_f = FAM(N=N, feature_dim=feature_dim).to(device)(reduced_x, reduced_y)
        restored_x = self.restore_conv(pseudo_valid_feas_f)

        if restored_x.shape[2] < x.shape[2]:
            pad_size = x.shape[2] - restored_x.shape[2]
            restored_x = nn.functional.pad(restored_x, (0, pad_size))

        elif restored_x.shape[2] > x.shape[2]:
            restored_x = restored_x[:, :, :x.shape[2]]  

        return restored_x



class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)

        self.model_3d = SPVCNN(config)
        if not self.baseline_only:
            self.model_2d = ResNetFCN(
                backbone=config.model_params.backbone_2d,
                pretrained=config.model_params.pretrained2d,
                config=config
            )
            self.fusion = xModalKD(config)
        else:
            print('Start vanilla training!')

    def forward(self, data_dict):
        # 3D network
        data_dict = self.model_3d(data_dict)

        # training with 2D network
        if self.training and not self.baseline_only:
            data_dict = self.model_2d(data_dict)

            data_dict = self.fusion(data_dict)

        return data_dict
