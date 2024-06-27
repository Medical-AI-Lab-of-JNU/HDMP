import torch
import torch.nn as nn
from models.convgru import PkGRU
from models.decoder import Decoder
from models.atten import Feature_Reweighting
import torch.nn.functional as F



def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class pregruuet(nn.Module):
    def __init__(self, args, device):
        super(pregruuet, self).__init__()
        self.args = args

        self.img_size = args.img_size

        self.ch = 128
        fea_dim = 192
        reduce_dim = 128
        self.hid = 16
        self.layer0 = DoubleConv(1, self.hid)
        self.layer1 = DoubleConv(self.hid, 2 * self.hid)
        self.layer2 = DoubleConv(2 * self.hid, 4 * self.hid)
        self.layer3 = DoubleConv(4 * self.hid, 8 * self.hid)
        #self.layer4 = DoubleConv(8 * self.hid, 8 * self.hid)
        self.down = nn.MaxPool2d(2)
        self.decoder = Decoder().to(device)
        self.PkGRU = PkGRU(in_channels=128,
                                  hidden_channels=128,
                                  kernel_size=(3, 3),
                                  num_layers=args.n_layer,
                                  device=device).to(device)
        self.Feature_Reweighting = Feature_Reweighting()

        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.corr_conv = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + 1, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.q_conv0 = nn.Sequential(
            nn.Conv2d(self.hid*3, self.hid, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid, self.hid, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.q_conv1 = nn.Sequential(
            nn.Conv2d(self.hid*6, self.hid*2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid * 2, self.hid * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.q_conv2 = nn.Sequential(
            nn.Conv2d(self.hid * 12, self.hid * 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid * 4, self.hid * 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.q_conv3 = nn.Sequential(
            nn.Conv2d(self.hid * 24, self.hid * 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hid * 8, self.hid * 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.relation_coding = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, 128, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
            nn.Sigmoid())

        self.k_shot = args.k_shot
        self.q_slice =args.slices
        self.batch_size = args.batch_size
        self.train_iter = args.train_iter
        self.eval_iter = args.eval_iter
        self.mode = 'Learnable'

    def forward(self, s_T0_x, s_T1_x, s_T2_x, s_y, q_T0_x, q_T1_x, q_T2_x, s_seed=None):
        """
        s_T0_x, s_T1_x, s_T2_x = B Shot Slices C H W tensors.
        s_y = B Shot Slices 1 H W tensors
        q_T0_x, q_T1_x, q_T2_x = B Slices C H W tensors
        """
        s_feats, q_feats = [], []
        q_ft_lists = []
        mask_list = []
        supp_feat_ave_lists = [[0] for _ in range(4)]
        for framed_id in range(self.q_slice):
            s_feats_shot = []
            mask_list1 = []
            supp_feat_ave_list = [0 for _ in range(4)]
            for i in range(self.k_shot):
                s_ft_list = []
                mask = (s_y[:, i, framed_id, :, :, :] == 1).float()

                s_x0 = s_T0_x[:, i, framed_id, :, :, :]     #[b, 1, w, h]
                s_x1 = s_T1_x[:, i, framed_id, :, :, :]
                s_x2 = s_T2_x[:, i, framed_id, :, :, :]


                s_x0_encoder_0, s_x1_encoder_0, s_x2_encoder_0 = self.layer0(s_x0), self.layer0(s_x1), self.layer0(
                    s_x2)  # b*16*256*256
                s_ft_list.append(s_x1_encoder_0)
                s_x0_encoder_0, s_x1_encoder_0, s_x2_encoder_0 = self.down(s_x0_encoder_0), self.down(s_x1_encoder_0), self.down(s_x2_encoder_0)
                s_x0_encoder_1, s_x1_encoder_1, s_x2_encoder_1 = self.layer1(s_x0_encoder_0), self.layer1(
                    s_x1_encoder_0), self.layer1(s_x2_encoder_0)  # b*32*128*128
                s_ft_list.append(s_x1_encoder_1)
                s_x0_encoder_1, s_x1_encoder_1, s_x2_encoder_1 = self.down(s_x0_encoder_1), self.down(s_x1_encoder_1), self.down(s_x2_encoder_1)
                s_x0_encoder_2, s_x1_encoder_2, s_x2_encoder_2 = self.layer2(s_x0_encoder_1), self.layer2(
                    s_x1_encoder_1), self.layer2(s_x2_encoder_1)  # b*64*64*64
                s_ft_list.append(s_x1_encoder_2)
                s_x0_encoder_2, s_x1_encoder_2, s_x2_encoder_2 = self.down(s_x0_encoder_2), self.down(s_x1_encoder_2), self.down(s_x2_encoder_2)
                s_x0_encoder_3, s_x1_encoder_3, s_x2_encoder_3 = self.layer3(s_x0_encoder_2), self.layer3(
                    s_x1_encoder_2), self.layer3(s_x2_encoder_2)  # b*128*32*32
                s_ft_list.append(s_x1_encoder_3)
                mask = F.interpolate(mask, size=(s_x0_encoder_3.size(2), s_x0_encoder_3.size(3)), mode='bilinear',
                                     align_corners=True)  # b*1*32*32


                mask_list1.append(mask)

                for j in range(4):
                    supp_feat_ave_list[j] += s_ft_list[j] / self.k_shot


                s_x0_encoder_2 = F.interpolate(s_x0_encoder_2, size=(s_x0_encoder_3.size(2), s_x0_encoder_3.size(3)),
                                               mode='bilinear', align_corners=True)
                s_x1_encoder_2 = F.interpolate(s_x1_encoder_2, size=(s_x0_encoder_3.size(2), s_x0_encoder_3.size(3)),
                                               mode='bilinear', align_corners=True)
                s_x2_encoder_2 = F.interpolate(s_x2_encoder_2, size=(s_x0_encoder_3.size(2), s_x0_encoder_3.size(3)),
                                               mode='bilinear', align_corners=True)
                s_x0_encoder = torch.cat([s_x0_encoder_3, s_x0_encoder_2], dim=1)
                s_x1_encoder = torch.cat([s_x1_encoder_3, s_x1_encoder_2], dim=1)
                s_x2_encoder = torch.cat([s_x2_encoder_3, s_x2_encoder_2], dim=1)
                s_x0_encoder = self.down_supp(s_x0_encoder)
                s_x1_encoder = self.down_supp(s_x1_encoder)
                s_x2_encoder = self.down_supp(s_x2_encoder)

                # PkGRU
                s_x_fwd_encoders_2 = torch.stack((s_x0_encoder, s_x1_encoder, s_x2_encoder), dim=1)  # [b,t,c,w,h]
                s_x_rev_encoders_2 = torch.stack((s_x2_encoder, s_x1_encoder, s_x0_encoder), dim=1)
                s_encoders = self.PkGRU(s_x_fwd_encoders_2, s_x_rev_encoders_2)  # [b, t, c, w, h]
                s_encoder = s_encoders[:, 1, :, :, :]  # b*c*h*w
                s_feats_shot.append(s_encoder)

            for j in range(4):
                supp_feat_ave_lists[j].append(supp_feat_ave_list[j])

            q_ft_list = []

            q_x0 = q_T0_x[:, framed_id, :, :, :]  # [B, 1, 256, 256]
            q_x1 = q_T1_x[:, framed_id, :, :, :]
            q_x2 = q_T2_x[:, framed_id, :, :, :]

            q_x0_encoder_0, q_x1_encoder_0, q_x2_encoder_0 = self.layer0(q_x0), self.layer0(q_x1), self.layer0(
                q_x2)  # b*16*256*256
            q_encoder_0 = torch.cat((q_x0_encoder_0, q_x1_encoder_0, q_x2_encoder_0), dim=1)
            q_encoder_0 = self.q_conv0(q_encoder_0)
            q_ft_list.append(q_encoder_0)
            q_x0_encoder_0, q_x1_encoder_0, q_x2_encoder_0 = self.down(q_x0_encoder_0), self.down(
                q_x1_encoder_0), self.down(q_x2_encoder_0)
            q_x0_encoder_1, q_x1_encoder_1, q_x2_encoder_1 = self.layer1(q_x0_encoder_0), self.layer1(
                q_x1_encoder_0), self.layer1(q_x2_encoder_0)  # b*32*128*128
            q_encoder_1 = torch.cat((q_x0_encoder_1, q_x1_encoder_1, q_x2_encoder_1), dim=1)
            q_encoder_1 = self.q_conv1(q_encoder_1)
            q_ft_list.append(q_encoder_1)
            q_x0_encoder_1, q_x1_encoder_1, q_x2_encoder_1 = self.down(q_x0_encoder_1), self.down(
                q_x1_encoder_1), self.down(q_x2_encoder_1)
            q_x0_encoder_2, q_x1_encoder_2, q_x2_encoder_2 = self.layer2(q_x0_encoder_1), self.layer2(
                q_x1_encoder_1), self.layer2(q_x2_encoder_1)  # b*64*64*64
            q_encoder_2 = torch.cat((q_x0_encoder_2, q_x1_encoder_2, q_x2_encoder_2), dim=1)
            q_encoder_2 = self.q_conv2(q_encoder_2)
            q_ft_list.append(q_encoder_2)
            q_x0_encoder_2, q_x1_encoder_2, q_x2_encoder_2 = self.down(q_x0_encoder_2), self.down(
                q_x1_encoder_2), self.down(q_x2_encoder_2)
            q_x0_encoder_3, q_x1_encoder_3, q_x2_encoder_3 = self.layer3(q_x0_encoder_2), self.layer3(
                q_x1_encoder_2), self.layer3(q_x2_encoder_2)  # b*128*32*32
            q_encoder_3 = torch.cat((q_x0_encoder_3, q_x1_encoder_3, q_x2_encoder_3), dim=1)
            q_encoder_3 = self.q_conv3(q_encoder_3)
            q_ft_list.append(q_encoder_3)

            q_ft_lists.append(q_ft_list)

            q_x0_encoder_2 = F.interpolate(q_x0_encoder_2, size=(q_x0_encoder_3.size(2), q_x0_encoder_3.size(3)),
                                           mode='bilinear', align_corners=True)
            q_x1_encoder_2_0 = F.interpolate(q_x1_encoder_2, size=(q_x0_encoder_3.size(2), q_x0_encoder_3.size(3)),
                                             mode='bilinear', align_corners=True)
            q_x2_encoder_2 = F.interpolate(q_x2_encoder_2, size=(q_x0_encoder_3.size(2), q_x0_encoder_3.size(3)),
                                           mode='bilinear', align_corners=True)
            q_x0_encoder = torch.cat([q_x0_encoder_3, q_x0_encoder_2], dim=1)
            q_x1_encoder = torch.cat([q_x1_encoder_3, q_x1_encoder_2_0], dim=1)
            q_x2_encoder = torch.cat([q_x2_encoder_3, q_x2_encoder_2], dim=1)
            q_x0_encoder = self.down_query(q_x0_encoder)
            q_x1_encoder = self.down_query(q_x1_encoder)
            q_x2_encoder = self.down_query(q_x2_encoder)

            # PkGRU
            q_x_fwd_encoders = torch.stack((q_x0_encoder, q_x1_encoder, q_x2_encoder), dim=1)  # [b,t,c,w,h]
            q_x_rev_encoders = torch.stack((q_x2_encoder, q_x1_encoder, q_x0_encoder), dim=1)
            q_encoders = self.PkGRU(q_x_fwd_encoders, q_x_rev_encoders)  # [b, t, c, w, h]
            q_encoder = q_encoders[:, 1, :, :, :]  # b*c*h*w
            q_feats.append(q_encoder)

            mask1 = torch.stack(mask_list1, dim=1)      #[b, shot, 1, w, h]
            mask_list.append(mask1)
            s_feat = torch.stack(s_feats_shot, dim=1)   #[b, shot, c, w, h]
            s_feats.append(s_feat)

        mask_list = torch.stack(mask_list, dim=1)       #[b, slice, shot, 1, w, h]
        s_feats_list = torch.stack(s_feats, dim=1)      #[b, slice, shot, c, w, h]
        q_feats_list = torch.stack(q_feats, dim=1)      #[b, slice, c, w, h]


########################### adaptive superpixel clustering ###########################
        _, _, _, max_num_sp, _ = s_seed.size()  # b x shot x slice x max_num_sp x 2
        guide_feat_list = []
        prob_map_list = []
        for bs_ in range(self.batch_size):
            guide_feat_list1 = []
            prob_map_list1 = []
            for framed_id in range(self.q_slice):
                sp_center_list = []
                query_feat_ = q_feats_list[bs_, framed_id, :, :, :]  # c x h x w
                for shot_ in range(self.k_shot):
                    with torch.no_grad():
                        supp_feat_ = s_feats_list[bs_, framed_id, shot_, :, :, :]  # c x h x w
                        supp_mask_ = mask_list[bs_, framed_id, shot_, :, :, :]  # 1 x h x w
                        supp_mask_bg = 1 - supp_mask_
                        supp_proto_bg = Weighted_GAP(supp_feat_.unsqueeze(0), supp_mask_bg.unsqueeze(0))
                        s_seed_ = s_seed[bs_, shot_, framed_id, :, :]  # max_num_sp x 2
                        num_sp = max(len(torch.nonzero(s_seed_[:, 0])), len(torch.nonzero(s_seed_[:, 1])))

                        # if num_sp == 0 or 1, use the Masked Average Pooling instead
                        if (num_sp == 0) or (num_sp == 1):
                            supp_proto = Weighted_GAP(supp_feat_.unsqueeze(0), supp_mask_.unsqueeze(0))  # 1 x c x 1 x 1
                            sp_center_list.append(
                                supp_proto.squeeze().unsqueeze(-1))  # c x 1
                            continue

                        s_seed_ = s_seed_[:num_sp, :]  # num_sp x 2
                        sp_init_center = supp_feat_[:, s_seed_[:, 0], s_seed_[:, 1]]  # c x num_sp (sp_seed)
                        sp_init_center = torch.cat([sp_init_center, s_seed_.transpose(1, 0).float()],
                                                   dim=0)  # (c + xy) x num_sp

                        if self.training:
                            sp_center = self.sp_center_iter(supp_feat_, supp_mask_, sp_init_center,
                                                            n_iter=self.train_iter)     #c x num_sp (sp_seed)
                            sp_center = torch.cat([sp_center, supp_proto_bg.squeeze().unsqueeze(-1)], dim=1)
                            sp_center_list.append(sp_center)
                        else:
                            sp_center = self.sp_center_iter(supp_feat_, supp_mask_, sp_init_center,
                                                            n_iter=self.eval_iter)
                            sp_center = torch.cat([sp_center, supp_proto_bg.squeeze().unsqueeze(-1)], dim=1)
                            sp_center_list.append(sp_center)

                sp_center = torch.cat(sp_center_list, dim=1)  # c x num_sp_all (collected from all shots)

                # when support only has one prototype in 1-shot training
                if (self.k_shot == 1) and (sp_center.size(1) == 1):
                    cos_sim_map = self.query_region_activate(query_feat_, sp_center, self.mode)
                    prob_map_list1.append(cos_sim_map.unsqueeze(0))
                    sp_center_tile = sp_center[None, ..., None].expand(-1, -1, query_feat_.size(1),
                                                                       query_feat_.size(2))  # 1 x c x h x w
                    guide_feat = torch.cat([query_feat_.unsqueeze(0), sp_center_tile], dim=1)  # 1 x 2c x h x w
                    guide_feat_list1.append(guide_feat)
                    continue

                #sp_center_rep = sp_center[..., None, None].repeat(1, 1, query_feat_.size(1), query_feat_.size(2))
                cos_sim_map = self.query_region_activate(query_feat_, sp_center, self.mode)  #The last channel: proto_bg
                prob_map, _ = torch.max(cos_sim_map[:-1, :, :], dim=0, keepdim=True)
                prob_map_list1.append(prob_map.unsqueeze(0))

                guide_map = cos_sim_map.max(0)[1]  # h x w
                sp_guide_feat = sp_center[:, guide_map]  # c x h x w
                guide_feat = torch.cat([query_feat_, sp_guide_feat], dim=0)  # 2c x h x w
                guide_feat_list1.append(guide_feat.unsqueeze(0))
            guide_feat1 = torch.cat(guide_feat_list1, dim=0)  # slice x 2c x h x w
            guide_feat_list.append(guide_feat1.unsqueeze(0))
            prob_map1 = torch.cat(prob_map_list1, dim=0)    # slice x 1 x h x w
            prob_map_list.append(prob_map1.unsqueeze(0))
        guide_feat = torch.cat(guide_feat_list, dim=0)  #b x slice x 2c x h x w
        prob_map = torch.cat(prob_map_list, dim=0)      #b x slice x 1 x h x w
        final_feat_list = []
        for i in range(self.q_slice):
            guide_feat_ = guide_feat[:, i, ...]
            prob_map_ = prob_map[:, i, ...]
            final_feat_ = self.corr_conv(torch.cat([guide_feat_, prob_map_], 1))   #b x c x h x w
            final_feat_list.append(final_feat_)
        final_feat = torch.stack(final_feat_list, dim=1)
        out = []
        for frame_id in range(self.q_slice):
            hi = final_feat[:, frame_id, :, :, :]
            q_ft_list = q_ft_lists[frame_id]

            # Cross-Fusion
            cf_lists = []
            for i in range(4):
                s_ft = supp_feat_ave_lists[i][frame_id + 1]
                referenced_feat = self.Feature_Reweighting(s_ft, q_ft_list[i])
                cf_lists.append(referenced_feat)

            y_pred = self.decoder(hi, cf_lists)
            out.append(y_pred)
        return out


    def query_region_activate(self, query_fea, prototypes, mode):
        """
        Input:  query_fea:      [c, h, w]
                prototypes:     [c, n]
                mode:           Cosine/Learnable
        Output: activation_map: [n, h, w]
        """
        query_fea = query_fea.unsqueeze(dim=0) #[b=1, c, h, w]
        prototypes = prototypes.transpose(0, 1).unsqueeze(dim=0).unsqueeze(-1).unsqueeze(-1)   #[b=1, n, c, 1, 1]

        b, c, h, w = query_fea.shape
        n = prototypes.shape[1]
        que_temp = query_fea.reshape(b, c, h*w)

        if mode == 'Cosine':
            que_temp = que_temp.unsqueeze(dim=1)           # [b, 1, c, h*w]
            prototypes_temp = prototypes.squeeze(dim=-1)   # [b, n, c, 1]
            map_temp = nn.CosineSimilarity(2)(que_temp, prototypes_temp)  # [n, c, h*w]
            activation_map = map_temp.reshape(b, n, h, w)
            activation_map = (activation_map+1)/2          # Normalize to (0,1)
            return activation_map.squeeze(dim=0)

        elif mode == 'Learnable':
            for p_id in range(n):
                prototypes_temp = prototypes[:,p_id,:,:,:]                         # [b, c, 1, 1]
                prototypes_temp = prototypes_temp.expand(b, c, h, w)
                concat_fea = torch.cat([query_fea, prototypes_temp], dim=1)        # [b, 2c, h, w]
                if p_id == 0:
                    activation_map = self.relation_coding(concat_fea)              # [b, 1, h, w]
                else:
                    activation_map_temp = self.relation_coding(concat_fea)              # [b, 1, h, w]
                    activation_map = torch.cat([activation_map,activation_map_temp], dim=1)
            return activation_map.squeeze(dim=0)
    def sp_center_iter(self, supp_feat, supp_mask, sp_init_center, n_iter):
        '''
        :param supp_feat: A Tensor of support feature, (C, H, W)
        :param supp_mask: A Tensor of support mask, (1, H, W)
        :param sp_init_center: A Tensor of initial sp center, (C + xy, num_sp)
        :param n_iter: The number of iterations
        :return: sp_center: The centroid of superpixels (prototypes)
        '''

        c_xy, num_sp = sp_init_center.size()
        _, h, w = supp_feat.size()
        h_coords = torch.arange(h).view(h, 1).contiguous().repeat(1, w).unsqueeze(0).float().cuda()
        w_coords = torch.arange(w).repeat(h, 1).unsqueeze(0).float().cuda()
        supp_feat = torch.cat([supp_feat, h_coords, w_coords], 0)
        supp_feat_roi = supp_feat[:, (supp_mask == 1).squeeze()]  # (C + xy) x num_roi

        num_roi = supp_feat_roi.size(1)
        supp_feat_roi_rep = supp_feat_roi.unsqueeze(-1).repeat(1, 1, num_sp)
        sp_center = torch.zeros_like(sp_init_center).cuda()  # (C + xy) x num_sp

        for i in range(n_iter):
            # Compute association between each pixel in RoI and superpixel
            if i == 0:
                sp_center_rep = sp_init_center.unsqueeze(1).repeat(1, num_roi, 1)
            else:
                sp_center_rep = sp_center.unsqueeze(1).repeat(1, num_roi, 1)
            assert supp_feat_roi_rep.shape == sp_center_rep.shape  # (C + xy) x num_roi x num_sp
            dist = torch.pow(supp_feat_roi_rep - sp_center_rep, 2.0)
            feat_dist = dist[:-2, :, :].sum(0)
            spat_dist = dist[-2:, :, :].sum(0)
            total_dist = torch.pow(feat_dist + spat_dist / 100, 0.5)
            p2sp_assoc = torch.neg(total_dist).exp()
            p2sp_assoc = p2sp_assoc / (p2sp_assoc.sum(0, keepdim=True))  # num_roi x num_sp

            sp_center = supp_feat_roi_rep * p2sp_assoc.unsqueeze(0)  # (C + xy) x num_roi x num_sp
            sp_center = sp_center.sum(1)

        return sp_center[:-2, :]
