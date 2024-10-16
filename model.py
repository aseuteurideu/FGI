import torch
import torch.nn as nn
from select_backbone import select_resnet_half


######################################################################################
class My_CNN_RawAud(nn.Module):
    def __init__(self):
        super(My_CNN_RawAud, self).__init__()

        self.netcnnaud_layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=80, stride=8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )  # output: batch x 128 x 5991

        self.netcnnaud_layer2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(192),
        )  # output: batch x 192 x 1497

    def forward(self, x):
        x1 = self.netcnnaud_layer1(x)
        x2 = self.netcnnaud_layer2(x1)
        return x2


class My_Network(nn.Module):
    def __init__(self, network='resnet18',with_attention=False,
                 residual_conn=False, spatial_size=28):
        super(My_Network, self).__init__()

        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32
        self.aud2visspace = None

        self.with_attention = with_attention
        self.residual_conn = residual_conn
        self.spatial_size = spatial_size

        self.netcnnaud = My_CNN_RawAud()

        a_size = [192, 1497]
        v_size = [128, 15, spatial_size, spatial_size]
        stride = int(a_size[1]/v_size[1])
        kernel_size = int(a_size[1] - (v_size[1] - 1) * stride)
        self.netcnnaud_to_vis = nn.Sequential(
                                    nn.Conv1d(a_size[0], v_size[0] // 2, kernel_size=kernel_size, stride=stride),
                                    nn.BatchNorm1d(v_size[0] // 2),
                                    nn.ReLU(),
                                    nn.Conv1d(v_size[0] // 2, v_size[0], kernel_size=3, stride=1, padding=1)
                                )

        self.netcnnlip, self.param = select_resnet_half(network, track_running_stats=False)

        map_size = v_size[2]*v_size[3]
        self.final_fc = nn.Sequential(
            nn.Linear(map_size, 1),
            nn.Sigmoid()
        )

        out_emb = v_size[0] // 4
        self.img_emb_layer = nn.Conv3d(v_size[0], out_emb, 1)
        self.aud_emb_layer = nn.Conv1d(v_size[0], out_emb, 1)

    def forward_aud(self, x):
        (B, C) = x.shape  # batch, 48000
        x = x.view(B, 1, C)  # batch x 1 x 48000

        x = self.netcnnaud(x)  # batch x 192 x 1497
        x = self.netcnnaud_to_vis(x)  # batch x 128 x 15

        return x


    def forward_lip(self, x):
        (B, N, C, NF, H, W) = x.shape
        x = x.view(B * N, C, NF, H, W)  # batch x 3 x 30 x 224 x 224

        x = self.netcnnlip(x, return_intermediate=False)  # batch x 128 x 15 x 28 x 28
        if self.spatial_size != 28:
            x = nn.functional.adaptive_avg_pool3d(x, (15, self.spatial_size, self.spatial_size))

        return x

    def forward(self, vid_seq, aud_seq):
        vid_out = self.forward_lip(vid_seq)  # batch x 128 x 15 x 28 x 28
        aud_out = self.forward_aud(aud_seq)  # batch x 128 x 15

        aud_out = aud_out.view(aud_out.shape[0], aud_out.shape[1], aud_out.shape[2], 1, 1)  # batch x 128 x 15 x 1 x 1

        vid_aud_distance_ = torch.pow((vid_out - aud_out), 2)  # batch x 128 x 15 x 28 x 28
        vid_aud_distance_ = vid_aud_distance_.view(vid_aud_distance_.shape[0], vid_aud_distance_.shape[1]*vid_aud_distance_.shape[2], vid_aud_distance_.shape[3]*vid_aud_distance_.shape[4])   # batch x 1920 x 784
        vid_aud_distance_ = torch.sqrt(torch.sum(vid_aud_distance_, dim=1))  # batch x 784

        if self.with_attention:
            img_emb = self.img_emb_layer(vid_out)  # batch x 32 x 15 x 28 x 28
            aud_emb = self.aud_emb_layer(aud_out.view(aud_out.shape[0], aud_out.shape[1], aud_out.shape[2]))  # batch x 32 x 15

            atts = []
            for i_emb, a_emb in zip(img_emb, aud_emb):
                atts.append(torch.tensordot(i_emb, a_emb, dims=([0, 1],[0, 1])))  # each, 28 x 28

            att = torch.stack(atts)  # batch x 28 x 28 
            att = att / (32 * 15)  # normalize with the C and T size

            att = att.view(att.shape[0], -1)  # batch x 784
            # relaxed softmax
            att = torch.nn.functional.softmax(att, dim=1)

            if self.residual_conn:
                vid_aud_distance_ = torch.mul(att, vid_aud_distance_) + vid_aud_distance_
            else:
                vid_aud_distance_ = torch.mul(att, vid_aud_distance_)

        else:
            att = None

        final_out = self.final_fc(vid_aud_distance_)

        return final_out, vid_aud_distance_, att

