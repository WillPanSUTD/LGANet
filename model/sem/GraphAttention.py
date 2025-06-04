import torch
import torch.nn as nn
from lib.pointops.functions import pointops


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16, block_idx=0):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.block_idx = block_idx
        self.linear_q1 = nn.Linear(in_planes, mid_planes)
        self.linear_k1 = nn.Linear(in_planes, mid_planes)
        self.linear_v1 = nn.Linear(in_planes, out_planes)
        self.linear_p1 = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w1 = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True), nn.Linear(mid_planes, mid_planes // share_planes),
                                       nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                       nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pnxo) -> torch.Tensor:
        p, n, x, o = pnxo  # (n, 3), (n, c), (b)
        x_q1 = self.linear_q1(x)
        # x_edge1, distance1 = pointops.queryandgroup_nor(self.nsample, p, p, n, x, None, o, o, edge=True)  # (n, nsample, in_planes)
        x_edge1, distance1 = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, edge=True)  # (n, nsample, c)
        p_r1, x_e1 = x_edge1[:, :, 0:3], x_edge1[:, :, 3:]
        x_k1 = self.linear_k1(x_e1); x_v1 = self.linear_v1(x_e1)
        distance_weight1 = torch.exp(-distance1.unsqueeze(-1))
        for i, layer in enumerate(self.linear_p1): p_r1 = layer(p_r1.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r1)  # (n, nsample, c)
        w1 = x_q1.unsqueeze(1) - x_k1 + p_r1  # (n, nsample, c)
        for i, layer in enumerate(self.linear_w1): w1 = layer(w1.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w1)
        w1 = self.softmax(w1)  # (n, nsample, c)
        num1, nsample1, c1 = x_v1.shape; s = self.share_planes
        x_v1 = x_v1 * distance_weight1
        x1 = ((x_v1 + p_r1).view(num1, nsample1, s, c1 // s) * w1.unsqueeze(2)).sum(1).view(num1, c1)
        # x1 = ((x_v1 + p_r1) * w1).sum(1).view(num1, c1)

        return x1


class EdgeConv(nn.Module):
    def __init__(self, in_planes, out_planes, nsample):
        super().__init__()
        self.nsample = nsample
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes * 2, out_planes, kernel_size=1, bias=False), nn.BatchNorm2d(out_planes), self.relu)
        self.conv2 = nn.Sequential(nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False), nn.BatchNorm2d(out_planes), self.relu)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))

    def forward(self, pnxo):
        p, n, x, o = pnxo
        x_edge, _ = pointops.queryandgroup_nor(self.nsample, p, p, n, x, None, o, o, edge=True)  # (n, nsample, in_planes)
        n_r, x_e = x_edge[:, :, 0:3], x_edge[:, :, 3:]
        for i, layer in enumerate(self.linear_p): n_r = layer(n_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(n_r)
        # x = torch.cat((x.unsqueeze(1).repeat(1, self.nsample, 1), x_e), dim=-1) + n_r  # (num, nsample, in_planes*2)
        x = torch.cat((x.unsqueeze(1).repeat(1, self.nsample, 1), x_e), dim=-1)  # (num, nsample, in_planes*2)
        x = x.permute(2, 0, 1).unsqueeze(0)  # (num, nsample, in_planes*2) -> (1, in_planes*2, num, nsample)
        n_r = n_r.permute(2, 0, 1).unsqueeze(0).contiguous()
        x = self.conv1(x)  # (1, in_planes*2, num, nsample) -> (1, out_planes, num, nsample)
        x = self.conv2(x) + n_r  # (1, out_planes, num, nsample) -> (1, out_planes, num, nsample)
        x = x.max(dim=-1, keepdim=False)[0].squeeze(0).transpose(0, 1).contiguous()  # (1, out_planes, num, nsample) -> (num, out_planes)

        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pnxo):
        p, n, x, o = pnxo  # (n, 3), (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.tensor(n_o, dtype=torch.int, device='cuda')
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p, n_n = p[idx.long(), :], n[idx.long(), :]
            x, _ = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  # (m, 3+c, nsample)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, n, o = n_p, n_n, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, n, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        self.in_planes = in_planes
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

    def forward(self, pnxo1, pnxo2=None):
        if pnxo2 is None:
            _, _, x, o = pnxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, _, x1, o1 = pnxo1
            p2, _, x2, o2 = pnxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class GraphAttentionBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, block_idx=0):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = GraphAttentionLayer(planes, planes, share_planes, nsample, block_idx)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, pnxo):
        p, n, x, o = pnxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, n, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, n, x, o]


class GraphConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, nsample):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.edgeConv = EdgeConv(in_planes, out_planes, nsample)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.linear = nn.Linear(out_planes, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pnxo):
        p, n, x, o = pnxo
        identity = x
        x = self.relu(self.bn1(self.edgeConv([p, n, x, o])))
        x = self.bn2(self.linear(x))
        x = self.relu(x + identity)
        return [p, n, x, o]


class GraphAttentionSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13):
        super().__init__()
        self.c = c
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        share_planes = 8
        stride, nsample, block_idx = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0], block_idx=block_idx[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1], block_idx=block_idx[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2], block_idx=block_idx[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3], block_idx=block_idx[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4], block_idx=block_idx[4])  # N/256
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True, block_idx=block_idx[5])  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3], block_idx=block_idx[6])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2], block_idx=block_idx[7])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1], block_idx=block_idx[8])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0], block_idx=block_idx[9])  # fusion p2 and p1
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], k))

    def _make_enc(self, block, planes, blocks, share_planes=4, stride=1, nsample=16, block_idx=0):
        layers = [TransitionDown(self.in_planes, planes, stride, nsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample, block_idx=block_idx))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=4, nsample=16, is_head=False, block_idx=0):
        layers = [TransitionUp(self.in_planes, None if is_head else planes)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(GraphConvBlock(self.in_planes, self.in_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pno):
        p0, n0, o0 = pno
        x0 = p0 if self.c == 3 else torch.cat((p0, n0), 1)
        p1, n1, x1, o1 = self.enc1([p0, n0, x0, o0])
        p2, n2, x2, o2 = self.enc2([p1, n1, x1, o1])
        p3, n3, x3, o3 = self.enc3([p2, n2, x2, o2])
        p4, n4, x4, o4 = self.enc4([p3, n3, x3, o3])
        p5, n5, x5, o5 = self.enc5([p4, n4, x4, o4])
        x5 = self.dec5[1:]([p5, n5, self.dec5[0]([p5, n5, x5, o5]), o5])[2]
        x4 = self.dec4[1:]([p4, n4, self.dec4[0]([p4, n4, x4, o4], [p5, n5, x5, o5]), o4])[2]
        x3 = self.dec3[1:]([p3, n3, self.dec3[0]([p3, n3, x3, o3], [p4, n4, x4, o4]), o3])[2]
        x2 = self.dec2[1:]([p2, n2, self.dec2[0]([p2, n2, x2, o2], [p3, n3, x3, o3]), o2])[2]
        x1 = self.dec1[1:]([p1, n1, self.dec1[0]([p1, n1, x1, o1], [p2, n2, x2, o2]), o1])[2]
        x = self.cls(x1)
        return x


def graphAttention_seg_repro(**kwargs):
    model = GraphAttentionSeg(GraphAttentionBlock, [2, 3, 4, 6, 3], **kwargs)  # [2, 3, 4, 6, 3]
    return model
