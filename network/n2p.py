import torch, math
from torch import nn
from einops import rearrange
import torch.nn.functional as F


def compute_feature_correlation(center, neighbors):
    """
    计算中心点与邻居之间的余弦相似度
    :param center: (B, 1, C)
    :param neighbors: (B, 1, K, C)
    :return: (B, 1, K)，范围 [-1, 1]
    """
    center_norm = F.normalize(center, dim=-1)              # (B, 1, C)
    neighbors_norm = F.normalize(neighbors, dim=-1)        # (B, 1, K, C)
    center_expanded = center_norm.unsqueeze(2)             # (B, 1, 1, C)
    neighbors_transposed = neighbors_norm.permute(0, 1, 3, 2)  # (B, 1, C, K)
    correlation = torch.matmul(center_expanded, neighbors_transposed)  # (B, 1, 1, K)
    return correlation.squeeze(2)  # (B, 1, K)


def single_point_edge_detection(pcd, center_idx, K=8, use_softmax=False):
    """
    只处理一个点的边缘检测逻辑
    :param pcd: (B, C, N)
    :param center_idx: 要分析的点索引
    :param K: 邻居数量
    :param use_softmax: 是否对相关性做 softmax 归一化
    :return: std, correlation_values
    """
    B, C, N = pcd.shape
    pcd_t = pcd.permute(0, 2, 1)  # (B, N, C)

    # 1. 提取中心点 (B, 1, C)
    center = pcd_t[:, center_idx:center_idx+1, :]

    # 2. 计算距离 & 选择 K 个最近邻 (B, 1, K)
    dists = torch.cdist(center, pcd_t)  # (B, 1, N)
    knn_idx = dists.topk(K, largest=False)[1]  # (B, 1, K)

    # 3. 获取邻居 (B, 1, K, C)
    neighbors = index_points(pcd_t, knn_idx)  # (B, 1, K, C)

    # 4. 计算原始相关性 (B, 1, K)
    corr = compute_feature_correlation(center, neighbors)

    if use_softmax:
        corr = torch.softmax(corr, dim=-1)

    # 5. 标准差 (B, 1)
    std_dev = torch.std(corr, dim=-1)

    return std_dev.squeeze(1), corr.squeeze(1)  # (B,), (B, K)






##################3

def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a ** 2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b ** 2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx


def select_neighbors(pcd, K, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
    if neighbor_type == 'neighbor':
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors


def group(pcd, K, group_type):
    if group_type == 'neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')  # neighbors.shape == (B, C, N, K)
        output = neighbors  # output.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')  # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output.contiguous()


class N2PAttention(nn.Module):
    def __init__(self):
        super(N2PAttention, self).__init__()
        self.heads = 4
        self.K = 32
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.k_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.v_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(64, 512, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(512, 64, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=128, stride=128)
        self.conv5 = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=128, stride=128)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)

    def forward(self, x):

        x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        p = x
        #print(x.shape)
        x = self.conv(x)
        #x1 = self.conv1(x)
        #print(x1.shape)
        #std_dev, edge_pixels = single_point_edge_detection(x1, 9)
        #torch.set_printoptions(threshold=10000)  # 或 torch.inf
        #print(std_dev,edge_pixels)
        neighbors = group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k,
                               'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention @ v,
                        'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)

        x = self.conv5(x)
        if x.shape[2] < p.shape[2]:
            pad_size = p.shape[2] - x.shape[2]
            x = nn.functional.pad(x, (0, pad_size))  # 如果不够原始尺寸 肯定填0
        # 如果 restored_x 的尺寸大于原始尺寸，截断它
        elif x.shape[2] > p.shape[2]:
            x = x[:, :, :p.shape[2]]  # 如果比原始尺寸大 截断他


        x = x.squeeze(0).transpose(0, 1)

        #print("n2pstart22222222222222222")


        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x


if __name__ == '__main__':
    # 批次
    num_points = 375595  # 点云的点数量
    channel_size = 64  # 输入的通道数

    block = N2PAttention()

    input_tensor = torch.rand(num_points, channel_size)  # (B, C, N)


    output = block(input_tensor)

    print("输入的形状:", input_tensor.size())
    print("输出的形状:",output)