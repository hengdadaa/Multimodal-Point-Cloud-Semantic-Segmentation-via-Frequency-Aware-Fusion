import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionMacthcing(nn.Module):
    def __init__(self, feature_dim=512, seq_len=5000):
        super(AttentionMacthcing, self).__init__()
        self.fc_spt = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_qry = nn.Sequential(
            nn.Linear(seq_len, seq_len // 10),
            nn.ReLU(),
            nn.Linear(seq_len // 10, seq_len),
        )
        self.fc_fusion = nn.Sequential(
            nn.Linear(seq_len , seq_len // 5),
            nn.ReLU(),
            nn.Linear(seq_len // 5,  seq_len),
        )
        self.sigmoid = nn.Sigmoid()

    def correlation_matrix(self, spt_fg_fts, qry_fg_fts):
        """
        Calculates the correlation matrix between the spatial foreground features and query foreground features.

        Args:
            spt_fg_fts (torch.Tensor): The spatial foreground features.
            qry_fg_fts (torch.Tensor): The query foreground features.

        Returns:
            torch.Tensor: The cosine similarity matrix. Shape: [1, 1, N].
        """
        spt_fg_fts = F.normalize(spt_fg_fts, p=2, dim=1)  # shape [1, 512, 900]
        qry_fg_fts = F.normalize(qry_fg_fts, p=2, dim=1)  # shape [1, 512, 900]

        cosine_similarity = torch.sum(spt_fg_fts * qry_fg_fts, dim=1, keepdim=True)  # shape: [1, 1, N]

        return cosine_similarity

    def forward(self, spt_fg_fts, qry_fg_fts, band):
        """
        Args:
            spt_fg_fts (torch.Tensor): Spatial foreground features.
            qry_fg_fts (torch.Tensor): Query foreground features.
            band (str): Band type, either 'low', 'high', or other.

        Returns:
            torch.Tensor: Fused tensor. Shape: [1, 512, 5000].
        """

        spt_proj = F.relu(self.fc_spt(spt_fg_fts))  # shape: [1, 512, 900]
        qry_proj = F.relu(self.fc_qry(qry_fg_fts))  # shape: [1, 512, 900]

        similarity_matrix = self.sigmoid(self.correlation_matrix(spt_fg_fts, qry_fg_fts))

        if band == 'low' or band == 'high':
            weighted_spt = (1 - similarity_matrix) * spt_proj  # shape: [1, 512, 900]
            weighted_qry = (1 - similarity_matrix) * qry_proj  # shape: [1, 512, 900]
        else:
            weighted_spt = similarity_matrix * spt_proj  # shape: [1, 512, 900]
            weighted_qry = similarity_matrix * qry_proj  # shape: [1, 512, 900]

        # 调试输出拼接前后的维度，确保它们不会不合适地扩大
        #print(f"Weighted spt shape before concatenation: {weighted_spt.shape}")
        #print(f"Weighted qry shape before concatenation: {weighted_qry.shape}")

        combined = weighted_spt + weighted_qry  # shape: [1, 1024, 900]
        #print(f"Combined shape after concatenation: {combined.shape}")

        fused_tensor = F.relu(self.fc_fusion(combined))  # shape: [1, 512, 900]
       # print(fused_tensor.shape)

        return fused_tensor



class FAM(nn.Module):
    def __init__(self, feature_dim=784, N=None):
        super(FAM, self).__init__()
        # 判断设备是否可用，选择 CPU 或 CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 定义AttentionMatching模块
        self.attention_matching = AttentionMacthcing(feature_dim, N)
        # 定义自适应平均池化模块
        self.adapt_pooling = nn.AdaptiveAvgPool1d(N)

    def forward(self, spt_fg_fts, qry_fg_fts):
        """
        FAM模块的前向传播

        Args:
            spt_fg_fts (torch.Tensor): 支持样本特征，形状为 [B, C, N]
            qry_fg_fts (torch.Tensor): 查询样本特征，形状为 [B, C, N]

        Returns:
            torch.Tensor: 融合后的特征
        """
        # 将输入张量转换为模块期望的格式
        spt_fg_fts = [[spt_fg_fts]]
        qry_fg_fts = [qry_fg_fts]

        # 应用自适应池化，将每个特征池化到N维
        spt_fg_fts = [[self.adapt_pooling(fts) for fts in way] for way in spt_fg_fts]
        qry_fg_fts = [self.adapt_pooling(fts) for fts in qry_fg_fts]

        # 获取频率带（低频，中频和高频）
        spt_fg_fts_low, spt_fg_fts_mid, spt_fg_fts_high = self.filter_frequency_bands(spt_fg_fts[0][0], cutoff=0.30)
        qry_fg_fts_low, qry_fg_fts_mid, qry_fg_fts_high = self.filter_frequency_bands(qry_fg_fts[0], cutoff=0.30)

        # 进行低频、中频和高频的注意力匹配
        fused_fts_low = self.attention_matching(spt_fg_fts_low, qry_fg_fts_low, 'low')
        fused_fts_mid = self.attention_matching(spt_fg_fts_mid, qry_fg_fts_mid, 'mid')
        fused_fts_high = self.attention_matching(spt_fg_fts_high, qry_fg_fts_high, 'high')

        # 返回融合后的特征
        #print("1famstart1111111111")
        return fused_fts_low + fused_fts_mid + fused_fts_high

    def reshape_to_square(self, tensor):
        """
        将输入的张量重塑为方形形状

        Args:
            tensor (torch.Tensor): 输入张量，形状为 (B, C, N)

        Returns:
            tuple: 包含以下内容：
                - square_tensor (torch.Tensor): 变形后的方形张量，形状为 (B, C, side_length, side_length)
                - side_length (int): 方形张量的边长
                - N (int): 输入张量中的元素数
        """
        B, C, N = tensor.shape
        side_length = int(np.ceil(np.sqrt(N)))  # 计算方形的边长
        padded_length = side_length ** 2

        # 创建一个零填充的张量
        padded_tensor = torch.zeros((B, C, padded_length), device=tensor.device)
        padded_tensor[:, :, :N] = tensor

        # 将张量变形为方形
        square_tensor = padded_tensor.view(B, C, side_length, side_length)

        return square_tensor, side_length, side_length, N

    def filter_frequency_bands(self, tensor, cutoff=0.2):
        """
        将输入的张量过滤为低频、中频和高频带

        Args:
            tensor (torch.Tensor): 输入张量，形状为 [B, C, H * W]
            cutoff (float): 用于切分频带的截止频率比例

        Returns:
            tuple: 包含三个频带（低频、中频、高频）的张量
        """
        device = tensor.device

        tensor = tensor.float()
        tensor, H, W, N = self.reshape_to_square(tensor)
        B, C, _, _ = tensor.shape

        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)  # 计算最大半径
        low_cutoff = max_radius * cutoff
        high_cutoff = max_radius * (1 - cutoff)

        # 对输入张量进行FFT变换
        fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))

        def create_filter(shape, low_cutoff, high_cutoff, mode='band', device=device):
            rows, cols = shape
            center_row, center_col = rows // 2, cols // 2

            y, x = torch.meshgrid(
                torch.arange(rows, device=device),
                torch.arange(cols, device=device)
            )
            distance = torch.sqrt((y - center_row) ** 2 + (x - center_col) ** 2)

            mask = torch.zeros((rows, cols), dtype=torch.float32, device=device)

            if mode == 'low':
                mask[distance <= low_cutoff] = 1
            elif mode == 'high':
                mask[distance >= high_cutoff] = 1
            elif mode == 'band':
                mask[(distance > low_cutoff) & (distance < high_cutoff)] = 1

            return mask

        # 创建低频、中频和高频的滤波器
        low_pass_filter = create_filter((H, W), low_cutoff, None, mode='low')[None, None, :, :]
        high_pass_filter = create_filter((H, W), None, high_cutoff, mode='high')[None, None, :, :]
        mid_pass_filter = create_filter((H, W), low_cutoff, high_cutoff, mode='band')[None, None, :, :]

        # 对FFT结果应用滤波器
        low_freq_fft = fft_tensor * low_pass_filter
        high_freq_fft = fft_tensor * high_pass_filter
        mid_freq_fft = fft_tensor * mid_pass_filter

        # 对滤波后的频谱进行逆FFT
        low_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(low_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        high_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real
        mid_freq_tensor = torch.fft.ifft2(torch.fft.ifftshift(mid_freq_fft, dim=(-2, -1)), dim=(-2, -1)).real

        # 将结果调整为适当的形状
        low_freq_tensor = low_freq_tensor.view(B, C, H * W)[:, :, :N]
        high_freq_tensor = high_freq_tensor.view(B, C, H * W)[:, :, :N]
        mid_freq_tensor = mid_freq_tensor.view(B, C, H * W)[:, :, :N]

        return low_freq_tensor, mid_freq_tensor, high_freq_tensor


if __name__ == '__main__':
    batch_size = 1
    feature_dim = 64
    num_elements = 2222

    input_tensor = torch.rand(batch_size, feature_dim, num_elements).to('cuda')

    block = FAM(feature_dim, num_elements).to('cuda')

    spt_fg_fts = torch.rand(batch_size, feature_dim, num_elements).to('cuda')
    qry_fg_fts = torch.rand(batch_size, feature_dim, num_elements).to('cuda')

    fused_fts_low = block(spt_fg_fts, qry_fg_fts)
    #fused_fts_low=fused_fts_low.view(  -1,64)

    print(f"Fused Low Frequency Features: {fused_fts_low.size()}")