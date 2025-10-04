# ======= SE Block for CNN =======
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class TransformerSEBlock(nn.Module):
    def __init__(self, embed_dim, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(embed_dim // reduction, embed_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b = x.shape[1]
        c = x.shape[2]
        y = x.mean(dim=0)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.unsqueeze(0).expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_se=False, groups=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CustomResNet(nn.Module):  
    def __init__(self, block, layers, groups=1):
        super().__init__()
        self.in_channels = 12
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 12, layers[0], stride=1, groups=groups, use_se=True) 
        self.layer2 = self._make_layer(block, 24, layers[1], stride=2, groups=groups, use_se=True)
        self.layer3 = self._make_layer(block, 24, layers[2], stride=2, groups=groups, use_se=True)
        self.layer4 = self._make_layer(block, 32, layers[3], stride=2, groups=groups, use_se=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, blocks, stride=1, groups=1, use_se=True):  
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, use_se=use_se, groups=groups))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_se=use_se, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

# ======= DecoupledRadarAttention ======

class DecoupledTriadAttention(nn.Module):
    def __init__(self, dim, num_heads, grid_dims, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.H_grid, self.W_grid = grid_dims
        self.num_patches = self.H_grid * self.W_grid

        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim * 3, dim)

        self.channel_num_heads = 4
        assert dim % self.channel_num_heads == 0, "dim should be divisible by channel_num_heads"
        self.channel_head_dim = dim // self.channel_num_heads
        self.channel_qkv = nn.Linear(dim, dim * 3)
        self.channel_attn_drop = nn.Dropout(dropout)
        self.channel_proj = nn.Linear(dim, dim)
        self.channel_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [785, B, 20]
        N, B, C = x.shape
        cls_token = x[0:1]  # [1, B, 20]
        patch_embeds = x[1:]  # [784, B, 20]

        # Reshape patch embedding
        x_reshaped = patch_embeds.permute(1, 0, 2).reshape(B, self.H_grid, self.W_grid, C)  # [B, 28, 28, 20]

        # Temporal Attention (WTA)
        qkv = self.qkv_linear(x).reshape(N, B, 3, self.num_heads, self.head_dim).permute(2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        q_cls, q_patch = q[:, :, 0:1], q[:, :, 1:]  # [B, num_heads, 1, head_dim], [B, num_heads, 784, head_dim]
        k_cls, k_patch = k[:, :, 0:1], k[:, :, 1:]  # [B, num_heads, 1, head_dim], [B, num_heads, 784, head_dim]
        v_cls, v_patch = v[:, :, 0:1], v[:, :, 1:]  # [B, num_heads, 1, head_dim], [B, num_heads, 784, head_dim]

        # Reshape patch embedding for WTA
        q_t = q_patch.reshape(B * self.H_grid, self.num_heads, self.W_grid, self.head_dim)  # [B * 28, 4, 28, 5]
        k_t = k_patch.reshape(B * self.H_grid, self.num_heads, self.W_grid, self.head_dim)
        v_t = v_patch.reshape(B * self.H_grid, self.num_heads, self.W_grid, self.head_dim)
        temporal_attn = (q_t @ k_t.transpose(-2, -1)) * self.scale
        temporal_attn = temporal_attn.softmax(dim=-1)
        temporal_attn = self.attn_drop(temporal_attn)
        temporal_output_patch = (temporal_attn @ v_t).reshape(B, self.H_grid, self.num_heads, self.W_grid, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B, self.H_grid, self.W_grid, C)

        # Freq/Spatial Attention (HFA)
        q_fs = q_patch.permute(0, 1, 3, 2).reshape(B * self.W_grid, self.num_heads, self.H_grid, self.head_dim)
        k_fs = k_patch.permute(0, 1, 3, 2).reshape(B * self.W_grid, self.num_heads, self.H_grid, self.head_dim)
        v_fs = v_patch.permute(0, 1, 3, 2).reshape(B * self.W_grid, self.num_heads, self.H_grid, self.head_dim)
        freq_spatial_attn = (q_fs @ k_fs.transpose(-2, -1)) * self.scale
        freq_spatial_attn = freq_spatial_attn.softmax(dim=-1)
        freq_spatial_attn = self.attn_drop(freq_spatial_attn)
        freq_spatial_output_patch = (freq_spatial_attn @ v_fs).reshape(B, self.W_grid, self.num_heads, self.H_grid, self.head_dim).permute(0, 3, 1, 2, 4).reshape(B, self.H_grid, self.W_grid, C)

        # Channel Attention with Multi-Head Attention
        channel_input = x_reshaped.mean(dim=[1,2])
        channel_input = channel_input.unsqueeze(1)
        channel_qkv = self.channel_qkv(channel_input).reshape(B, 1, 3, self.channel_num_heads, self.channel_head_dim).permute(2, 0, 3, 1, 4)
        channel_q, channel_k, channel_v = channel_qkv[0], channel_qkv[1], channel_qkv[2]
        channel_attn = (channel_q @ channel_k.transpose(-2, -1)) * (self.channel_head_dim ** -0.5)
        channel_attn = channel_attn.softmax(dim=-1)
        channel_attn = self.channel_attn_drop(channel_attn)
        channel_output = (channel_attn @ channel_v).reshape(B, 1, C)
        channel_output = self.channel_proj(channel_output)
        channel_output = self.channel_sigmoid(channel_output)
        channel_output_raw = x_reshaped * channel_output.unsqueeze(2)

        combined_output_reshaped = torch.cat((temporal_output_patch, freq_spatial_output_patch, channel_output_raw), dim=3)  # [B, 28, 28, 60]
        combined_output = combined_output_reshaped.reshape(B, N-1, C * 3).permute(1, 0, 2)  # [784, B, 60]
        combined_output = self.proj(combined_output)  # [784, B, 20]

        combined_output = torch.cat((cls_token, combined_output), dim=0)  # [785, B, 20]

        return combined_output

class TransformerEncoder(nn.Module):
    def __init__(self, dim=20, num_heads=4, mlp_dim=40, grid_dims=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DecoupledTriadAttention(dim, num_heads, grid_dims=grid_dims, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_output = self.attn(h)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x
# ======= Patch Embedding =======
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_chans=3, embed_dim=20):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.H_grid = img_size // patch_size
        self.W_grid = img_size // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Class token

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, 784, 20]
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 20]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 785, 20]
        x = x + self.pos_embed
        return x.transpose(0, 1)  # [785, B, 20]

# ======= HybridRadarClassifier =======
class DTANet(nn.Module):
    def __init__(self, num_classes=12, groups=1, img_size=224, patch_size=8):
        super().__init__()
        H_grid = img_size // patch_size
        W_grid = img_size // patch_size
        self.resnet_branch = CustomResNet(BasicBlock, [1, 1, 1, 1], groups=groups)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=20)
        self.transformer = nn.Sequential(
            TransformerEncoder(dim=20, num_heads=4, mlp_dim=40, grid_dims=(H_grid, W_grid)),
            TransformerEncoder(dim=20, num_heads=4, mlp_dim=40, grid_dims=(H_grid, W_grid))
        )
        self.transformer_se = TransformerSEBlock(embed_dim=20, reduction=4)
        combined_dim = 32 + 20
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(20, num_classes)
        )

    def forward(self, x):
        feat_cnn = self.resnet_branch(x)
        x_patch = self.patch_embed(x)  # [785, B, 20]
        x_trans = self.transformer(x_patch)  # [785, B, 20]
        x_trans = self.transformer_se(x_trans)  # [785, B, 20]
        x_trans = x_trans[0]  
        combined = torch.cat([feat_cnn, x_trans], dim=1)
        return self.classifier(combined)
    
num_classes = 12
img_size = 224


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DTANet(num_classes=num_classes, groups=1, img_size=img_size, patch_size=8).to(device)
