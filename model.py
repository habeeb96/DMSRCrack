import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from math import gcd

# --- Helper Blocks ---
class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    def forward(self, x): return self.act(self.norm(self.conv(x)))

class nnUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ConvNormAct(out_channels, out_channels, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
    def forward(self, x): return self.conv2(self.conv1(x)) + self.shortcut(x)

class LKABlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=21):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(out_channels)
        self.convLKA = nn.Conv2d(out_channels, out_channels, kernel_size, 1, kernel_size//2, groups=out_channels, bias=False); self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False); self.bn3 = nn.BatchNorm2d(out_channels); self.pw = nn.Conv2d(out_channels, out_channels, 1, bias=False)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); u = x
        x = F.relu(self.bn2(self.convLKA(x))); a = F.relu(self.bn3(self.conv2(x)))
        return self.pw(a) * u

class AttentionModule(nn.Module):
    def __init__(self, in_channels, cross_channels=0, use_depthwise_spatial=True):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        dim = in_channels*2 + cross_channels*2
        self.shared_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, groups=gcd(dim, in_channels*2)), nn.ReLU(), nn.Conv2d(dim, in_channels, 1))
        self.sigmoid = nn.Sigmoid(); self.use_depthwise_spatial = use_depthwise_spatial
        if self.use_depthwise_spatial: self.spatial_conv_dw = nn.Conv2d(2, 2, 3, 1, 1, groups=2); self.spatial_reduce = nn.Conv2d(2, 1, 1)
        else: self.spatial_conv = nn.Conv2d(2, 1, 3, 1, 2, dilation=2)
    def forward(self, x, cross_input=None):
        pool = torch.cat([self.avg_pool(x), self.max_pool(x)], 1)
        if cross_input is not None: pool = torch.cat([pool, self.avg_pool(cross_input), self.max_pool(cross_input)], 1)
        cw = self.sigmoid(self.shared_conv(pool))
        sp_in = torch.cat([x.mean(1,True), x.max(1,True)[0]], 1)
        sw = self.spatial_reduce(self.spatial_conv_dw(sp_in)) if self.use_depthwise_spatial else self.spatial_conv(sp_in)
        return x * cw * self.sigmoid(sw), self.sigmoid(sw)

class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_attention=True, cross_ch=0):
        super().__init__()
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, 1, 1); self.convLKA = LKABlock(in_ch, out_ch); self.bn = nn.BatchNorm2d(out_ch*2)
        self.attn = AttentionModule(out_ch*2, cross_ch) if use_attention else nn.Identity()
        self.sc = nn.Conv2d(in_ch, out_ch*2, 1) if in_ch != out_ch*2 else nn.Identity()
        self.cross = nn.Conv2d(cross_ch, out_ch*2, 1) if cross_ch > 0 else nn.Identity()
    def forward(self, x, cross_input=None):
        out = self.bn(torch.cat([F.relu(self.conv3(x)), F.relu(self.convLKA(x))], 1))
        if cross_input is not None:
            if isinstance(self.attn, nn.Identity): out = self.bn(out + F.interpolate(self.cross(cross_input), size=out.shape[2:], mode='bilinear', align_corners=False))
            else: out, _ = self.attn(out, cross_input)
        else:
            if not isinstance(self.attn, nn.Identity): out, _ = self.attn(out)
        return F.relu(out + self.sc(x)), None

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Linear(int(embed_dim * mlp_ratio), embed_dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.mlp(self.norm2(x))

class TinyViT(nn.Module):
    def __init__(self, in_ch=3, patch_size=16, embed_dim=96, num_layers=4, num_heads=2, mlp_ratio=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, 1, 1) * 0.02)
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)])
        self.down1 = nn.Conv2d(embed_dim, embed_dim, 2, 2); self.patch_size = patch_size
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x) + F.interpolate(self.pos_embed, size=(x.shape[2]//self.patch_size, x.shape[3]//self.patch_size), mode='bilinear', align_corners=False)
        xf = x.flatten(2).transpose(1, 2)
        for layer in self.layers: xf = layer(xf)
        return [self.down1(xf.transpose(1, 2).view(B, -1, H // self.patch_size, W // self.patch_size))]

class AttentionModuleFusion(nn.Module):
    def __init__(self, in_ch, use_depthwise_spatial=True):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_conv = nn.Sequential(nn.Conv2d(in_ch * 2, in_ch * 2, 1, groups=gcd(in_ch * 2, in_ch * 2)), nn.ReLU(), nn.Conv2d(in_ch * 2, in_ch, 1))
        self.sigmoid = nn.Sigmoid()
        if use_depthwise_spatial: self.sdw = nn.Conv2d(2, 2, 3, padding=1, groups=2); self.srd = nn.Conv2d(2, 1, 1)
        else: self.sp = nn.Conv2d(2, 1, 3, padding=2, dilation=2)
    def forward(self, x):
        channel = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        channel_weights = self.sigmoid(self.shared_conv(channel))
        spatial = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        sp_w = self.srd(self.sdw(spatial)) if hasattr(self, 'sdw') else self.sigmoid(self.sp(spatial))
        return x * channel_weights * sp_w, sp_w

class MultiScaleFusion(nn.Module):
    def __init__(self, sam_ch=[128, 256, 512], out_ch=512, vit_ch=96, use_attention=True):
        super().__init__()
        self.sam_convs = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in sam_ch])
        self.vit_convs = nn.ModuleList([nn.Conv2d(vit_ch, out_ch, 1) for _ in range(1)])
        self.attn = AttentionModuleFusion(out_ch) if use_attention else nn.Identity()
    def forward(self, sam_features, vit_features):
        B, _, H, W = sam_features[-1].shape; fused = 0
        for sf, conv in zip(sam_features, self.sam_convs): fused += conv(F.interpolate(sf, size=(H, W), mode='bilinear', align_corners=False))
        for vf, conv in zip(vit_features, self.vit_convs): fused += conv(F.interpolate(vf, size=(H, W), mode='bilinear', align_corners=False))
        if isinstance(self.attn, nn.Identity): return fused, None, None
        fused, sp_w = self.attn(fused)
        return fused, sam_features[0], sp_w

class AGCF(nn.Module):
    def __init__(self, in_ch=256, out_ch=256):
        super().__init__()
        self.sam_ctx = nn.Conv2d(in_ch, in_ch, 3, 1, 1); self.vit_ctx = nn.Conv2d(in_ch, in_ch, 1)
        self.gate_mlp = nn.Sequential(nn.Conv2d(in_ch*2, in_ch//4, 1), nn.ReLU(), nn.Conv2d(in_ch//4, in_ch, 1), nn.Sigmoid())
        self.res_conv = nn.Conv2d(in_ch*2, in_ch, 1); self.bn = nn.BatchNorm2d(out_ch); self.final = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
    def forward(self, sam_f, vit_f): return self.final(self.bn(self.res_conv(torch.cat([sam_f, vit_f], 1))))

class EfficientCrossModalAttention(nn.Module):
    def __init__(self, channels, patch_size=16, reduction=8):
        super().__init__()
        self.qkv = nn.Conv2d(channels, channels // reduction * 3, 1); self.proj = nn.Conv2d(channels // reduction, channels, 1); self.reduction = reduction
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, 1)
        q = q.flatten(2).transpose(1, 2); k = k.flatten(2).transpose(1, 2); v = v.flatten(2).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) / (C // self.reduction)**0.5, dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, -1, H, W)) + x

class TokenMixerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Conv2d(dim, dim * 4, 1), nn.GELU(), nn.Conv2d(dim * 4, dim * 4, 3, 1, 1, groups=dim * 4), nn.GELU(), nn.Conv2d(dim * 4, dim, 1))
    def forward(self, x): return self.mlp(self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)) + x

class DF_RFEM_2(nn.Module):
    def __init__(self, in_ch, skip_ch, embed_dim=64, guide_scale=16.0):
        super().__init__()
        self.cross_attn = EfficientCrossModalAttention(in_ch); self.edge_conv = nn.Conv2d(1, skip_ch, 3, 1, 1, bias=False)
        self.thick_conv = nn.Sequential(nn.Conv2d(skip_ch, 1, 1), nn.Sigmoid())
        self.offset = nn.Conv2d(skip_ch, 18, 3, 1, 1); self.deform = DeformConv2d(skip_ch, skip_ch, 3, 1, 1)
        self.token_mix = TokenMixerBlock(embed_dim); self.down = nn.Conv2d(in_ch + skip_ch, embed_dim, 1); self.up = nn.Conv2d(embed_dim, in_ch, 1)
        self.guide_scale = guide_scale; self.last_offset = None
    def forward(self, x, skip):
        skip_u = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x2 = self.cross_attn(x); edge = self.edge_conv(skip_u.mean(1, True))
        off = self.offset(edge) * (1 + self.guide_scale * self.thick_conv(skip_u))
        self.last_offset = off
        return self.up(self.token_mix(self.down(torch.cat([x2, self.deform(skip_u, off) + edge], 1))))

class UNetDecoder(nn.Module):
    def __init__(self, in_channels=512, skip_channels=[512, 256, 128, 64, 32], num_classes=2):
        super().__init__()
        self.dec_channels = [256, 128, 64, 32, 32]
        self.stage1_block = nnUNetBlock(in_channels + skip_channels[0], self.dec_channels[0])
        self.stage2_block = nnUNetBlock(self.dec_channels[0] + skip_channels[1], self.dec_channels[1])
        self.stage3_block = nnUNetBlock(self.dec_channels[1] + skip_channels[2], self.dec_channels[2])
        self.stage4_block = nnUNetBlock(self.dec_channels[2] + skip_channels[3], self.dec_channels[3])
        self.stage5_block = nnUNetBlock(self.dec_channels[3] + skip_channels[4], self.dec_channels[4])
        self.df_rfem3 = DF_RFEM_2(in_ch=self.dec_channels[2], skip_ch=skip_channels[2], embed_dim=64)
        self.attn2 = AttentionModule(self.dec_channels[0] + skip_channels[1])
        self.final_conv = nn.Conv2d(self.dec_channels[-1], num_classes-1, 1)
    def forward(self, fused, skip_features, input_size=(448, 448)):
        x4, skip4, skip3, skip2, skip1 = skip_features
        x = self.stage1_block(torch.cat([fused, x4], 1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x, _ = self.attn2(torch.cat([x, skip4], 1))
        x = F.interpolate(self.stage2_block(x), scale_factor=2, mode='bilinear', align_corners=False)
        x = self.df_rfem3(self.stage3_block(torch.cat([x, skip3], 1)), skip3)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(self.stage4_block(torch.cat([x, skip2], 1)), scale_factor=2, mode='bilinear', align_corners=False)
        if x.shape[2:] != input_size: x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        final = self.final_conv(self.stage5_block(torch.cat([x, skip1], 1)))
        if final.shape[2:] != input_size: final = F.interpolate(final, size=input_size, mode='bilinear', align_corners=False)
        return final

class BoundaryRefinementHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv = nn.Conv2d(4, 32, 3, 1, 1); self.bn = nn.BatchNorm2d(32)
        self.seg_conv = nn.Conv2d(32, 1, 1); self.boundary_conv = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(), nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid())
    def forward(self, seg_output, image):
        x = F.relu(self.bn(self.shared_conv(torch.cat([seg_output, image], 1))))
        return self.seg_conv(x), self.boundary_conv(x)

class DMSRCrack(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1); self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2); self.block1 = MultiScaleBlock(32, 32, False, 0)
        self.pool2 = nn.MaxPool2d(2, 2); self.block2 = MultiScaleBlock(64, 64, False, 0)
        self.pool3 = nn.MaxPool2d(2, 2); self.block3 = MultiScaleBlock(128, 128, False, 64)
        self.pool4 = nn.MaxPool2d(2, 2); self.block4 = MultiScaleBlock(256, 256, True, 128)
        self.vit = TinyViT(in_ch=in_channels)
        self.fusion = MultiScaleFusion(sam_ch=[128, 256, 512], out_ch=512, vit_ch=96)
        self.AGCF = AGCF(in_ch=256, out_ch=256)
        self.decoder = UNetDecoder(in_channels=512, skip_channels=[512, 256, 128, 64, 32], num_classes=num_classes)
        self.refine = BoundaryRefinementHead()
    def forward(self, x):
        B, C, H, W = x.shape
        input_size = (H, W)
        if input_size != (448, 448): x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False); input_size = (448, 448)
        x_sam = self.conv2(self.conv1(x)); skip1 = x_sam
        x_sam = self.pool1(x_sam); x1, _ = self.block1(x_sam); skip2 = x1
        x_sam = self.pool2(x1); x2, _ = self.block2(x_sam); skip3 = x2
        x_sam = self.pool3(x2); x3, _ = self.block3(x_sam, x1); skip4 = x3
        x_sam = self.pool4(x3); x4, _ = self.block4(x_sam, x2)
        vit_features = self.vit(x)
        sam_features = [x2, x3, x4]; skip_features = [x4, skip4, skip3, skip2, skip1]
        fused, _, _ = self.fusion(sam_features, vit_features)
        seg = self.decoder(fused, skip_features, input_size=input_size)
        ref, bnd = self.refine(seg, x)
        return ref, bnd
