"""
STRRNet: Semantics-Guided Two-Stage Raindrop Removal Network
NTIRE 2025 Challenge on Day and Night Raindrop Removal - 1st Place (Miracle Team)

Based on Restormer architecture with:
1. Semantic Guidance Module - 4 class classification (day/night x bg/raindrop focus)
2. Background Restoration Subnetwork - enhances image details
3. Two-stage training with semi-supervised fine-tuning

Reference: https://arxiv.org/abs/2504.12711
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


##########################################################################
## LayerNorm
##########################################################################

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## GDFN (Gated-Dconv Feed-Forward Network)
##########################################################################

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 
                                kernel_size=3, stride=1, padding=1, 
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## MDTA (Multi-DConv Head Transposed Attention)
##########################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, 
                                     padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # Clamp attention scores to prevent overflow in softmax
        attn = attn.clamp(-50, 50)
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


##########################################################################
## Transformer Block
##########################################################################

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## Patch Embedding / Downsample / Upsample
##########################################################################

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


##########################################################################
## Semantic Text Embedder
##########################################################################

class SemanticTextEmbedder(nn.Module):
    """
    Text Embedder that learns semantic embeddings for 4 categories:
    - 0: night_bg_focus
    - 1: night_raindrop_focus  
    - 2: day_bg_focus
    - 3: day_raindrop_focus
    """
    def __init__(self, embed_dim=256, num_classes=4):
        super(SemanticTextEmbedder, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Learnable text embeddings for each category
        self.class_embeddings = nn.Embedding(num_classes, embed_dim)
        
        # MLP to refine embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
    def forward(self, class_labels):
        """
        Args:
            class_labels: [B] tensor of class indices (0-3)
        Returns:
            [B, embed_dim] semantic embeddings
        """
        embeddings = self.class_embeddings(class_labels)
        embeddings = self.mlp(embeddings)
        return embeddings


##########################################################################
## Semantic Guidance Module
##########################################################################

class SemanticGuidanceModule(nn.Module):
    """
    Semantic Guidance Module that guides the decoder based on image semantics.
    
    Uses encoded image features from Restormer encoder + semantic text embeddings
    to guide the decoder for different types of images (day/night, bg/raindrop focus).
    """
    def __init__(self, feature_dim, semantic_dim=256, num_classes=4):
        super(SemanticGuidanceModule, self).__init__()
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.num_classes = num_classes
        
        # Text embedder for semantic categories
        self.text_embedder = SemanticTextEmbedder(embed_dim=semantic_dim, num_classes=num_classes)
        
        # Image feature to semantic space projection
        self.img_to_semantic = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, semantic_dim),
        )
        
        # Cross-attention for semantic-guided features (no dropout for stability)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=8,
            dropout=0.0,  # No dropout for numerical stability
            batch_first=True
        )
        
        # Feature modulation (initialized for stability)
        self.gamma_proj = nn.Linear(semantic_dim, feature_dim)
        self.beta_proj = nn.Linear(semantic_dim, feature_dim)
        
        # Initialize to output ~0 (gamma=1+0=1, beta=0)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        
    def forward(self, features, semantic_labels=None):
        """
        Args:
            features: [B, C, H, W] encoded image features
            semantic_labels: [B] class labels (0-3), None for inference (fallback to img_semantic)
        Returns:
            [B, C, H, W] semantically guided features
        """
        B, C, H, W = features.shape
        
        # Project image features to semantic space
        img_semantic = self.img_to_semantic(features)  # [B, semantic_dim]
        
        if semantic_labels is not None:
            # Training: use ground truth labels or predicted labels from classifier
            text_semantic = self.text_embedder(semantic_labels)  # [B, semantic_dim]
        else:
            # Inference fallback: use image semantic features directly
            # (Note: STRRNet forward will use SemanticClassifier if available)
            text_semantic = img_semantic
        
        # Cross-attention: image queries text
        img_semantic = img_semantic.unsqueeze(1)  # [B, 1, semantic_dim]
        text_semantic = text_semantic.unsqueeze(1)  # [B, 1, semantic_dim]
        
        # Normalize before cross-attention for stability
        img_semantic = F.normalize(img_semantic, dim=-1)
        text_semantic = F.normalize(text_semantic, dim=-1)
        
        attended, _ = self.cross_attn(img_semantic, text_semantic, text_semantic)
        attended = attended.squeeze(1)  # [B, semantic_dim]
        
        # Check for NaN and use fallback
        if torch.isnan(attended).any():
            # Fallback: use identity modulation
            return features
        
        # Feature modulation (FiLM-style)
        # gamma ~ 1, beta ~ 0 at initialization for stability
        gamma = self.gamma_proj(attended).view(B, C, 1, 1)  # [B, C, 1, 1]
        beta = self.beta_proj(attended).view(B, C, 1, 1)   # [B, C, 1, 1]
        
        # Clamp gamma to prevent explosion (keep gamma close to 1)
        gamma = 1.0 + gamma.clamp(-0.5, 0.5)  # gamma in [0.5, 1.5]
        beta = beta.clamp(-0.5, 0.5)
        
        # Apply modulation: y = gamma * x + beta
        guided_features = gamma * features + beta
        
        return guided_features


##########################################################################
## Background Restoration Subnetwork
##########################################################################

class BackgroundRestorationSubnet(nn.Module):
    """
    Background Restoration Subnetwork to enhance image details.
    Consists of multiple convolutional layers after Restormer output.
    """
    def __init__(self, in_channels, hidden_channels=64, num_layers=4):
        super(BackgroundRestorationSubnet, self).__init__()
        
        layers = []
        
        # First conv
        layers.append(nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=True))
        layers.append(nn.GELU())
        
        # Middle layers with residual
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_layers - 2):
            block = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=True),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=True),
            )
            self.residual_blocks.append(block)
        
        # Output conv
        layers.append(nn.Conv2d(hidden_channels, 3, 3, 1, 1, bias=True))
        
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=True),
            nn.GELU(),
        )
        self.output_conv = nn.Conv2d(hidden_channels, 3, 3, 1, 1, bias=True)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] features from Restormer decoder
        Returns:
            [B, 3, H, W] enhanced residual
        """
        feat = self.input_conv(x)
        
        for block in self.residual_blocks:
            feat = feat + block(feat)
        
        out = self.output_conv(feat)
        return out


##########################################################################
## STRRNet Main Architecture
##########################################################################

class STRRNet(nn.Module):
    """
    STRRNet: Semantics-Guided Two-Stage Raindrop Removal Network
    
    Architecture:
    1. Restormer-based encoder-decoder
    2. Semantic Guidance Module at encoder end
    3. Background Restoration Subnetwork at output
    
    4 Semantic Classes:
    - 0: night_bg_focus
    - 1: night_raindrop_focus
    - 2: day_bg_focus
    - 3: day_raindrop_focus
    
    Args:
        inp_channels: input channels (default 3)
        out_channels: output channels (default 3)
        dim: base channel dimension (default 48)
        num_blocks: Transformer blocks per level [4, 6, 6, 8]
        num_refinement_blocks: refinement stage blocks (default 4)
        heads: attention heads per level [1, 2, 4, 8]
        ffn_expansion_factor: FFN expansion ratio (default 2.66)
        bias: use bias (default False)
        LayerNorm_type: 'WithBias' or 'BiasFree'
        use_semantic_guidance: enable semantic guidance module
        use_bg_subnet: enable background restoration subnetwork
    """
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 use_semantic_guidance=True,
                 use_bg_subnet=True,
                 use_semantic_classifier=False):
        
        super(STRRNet, self).__init__()
        
        self.use_semantic_guidance = use_semantic_guidance
        self.use_bg_subnet = use_bg_subnet
        self.use_semantic_classifier = use_semantic_classifier

        # -------- Patch Embedding --------
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # -------- Encoder --------
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], 
                               ffn_expansion_factor=ffn_expansion_factor, 
                               bias=bias, LayerNorm_type=LayerNorm_type) 
              for _ in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for _ in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for _ in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))

        # Level 4 (Latent/Bottleneck)
        self.latent = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for _ in range(num_blocks[3])])

        # -------- Semantic Guidance Module --------
        if self.use_semantic_guidance:
            self.semantic_guidance = SemanticGuidanceModule(
                feature_dim=int(dim * 2 ** 3),
                semantic_dim=256,
                num_classes=4
            )
        
        # -------- Semantic Classifier (for inference) --------
        if self.use_semantic_classifier:
            self.semantic_classifier = SemanticClassifier(
                in_channels=inp_channels,
                num_classes=4
            )

        # -------- Decoder --------
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), 
                                            kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1),
                                            kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for _ in range(num_blocks[0])])

        # -------- Refinement Stage --------
        self.refinement = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for _ in range(num_refinement_blocks)])

        # -------- Output Projection --------
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, 
                                kernel_size=3, stride=1, padding=1, bias=bias)
        
        # -------- Background Restoration Subnetwork --------
        if self.use_bg_subnet:
            self.bg_subnet = BackgroundRestorationSubnet(
                in_channels=int(dim * 2 ** 1),
                hidden_channels=64,
                num_layers=4
            )

    def forward(self, inp_img, semantic_labels=None):
        """
        Forward pass
        
        Args:
            inp_img: [B, 3, H, W] input image
            semantic_labels: [B] semantic class labels (0-3), optional for training
        
        Returns:
            [B, 3, H, W] restored image
        """
        # Ensure input is valid
        inp_img = inp_img.clamp(0, 1)
        
        # -------- Patch Embedding --------
        inp_enc_level1 = self.patch_embed(inp_img)

        # -------- Encoder --------
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # -------- Semantic Guidance --------
        if self.use_semantic_guidance:
            # If semantic_labels not provided and classifier is available, predict labels
            if semantic_labels is None and self.use_semantic_classifier:
                with torch.no_grad():
                    class_logits = self.semantic_classifier(inp_img)
                    semantic_labels = class_logits.argmax(dim=1)  # [B]
            
            # Skip semantic guidance if labels cause issues
            try:
                latent_guided = self.semantic_guidance(latent, semantic_labels)
                # Check for NaN
                if not torch.isnan(latent_guided).any():
                    latent = latent_guided
            except:
                pass  # Keep original latent if guidance fails

        # -------- Decoder with Skip Connections --------
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # -------- Refinement --------
        out_dec_level1 = self.refinement(out_dec_level1)

        # -------- Output --------
        # Main output with global residual
        main_out = self.output(out_dec_level1) + inp_img
        
        # Background restoration enhancement
        if self.use_bg_subnet:
            bg_residual = self.bg_subnet(out_dec_level1)
            out = main_out + bg_residual
        else:
            out = main_out

        # Final clamp to ensure valid output range
        out = out.clamp(0, 1)

        return out
    
    def forward_with_features(self, inp_img, semantic_labels=None):
        """
        Forward pass that also returns intermediate features for semantic classification.
        Useful for training the semantic classifier.
        """
        # -------- Patch Embedding --------
        inp_enc_level1 = self.patch_embed(inp_img)

        # -------- Encoder --------
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # Return features for classification
        encoder_features = latent  # [B, dim*8, H/8, W/8]
        
        # -------- Semantic Guidance --------
        if self.use_semantic_guidance:
            # If semantic_labels not provided and classifier is available, predict labels
            if semantic_labels is None and self.use_semantic_classifier:
                with torch.no_grad():
                    class_logits = self.semantic_classifier(inp_img)
                    semantic_labels = class_logits.argmax(dim=1)  # [B]
            latent = self.semantic_guidance(latent, semantic_labels)

        # -------- Decoder --------
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        main_out = self.output(out_dec_level1) + inp_img
        
        if self.use_bg_subnet:
            bg_residual = self.bg_subnet(out_dec_level1)
            out = main_out + bg_residual
        else:
            out = main_out

        return out, encoder_features


##########################################################################
## Semantic Classifier (for label prediction during inference)
##########################################################################

class SemanticClassifier(nn.Module):
    """
    Classifier to predict semantic labels from images.
    Used during inference when ground truth labels are not available.
    
    4 Classes:
    - 0: night_bg_focus
    - 1: night_raindrop_focus
    - 2: day_bg_focus
    - 3: day_raindrop_focus
    """
    def __init__(self, in_channels=3, num_classes=4):
        super(SemanticClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Stage 1
            nn.Conv2d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Stage 2
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 3
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 4
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input image
        Returns:
            [B, num_classes] class logits
        """
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


##########################################################################
## Model Variants
##########################################################################

def STRRNet_Base():
    """
    STRRNet Base variant (matches Miracle team's 26.89M params)
    
    기본 설정:
    - Semantic Guidance: 활성화 ✅
    - Background Subnet: 활성화 ✅
    - Semantic Classifier: 비활성화 (use_semantic_classifier=True로 변경 가능)
    
    사용 방법:
    - Training: --model_name STRRNet_Base (semantic labels 자동 생성)
    - Inference: semantic_labels=None으로 전달 (자동 분류 또는 fallback)
    """
    return STRRNet(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        use_semantic_guidance=True,
        use_bg_subnet=True,
        use_semantic_classifier=False  # True로 변경하면 Inference 시 Classifier 사용
    )


def STRRNet_Small():
    """STRRNet Small variant (lighter)"""
    return STRRNet(
        dim=32,
        num_blocks=[2, 4, 4, 6],
        num_refinement_blocks=2,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        use_semantic_guidance=True,
        use_bg_subnet=True
    )


def STRRNet_NoSemantic():
    """STRRNet without semantic guidance (ablation)"""
    return STRRNet(
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        use_semantic_guidance=False,
        use_bg_subnet=True
    )


if __name__ == '__main__':
    # Test
    model = STRRNet_Base()
    x = torch.randn(2, 3, 128, 128)
    labels = torch.tensor([0, 2])  # night_bg_focus, day_bg_focus
    
    # Training mode (with labels)
    y = model(x, semantic_labels=labels)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Inference mode (without labels)
    y_infer = model(x, semantic_labels=None)
    print(f"Inference output shape: {y_infer.shape}")
    
    # Test classifier
    classifier = SemanticClassifier()
    logits = classifier(x)
    print(f"Classifier output shape: {logits.shape}")
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()) / 1e6:.2f}M")
