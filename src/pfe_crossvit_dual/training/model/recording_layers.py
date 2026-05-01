import torch.nn as nn
import torch
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        mlp_ratio,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=[0.0],
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                # NOTE Changed drop with proj_drop (was deprecated) - youenn B
                # NOTE Changed timm Block with our custom RecordingBlock to record Attention and perform Attention Rollout
                tmp.append(
                    RecordingBlock(
                        dim=dim[d],
                        num_heads=num_heads[d],
                        mlp_ratio=mlp_ratio[d],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        proj_drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i],
                        norm_layer=norm_layer,
                    )
                )
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        """if len(self.blocks) == 0:
            self.blocks = None"""

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [
                    norm_layer(dim[d]),
                    act_layer(),
                    nn.Linear(dim[d], dim[(d + 1) % num_branches]),
                ]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(
                    CrossAttentionBlock(
                        dim=dim[d_],
                        num_heads=nh,
                        mlp_ratio=mlp_ratio[d],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[-1],
                        norm_layer=norm_layer,
                        has_mlp=False,
                    )
                )
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(
                        CrossAttentionBlock(
                            dim=dim[d_],
                            num_heads=nh,
                            mlp_ratio=mlp_ratio[d],
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop,
                            attn_drop=attn_drop,
                            drop_path=drop_path[-1],
                            norm_layer=norm_layer,
                            has_mlp=False,
                        )
                    )
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [
                    norm_layer(dim[(d + 1) % num_branches]),
                    act_layer(),
                    nn.Linear(dim[(d + 1) % num_branches], dim[d]),
                ]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat(
                (proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]),
                dim=1,
            )
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs


class RecordingAttention(nn.Module):
    """
    Custom Atention module which stores the attention heads to perform attention rollout.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # We store the the attention heads in a variable
        self.last_attn_map = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Storing
        # After the prediction, detaching to cpu to preserve GPU VRAM, before Dropout to capture model's reel intention
        # NOTE Attention : verifier si detach n'empeche pas le optimize.step() de prendre en compte notre loss custom pour l'iou
        self.last_attn_map = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RecordingBlock(nn.Module):
    """
    Custom block which stores attention heads to perform attention rollout.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = RecordingAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RecordingCrossAttention(nn.Module):
    """
    custom Cross Atention module which stores the attention heads to perform attention rollout.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # We store the the attention heads in a variable
        self.last_attn_map = None

    def forward(self, x):
        B, N, C = x.shape
        # Attention : CrossViT projette x différemment pour q, k, v
        q = (
            self.wq(x[:, 0:1, ...])
            .reshape(B, 1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Storing
        self.last_attn_map = attn.detach().cpu()

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Custom block which stores cross attention to perform attention rollout.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        has_mlp=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # custom Cross Atention module which stores the attention heads
        self.attn = RecordingCrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
