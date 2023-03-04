import mindspore
from mindspore import nn, ops, train
from typing import Optional, Dict


class Attention(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = mindspore.Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(keep_prob)

        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        identity = x
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = ops.unstack(qkv, axis=0)
        attn = self.q_matmul_k(q, k)
        attn = ops.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = self.attn_matmul_v(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, n, c))
        out = self.out(out)
        out = self.out_drop(out)

        return out + identity


class FeedForward(nn.Cell):
    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: nn.Cell = nn.ReLU,
                 keep_prob: float = 1.0):
        super(FeedForward, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Dense(in_features, hidden_features)
        self.activation = activation()
        self.dense2 = nn.Dense(hidden_features, out_features)
        self.dropout = nn.Dropout(keep_prob)

    def construct(self, x):
        identity = x
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x + identity

class DynamicConv(nn.Cell):
    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=7,
                 with_proj=True,
                 activation=nn.ReLU,
                 norm=nn.LayerNorm,
                 ):
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.with_proj = with_proj
        self.out_channels = out_channels if out_channels else in_channels
        
        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer = nn.Dense(
            self.in_channels, self.num_params_in + self.num_params_out)
        
        self.norm_in = norm((self.feat_channels,))
        self.norm_out = norm((self.out_channels,))
        
        self.activation = activation()
        
        self.bmm = ops.BatchMatMul()

        num_output = self.out_channels * input_feat_shape**2
        if self.with_proj:
            self.fc_layer = nn.Dense(num_output, self.out_channels)
            self.fc_norm = norm((self.out_channels,))
    
    def construct(self, param_feature, input_feature):
        b, c, h, w = input_feature.shape
        input_feature = ops.reshape(input_feature, (b, c, h*w))
        input_feature = ops.transpose(input_feature, (0, 2, 1))

        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].reshape(
            -1, self.in_channels, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].reshape(
            -1, self.feat_channels, self.out_channels)

        features = self.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        features = self.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        if self.with_proj:
            features = ops.reshape(features, (b, -1))
            features = self.fc_layer(features)
            features = self.fc_norm(features)
            features = self.activation(features)

        return features
