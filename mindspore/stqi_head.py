from transformer import Attention, FeedForward, DynamicConv
from mindspore import nn, ops

class DIIHead(nn.Cell):
    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 dropout=1.0,
                 ffn_act_cfg=nn.ReLU,
                 dynamic_conv=DynamicConv(
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=7),
                 ):
        super(DIIHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.attention = Attention(in_channels, num_heads, dropout)
        self.attention_norm = nn.LayerNorm((in_channels,))

        self.instance_interactive_conv = dynamic_conv
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = nn.LayerNorm((in_channels,))

        self.ffn = FeedForward(
            in_channels,
            feedforward_channels)
        self.ffn_norm = nn.LayerNorm((in_channels,))

        self.cls_fcs = nn.CellList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Dense(in_channels, in_channels, has_bias=False))
            self.cls_fcs.append(
                nn.LayerNorm((in_channels,)))
            self.cls_fcs.append(
                ffn_act_cfg())

        self.fc_cls = nn.Dense(in_channels, self.num_classes)

        self.reg_fcs = nn.CellList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Dense(in_channels, in_channels, has_bias=False))
            self.reg_fcs.append(
                nn.LayerNorm((in_channels,)))
            self.reg_fcs.append(
                ffn_act_cfg())

        self.fc_reg = nn.Dense(in_channels, 4)
    
    def construct(self, roi_feat, proposal_feat):
        N, num_proposals = proposal_feat.shape[:2]

        # self attention
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        
        # temporal self attention
        proposal_feat = ops.transpose(proposal_feat, (1, 0, 2))
        proposal_feat = self.attention_norm(self.attention(proposal_feat))
        proposal_feat = ops.transpose(proposal_feat, (1, 0, 2))

        attn_feats = proposal_feat

        # instance interactive
        proposal_feat = attn_feats.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat
        reg_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).reshape(
            N, num_proposals, self.num_classes)
        bbox_delta = self.fc_reg(reg_feat).reshape(N, num_proposals, 4)

        return cls_score, bbox_delta, obj_feat.reshape(
            N, num_proposals, self.in_channels), attn_feats
