from mindspore import nn, ops
from transformer import DynamicConv


class DynamicMaskHead(nn.Cell):
    def __init__(self,
                 input_channels=256,
                 output_channels=256,
                 num_classes=80,
                 ):
        super(DynamicMaskHead, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.instance_interactive_conv = DynamicConv(
                                            in_channels=input_channels,
                                            feat_channels=64,
                                            out_channels=output_channels,
                                            input_feat_shape=14,
                                            with_proj=False)

        self.convs = nn.SequentialCell([
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=0, has_bias=False, pad_mode="same"),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=0, has_bias=False, pad_mode="same"),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=0, has_bias=False, pad_mode="same"),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=0, has_bias=False, pad_mode="same"),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        ])
        self.upsample = nn.Conv2dTranspose(output_channels, output_channels, kernel_size=2, stride=2, pad_mode="valid", has_bias=True)
        self.conv_logits = nn.Conv2d(output_channels, num_classes, kernel_size=1, padding=0, pad_mode="valid", has_bias=True)
        self.relu = nn.ReLU()

    def construct(self, roi_feat, proposal_feat):
        proposal_feat = proposal_feat.reshape(-1, self.input_channels)
        proposal_feat_iic = self.instance_interactive_conv(
            proposal_feat, roi_feat)

        proposal_feat_iic = ops.transpose(proposal_feat_iic, (0, 2, 1))
        x = ops.reshape(proposal_feat_iic, roi_feat.shape)

        x = self.convs(x)
        x = self.upsample(x)
        x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred
