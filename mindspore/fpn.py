import mindspore
import mindspore.nn as nn


class FeatPyramidNeck(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(FeatPyramidNeck, self).__init__()
        self.in_channels = in_channels
        self.fpn_layer = len(self.in_channels)

        lateral_convs_list_ = []
        fpn_convs_ = []

        for _, channel in enumerate(in_channels):
            l_conv = nn.Conv2d(channel, out_channels, kernel_size=1, stride=1, padding=0, pad_mode='valid', has_bias=True,)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='same', has_bias=True,)
            lateral_convs_list_.append(l_conv)
            fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.CellList(lateral_convs_list_)
        self.fpn_convs_list = nn.CellList(fpn_convs_)

    def construct(self, inputs):
        x = ()
        for i in range(self.fpn_layer):
            x += (self.lateral_convs_list[i](inputs[i]),)

        y = (x[3],)
        y = y + (x[2] + mindspore.ops.ResizeNearestNeighbor((x[2].shape[-2], x[2].shape[-1]))(y[self.fpn_layer - 4]),)
        y = y + (x[1] + mindspore.ops.ResizeNearestNeighbor((x[1].shape[-2], x[1].shape[-1]))(y[self.fpn_layer - 3]),)
        y = y + (x[0] + mindspore.ops.ResizeNearestNeighbor((x[0].shape[-2], x[0].shape[-1]))(y[self.fpn_layer - 2]),)

        z = ()
        for i in range(self.fpn_layer - 1, -1, -1):
            z = z + (y[i],)

        outs = ()
        for i in range(self.fpn_layer):
            outs = outs + (self.fpn_convs_list[i](z[i]),)

        return outs
