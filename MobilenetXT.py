import tensorflow as tf
from LayersXT import LayerProvider




class MobileNetXT:

    def __init__(self,shape, totalclass):

        tf.reset_default_graph()# 利用这个可清空default graph以及nodes

        lProvider = LayerProvider(True)

        adaptChannels = lambda totalLayer: int(1 * totalLayer)

        self.inputImage = tf.placeholder(tf.float32, shape=shape, name='Image')
        self.transtrain = True

        output = lProvider.convb(self.inputImage, 3, 3, adaptChannels(32), 2, "1-conv-32-2-1", relu=True)
        print("1-conv-32-2-1 : " + str(output.shape))

        # architecture description

        sand_glass_setting = [
            # t, c,  b, s
            [2, 96,  1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 1],
            [6, 576, 4, 2],
            [6, 960, 2, 1],
            [6, 1280, 1, 1]
        ]
        self.sandglass_type = 0
        for t, c, n, s in sand_glass_setting:
            self.sandglass_type += 1
            output_channel = adaptChannels(c)
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                layerDescription = "l" + str(self.sandglass_type) + "-sandglass-n" + str(i+1)
                output = lProvider.inverted_bottleneck(output, t, output_channel, stride, k_s=3, dilation=1, scope=layerDescription)


        output = lProvider.convb(output, 1, 1, adaptChannels(1280), 1, "2-conv-1280-2-1", relu=True)
        output = tf.layers.average_pooling2d(output, 7, 1,
                                    padding='valid', data_format='channels_last', name="avgpool")
        output = lProvider.convb(output, 1, 1, totalclass, 1, "ptconv-1280-clsnum", relu=True)
        self.output = tf.squeeze(output,[1,2],"output")


    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def getInput(self):
        return self.inputImage

    def getIntermediateOutputs(self):
        return self.intermediateSupervisionOutputs[:]

    def getOutput(self):
        return self.output

