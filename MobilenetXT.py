import tensorflow as tf
from opt import opt
from src.models.outputlayer import finallayerforoffsetoption
from src.models.Layerprovider_NX import LayerProvider




class PoseNetNX:

    def __init__(self,shape, is4Train, mobilenetVersion=1, totalJoints=opt.totaljoints):

        tf.reset_default_graph()# 利用这个可清空default graph以及nodes

        lProvider = LayerProvider(is4Train)
        outputlayer = finallayerforoffsetoption()

        adaptChannels = lambda totalLayer: int(mobilenetVersion * totalLayer)

        self.inputImage = tf.placeholder(tf.float32, shape=shape, name='Image')
        self.transtrain = opt.isTrainpre

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
        self.bottleneck_type = 0
        for t, c, n, s in sand_glass_setting:
            self.bottleneck_type += 1
            output_channel = adaptChannels(c)
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                layerDescription = "l" + str(self.bottleneck_type) + "-bottleneck-n" + str(i+1)
                output = lProvider.inverted_bottleneck(output, t, output_channel, stride, k_s=3, dilation=1, scope=layerDescription)


        output = lProvider.convb(output, 1, 1, adaptChannels(1280), 1, "2-conv-1280-2-1", relu=True)
        self.output = outputlayer.fornetworks(output, totalJoints)


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

