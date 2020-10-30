from MobilenetXT import MobileNetXT


inputshape = (None, 224, 224, 3)
numclass = 1
model = MobileNetXT(inputshape,numclass)
output = model.getOutput()
print(output.shape())

