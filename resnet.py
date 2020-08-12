from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Dense, MaxPooling2D, AveragePooling2D, Flatten, add, Input
import tensorflow.keras.backend as BK
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
class ResNet:
    @staticmethod
    def residual_module(data, K, chan_dim, stride, mom=0.9, eps=1e-3, reg=0.0005, red=False):
        shortcut = data
        bn1 = BatchNormalization(axis=chan_dim, momentum=mom, epsilon=eps)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K*0.25), (1,1),use_bias=False, kernel_regularizer=l2(reg))(act1)

        bn2 = BatchNormalization(axis=chan_dim, momentum=mom, epsilon=eps)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K*0.25), (3,3), strides=stride, padding='same', use_bias=False, kernel_regularizer=l2(reg))(act2)

        bn3 = BatchNormalization(axis=chan_dim, momentum=mom, epsilon=eps)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1,1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        if red:
            shortcut = Conv2D(K, (1,1), strides=stride,use_bias=False, kernel_regularizer=l2(reg))(act1)
        
        x = add([conv3, shortcut])
        return x
    @staticmethod
    def build(width, height, depth, num_classes,stages,filters, mom=0.9, eps=1e-3, reg=0.0005 ):
        inputShape = (width, height, depth)
        chan_dim = -1
        if BK.image_data_format() == "channels_first":
            inputShape = (depth, width, height)
            chan_dim = 1
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chan_dim, momentum=mom, epsilon=eps)(inputs)
        x = Conv2D(filters[0], (3,3),use_bias=False, kernel_regularizer=l2(reg))(x)


        for i in range(0, len(stages)):
            stride = (1,1) if i==0 else (2,2)
            x = ResNet.residual_module(x, filters[i+1], chan_dim = chan_dim, stride=stride, red=True)
            for j in range(0, stages[i]-1):
                x = ResNet.residual_module(x, filters[i+1], chan_dim=chan_dim,stride=(1,1))
            
            x = BatchNormalization(axis=chan_dim, momentum=mom, epsilon=eps)(x)
            x = Activation("relu")(x)
            x = AveragePooling2D((8,8))(x)
            x = Flatten()(x)
            x = Dense(num_classes, kernel_regularizer=l2(reg))(x)
            x = Activation("softmax")(x)

            model = Model(inputs,x,name='resnet')
        return model


model = ResNet.build(64,64,3,200,(9,9,9),(32,64,128,256))
plot_model(model, to_file="./ResNet.png")

        



