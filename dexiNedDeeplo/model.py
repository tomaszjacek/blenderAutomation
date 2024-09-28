from tensorflow.keras import callbacks
from tensorflow.keras import layers, regularizers
from tensorflow.keras import optimizers, metrics, losses
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Conv1DTranspose,AveragePooling1D,GlobalMaxPool1D,GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D,MaxPool2D,Conv2DTranspose,AveragePooling2D,GlobalMaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Conv3D,MaxPooling3D,Conv3DTranspose,AveragePooling3D,GlobalMaxPool3D,GlobalAveragePooling3D
from tensorflow.keras.layers import Dense,Flatten,Dropout,Concatenate,Layer,BatchNormalization,Input,Add,Activation,Average
from keras.initializers import RandomNormal
glorot_normal = RandomNormal(stddev=0.01)
l2 = regularizers.l2
w_decay=1e-8 #0.0#2e-4#1e-3, 2e-4 # please define weight decay
K.clear_session()
# weight_init = tf.initializers.RandomNormal(mean=0.,stddev=0.01)
# weight_init = tf.initializers.glorot_normal()
weight_init = tf.initializers.glorot_normal()

class _DenseLayer(layers.Layer):
    """_DenseBlock model.

       Arguments:
         out_features: number of output features
    """

    def __init__(self, out_features,**kwargs):
        super(_DenseLayer, self).__init__(**kwargs)
        k_reg = None if w_decay is None else l2(w_decay)
        self.layers = []
        self.layers.append(tf.keras.Sequential(
            [
                layers.ReLU(),
                layers.Conv2D(
                    filters=out_features, kernel_size=(3,3), strides=(1,1), padding='same',
                    use_bias=True, kernel_initializer=weight_init,
                kernel_regularizer=k_reg),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    filters=out_features, kernel_size=(3,3), strides=(1,1), padding='same',
                    use_bias=True, kernel_initializer=weight_init,
                    kernel_regularizer=k_reg),
                layers.BatchNormalization(),
            ])) # first relu can be not needed


    def call(self, inputs):
        x1, x2 = tuple(inputs)
        new_features = x1
        for layer in self.layers:
            new_features = layer(new_features)

        return 0.5 * (new_features + x2), x2


class _DenseBlock(layers.Layer):
    """DenseBlock layer.

       Arguments:
         num_layers: number of _DenseLayer's per block
         out_features: number of output features
    """

    def __init__(self,
                 num_layers,
                 out_features,**kwargs):
        super(_DenseBlock, self).__init__(**kwargs)
        self.layers = [_DenseLayer(out_features) for i in range(num_layers)]

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs


class UpConvBlock(layers.Layer):
    """UpConvDeconvBlock layer.

       Arguments:
         up_scale: int
    """

    def __init__(self, up_scale,**kwargs):
        super(UpConvBlock, self).__init__(**kwargs)
        constant_features = 16
        k_reg = None if w_decay is None else l2(w_decay)
        features = []
        total_up_scale = 2 ** up_scale
        for i in range(up_scale):
            out_features = 1 if i == up_scale-1 else constant_features
            if i==up_scale-1:
                features.append(layers.Conv2D(
                    filters=out_features, kernel_size=(1,1), strides=(1,1), padding='same',
                    activation='relu', kernel_initializer=tf.initializers.RandomNormal(mean=0.),
                    kernel_regularizer=k_reg,use_bias=True)) #tf.initializers.TruncatedNormal(mean=0.)
                features.append(layers.Conv2DTranspose(
                    out_features, kernel_size=(total_up_scale,total_up_scale),
                    strides=(2,2), padding='same',
                    kernel_initializer=tf.initializers.RandomNormal(stddev=0.1),
                    kernel_regularizer=k_reg,use_bias=True)) # stddev=0.1
            else:

                features.append(layers.Conv2D(
                    filters=out_features, kernel_size=(1,1), strides=(1,1), padding='same',
                    activation='relu',kernel_initializer=weight_init,
                kernel_regularizer=k_reg,use_bias=True))
                features.append(layers.Conv2DTranspose(
                    out_features, kernel_size=(total_up_scale,total_up_scale),
                    strides=(2,2), padding='same', use_bias=True,
                    kernel_initializer=weight_init, kernel_regularizer=k_reg))

        self.features = keras.Sequential(features)

    def call(self, inputs):
        return self.features(inputs)


class SingleConvBlock(layers.Layer):
    """SingleConvBlock layer.

       Arguments:
         out_features: number of output features
         stride: stride per convolution
    """

    def __init__(self, out_features, k_size=(1,1),stride=(1,1),
                 use_bs=False, use_act=False,w_init=None,**kwargs): # bias_init=tf.constant_initializer(0.0)
        super(SingleConvBlock, self).__init__(**kwargs)
        self.use_bn = use_bs
        self.use_act = use_act
        k_reg = None if w_decay is None else l2(w_decay)
        self.conv = layers.Conv2D(
            filters=out_features, kernel_size=k_size, strides=stride,
            padding='same',kernel_initializer=w_init,
            kernel_regularizer=k_reg)#, use_bias=True, bias_initializer=bias_init
        if self.use_bn:
            self.bn = layers.BatchNormalization()
        if self.use_act:
            self.relu = layers.ReLU()

    def call(self, inputs):
        x =self.conv(inputs)
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.relu(x)
        return x


class DoubleConvBlock(layers.Layer):
    """DoubleConvBlock layer.

       Arguments:
         mid_features: number of middle features
         out_features: number of output features
         stride: stride per mid-layer convolution
    """

    def __init__(self, mid_features, out_features=None, stride=(1,1),
                 use_bn=True,use_act=True,**kwargs):
        super(DoubleConvBlock, self).__init__(**kwargs)
        self.use_bn =use_bn
        self.use_act =use_act
        out_features = mid_features if out_features is None else out_features
        k_reg = None if w_decay is None else l2(w_decay)

        self.conv1 = layers.Conv2D(
            filters=mid_features, kernel_size=(3, 3), strides=stride, padding='same',
        use_bias=True, kernel_initializer=weight_init,
        kernel_regularizer=k_reg)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(
            filters=out_features, kernel_size=(3, 3), padding='same',strides=(1,1),
        use_bias=True, kernel_initializer=weight_init,
        kernel_regularizer=k_reg)
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)
        return x


class DexiNed(tf.keras.Model):
    """DexiNet model."""

    def __init__(self,rgb_mean=None,
                 **kwargs):
        super(DexiNed, self).__init__(**kwargs)

        self.rgbn_mean = rgb_mean
        self.block_1 = DoubleConvBlock(32, 64, stride=(2,2),use_act=False)
        self.block_2 = DoubleConvBlock(128,use_act=False)
        self.dblock_3 = _DenseBlock(2, 256)
        self.dblock_4 = _DenseBlock(3, 512)
        self.dblock_5 = _DenseBlock(3, 512)
        self.dblock_6 = _DenseBlock(3, 256)
        self.maxpool = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(128,k_size=(1,1),stride=(2,2),use_bs=True,
                                      w_init=weight_init)
        self.side_2 = SingleConvBlock(256,k_size=(1,1),stride=(2,2),use_bs=True,
                                      w_init=weight_init)
        self.side_3 = SingleConvBlock(512,k_size=(1,1),stride=(2,2),use_bs=True,
                                      w_init=weight_init)
        self.side_4 = SingleConvBlock(512,k_size=(1,1),stride=(1,1),use_bs=True,
                                      w_init=weight_init)
        # self.side_5 = SingleConvBlock(256,k_size=(1,1),stride=(1,1),use_bs=True,
        #                               w_init=weight_init)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(256,k_size=(1,1),stride=(2,2),
                                      w_init=weight_init) # use_bn=True
        self.pre_dense_3 = SingleConvBlock(256,k_size=(1,1),stride=(1,1),use_bs=True,
                                      w_init=weight_init)
        self.pre_dense_4 = SingleConvBlock(512,k_size=(1,1),stride=(1,1),use_bs=True,
                                      w_init=weight_init)
        # self.pre_dense_5_0 = SingleConvBlock(512, k_size=(1,1),stride=(2,2),
        #                               w_init=weight_init) # use_bn=True
        self.pre_dense_5 = SingleConvBlock(512,k_size=(1,1),stride=(1,1),use_bs=True,
                                      w_init=weight_init)
        self.pre_dense_6 = SingleConvBlock(256,k_size=(1,1),stride=(1,1),use_bs=True,
                                      w_init=weight_init)
        # USNet
        self.up_block_1 = UpConvBlock(1)
        self.up_block_2 = UpConvBlock(1)
        self.up_block_3 = UpConvBlock(2)
        self.up_block_4 = UpConvBlock(3)
        self.up_block_5 = UpConvBlock(4)
        self.up_block_6 = UpConvBlock(4)

        self.block_cat = SingleConvBlock(
            1,k_size=(1,1),stride=(1,1),
            w_init=tf.constant_initializer(1/5))


    def slice(self, tensor, slice_shape):
        height, width = slice_shape
        return tensor[..., :height, :width]


    def call(self, x):
        # Block 1
        x = x-self.rgbn_mean[:-1]
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2) # the key for the second skip connec...
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add) #

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_4_pre_dense_256 = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_4_pre_dense_256 + block_3_down)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        # block_5_pre_dense_512 = self.pre_dense_5_0(block_4_pre_dense_256)
        block_5_pre_dense = self.pre_dense_5(block_4_down )
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        height, width = x.shape[1:3]
        slice_shape = (height, width)
        out_1 = self.up_block_1(block_1) # self.slice(, slice_shape)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = tf.concat(results, 3)  # BxHxWX6
        block_cat = self.block_cat(block_cat)  # BxHxWX1

        results.append(block_cat)

        return results

def weighted_cross_entropy_loss(input, label):
    y = tf.cast(label,dtype=tf.float32)
    negatives = tf.math.reduce_sum(1.-y)
    positives = tf.math.reduce_sum(y)

    beta = negatives/(negatives + positives)
    pos_w = beta/(1-beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(
    labels=label, logits=input, pos_weight=pos_w, name=None)
    cost = tf.reduce_sum(cost*(1-beta))
    return tf.where(tf.equal(positives, 0.0), 0.0, cost)


def pre_process_binary_cross_entropy(bc_loss,input, label,arg, use_tf_loss=False):
    # preprocess data
    y = label
    loss = 0
    w_loss=1.0
    preds = []
    for tmp_p in input:
        # tmp_p = input[i]

        # loss processing
        tmp_y = tf.cast(y, dtype=tf.float32)
        mask = tf.dtypes.cast(tmp_y > 0., tf.float32)
        b,h,w,c=mask.get_shape()
        positives = tf.math.reduce_sum(mask, axis=[1, 2, 3], keepdims=True)
        negatives = h*w*c-positives

        beta2 = (1.*positives) / (negatives + positives) # negatives in hed
        beta = (1.1*negatives)/ (positives + negatives) # positives in hed
        pos_w = tf.where(tf.equal(y, 0.0), beta2, beta)
        logits = tf.sigmoid(tmp_p)

        l_cost = bc_loss(y_true=tmp_y, y_pred=logits,
                         sample_weight=pos_w)

        preds.append(logits)
        loss += (l_cost*w_loss)

    return preds, loss

class DexiNedTj(tf.keras.Model):
    """DexiNet model."""

    def __init__(self,rgb_mean=None,
                 **kwargs):
        super(DexiNedTj, self).__init__(**kwargs)

        self.rgbn_mean = rgb_mean
        self.conv2D_3 = Conv2D(32 , kernel_size=(3,3),strides=(2,2),padding = 'same',use_bias=True,kernel_initializer= glorot_normal)
        self.batchnormalization_24 = BatchNormalization()
        self.conv2D_40 = Conv2D(64 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True,kernel_initializer= glorot_normal)
        self.batchnormalization_18 = BatchNormalization()
        self.activation_11 = Activation(activation='relu')
        self.conv2D_1 = Conv2D(128 , kernel_size=(3,3),strides=(1,1),padding = 'same')
        self.conv2D_16 = Conv2D(128 , kernel_size=(1,1),strides=(2,2),padding = 'same',use_bias=True)
        self.conv2D_51 = Conv2D(1 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_1 = BatchNormalization()
        self.transpoze2D_9 = Conv2DTranspose(1 , kernel_size=(2,2),strides=(2,2),padding = 'same')
        self.conv2D_49 = Conv2D(128 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_8 = BatchNormalization()
        self.activation_1 = Activation(activation='relu')
        self.conv2D_17 = Conv2D(1 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.maxpool2D_1 = MaxPool2D(pool_size=(3,3),strides=(2,2),padding = 'same')
        self.transpoze2D_10 = Conv2DTranspose(1 , kernel_size=(2,2),strides=(2,2),padding = 'same')
        self.conv2D_28 = Conv2D(256 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.conv2D_36 = Conv2D(256 , kernel_size=(1,1),strides=(2,2),padding = 'same',use_bias=True)
        self.activation_8 = Activation(activation='relu')
        self.conv2D_47 = Conv2D(256 , kernel_size=(1,1),strides=(2,2),padding = 'same',use_bias=True)
        self.conv2D_48 = Conv2D(512 , kernel_size=(1,1),strides=(2,2),padding = 'same')
        self.conv2D_33 = Conv2D(256 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_6 = BatchNormalization()
        self.activation_9 = Activation(activation='relu')
        self.conv2D_31 = Conv2D(256 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_22 = BatchNormalization()
        self.activation_6 = Activation(activation='relu')
        self.conv2D_30 = Conv2D(256 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_4 = BatchNormalization()
        self.activation_7 = Activation(activation='relu')
        self.conv2D_18 = Conv2D(256 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_20 = BatchNormalization()
        self.conv2D_38 = Conv2D(16 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.maxpool2D_2 = MaxPool2D(pool_size=(3,3),strides=(2,2),padding = 'same')
        self.transpoze2D_15 = Conv2DTranspose(16 , kernel_size=(4,4),strides=(2,2),padding = 'same')
        self.conv2D_11 = Conv2D(1 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.conv2D_34 = Conv2D(512 , kernel_size=(1,1),strides=(2,2),padding = 'same',use_bias=True)
        self.conv2D_50 = Conv2D(512 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.transpoze2D_2 = Conv2DTranspose(1 , kernel_size=(4,4),strides=(2,2),padding = 'same')
        self.activation_10 = Activation(activation='relu')
        self.conv2D_32 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_10 = BatchNormalization()
        self.activation_5 = Activation(activation='relu')
        self.conv2D_27 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_19 = BatchNormalization()
        self.activation_24 = Activation(activation='relu')
        self.conv2D_26 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_13 = BatchNormalization()
        self.activation_16 = Activation(activation='relu')
        self.conv2D_13 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_17 = BatchNormalization()
        self.activation_22 = Activation(activation='relu')
        self.conv2D_29 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_15 = BatchNormalization()
        self.activation_23 = Activation(activation='relu')
        self.conv2D_24 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_12 = BatchNormalization()
        self.conv2D_44 = Conv2D(16 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.maxpool2D_3 = MaxPool2D(pool_size=(3,3),strides=(2,2),padding = 'same')
        self.transpoze2D_7 = Conv2DTranspose(16 , kernel_size=(8,8),strides=(2,2),padding = 'same')
        self.conv2D_45 = Conv2D(16 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.activation_15 = Activation(activation='relu')
        self.conv2D_6 = Conv2D(512 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.conv2D_2 = Conv2D(512 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.transpoze2D_8 = Conv2DTranspose(16 , kernel_size=(8,8),strides=(2,2),padding = 'same')
        self.conv2D_43 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.conv2D_41 = Conv2D(1 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_7 = BatchNormalization()
        self.transpoze2D_14 = Conv2DTranspose(1 , kernel_size=(8,8),strides=(2,2),padding = 'same')
        self.activation_17 = Activation(activation='relu')
        self.conv2D_19 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_16 = BatchNormalization()
        self.activation_13 = Activation(activation='relu')
        self.conv2D_21 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_14 = BatchNormalization()
        self.activation_14 = Activation(activation='relu')
        self.conv2D_15 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_5 = BatchNormalization()
        self.activation_18 = Activation(activation='relu')
        self.conv2D_14 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_26 = BatchNormalization()
        self.activation_19 = Activation(activation='relu')
        self.conv2D_5 = Conv2D(512 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_3 = BatchNormalization()
        self.conv2D_8 = Conv2D(256 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.conv2D_35 = Conv2D(16 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.activation_20 = Activation(activation='relu')
        self.transpoze2D_4 = Conv2DTranspose(16 , kernel_size=(16,16),strides=(2,2),padding = 'same')
        self.conv2D_22 = Conv2D(256 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.conv2D_12 = Conv2D(16 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_11 = BatchNormalization()
        self.transpoze2D_12 = Conv2DTranspose(16 , kernel_size=(16,16),strides=(2,2),padding = 'same')
        self.activation_21 = Activation(activation='relu')
        self.conv2D_37 = Conv2D(16 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.conv2D_10 = Conv2D(256 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.transpoze2D_5 = Conv2DTranspose(16 , kernel_size=(16,16),strides=(2,2),padding = 'same')
        self.batchnormalization_21 = BatchNormalization()
        self.conv2D_42 = Conv2D(16 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.transpoze2D_13 = Conv2DTranspose(1 , kernel_size=(16,16),strides=(2,2),padding = 'same')
        self.activation_3 = Activation(activation='relu')
        self.conv2D_20 = Conv2D(256 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_9 = BatchNormalization()
        self.activation_4 = Activation(activation='relu')
        self.conv2D_23 = Conv2D(256 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_2 = BatchNormalization()
        self.activation_2 = Activation(activation='relu')
        self.conv2D_7 = Conv2D(256 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_23 = BatchNormalization()
        self.activation_12 = Activation(activation='relu')
        self.conv2D_25 = Conv2D(256 , kernel_size=(3,3),strides=(1,1),padding = 'same',use_bias=True)
        self.batchnormalization_25 = BatchNormalization()
        self.conv2D_39 = Conv2D(16 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.transpoze2D_11 = Conv2DTranspose(16 , kernel_size=(16,16),strides=(2,2),padding = 'same')
        self.conv2D_4 = Conv2D(16 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.transpoze2D_3 = Conv2DTranspose(16 , kernel_size=(16,16),strides=(2,2),padding = 'same')
        self.conv2D_9 = Conv2D(16 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.transpoze2D_6 = Conv2DTranspose(16 , kernel_size=(16,16),strides=(2,2),padding = 'same')
        self.conv2D_46 = Conv2D(1 , kernel_size=(1,1),strides=(1,1),padding = 'same',use_bias=True)
        self.transpoze2D_1 = Conv2DTranspose(1 , kernel_size=(16,16),strides=(2,2),padding = 'same')



    def call(self, x):
        x = x-self.rgbn_mean[:-1]
        conv2D_3 = self.conv2D_3(x)
        batchnormalization_24 = self.batchnormalization_24(conv2D_3)
        conv2D_40 = self.conv2D_40(batchnormalization_24)
        batchnormalization_18 = self.batchnormalization_18(conv2D_40)
        activation_11 = self.activation_11(batchnormalization_18)
        conv2D_1 = self.conv2D_1(activation_11)
        conv2D_16 = self.conv2D_16(activation_11)
        conv2D_51 = self.conv2D_51(activation_11)
        batchnormalization_1 = self.batchnormalization_1(conv2D_1)
        transpoze2D_9 = self.transpoze2D_9(conv2D_51)
        conv2D_49 = self.conv2D_49(batchnormalization_1)
        batchnormalization_8 = self.batchnormalization_8(conv2D_49)
        activation_1 = self.activation_1(batchnormalization_8)
        conv2D_17 = self.conv2D_17(activation_1)
        maxpool2D_1 = self.maxpool2D_1(activation_1)
        transpoze2D_10 = self.transpoze2D_10(conv2D_17)
        add_1 = Add()([conv2D_16,maxpool2D_1])
        conv2D_28 = self.conv2D_28(maxpool2D_1)
        conv2D_36 = self.conv2D_36(maxpool2D_1)
        activation_8 = self.activation_8(add_1)
        conv2D_47 = self.conv2D_47(add_1)
        conv2D_48 = self.conv2D_48(conv2D_36)
        conv2D_33 = self.conv2D_33(activation_8)
        batchnormalization_6 = self.batchnormalization_6(conv2D_33)
        activation_9 = self.activation_9(batchnormalization_6)
        conv2D_31 = self.conv2D_31(activation_9)
        batchnormalization_22 = self.batchnormalization_22(conv2D_31)
        average_5 = Average()([batchnormalization_22,conv2D_28])
        activation_6 = self.activation_6(average_5)
        conv2D_30 = self.conv2D_30(activation_6)
        batchnormalization_4 = self.batchnormalization_4(conv2D_30)
        activation_7 = self.activation_7(batchnormalization_4)
        conv2D_18 = self.conv2D_18(activation_7)
        batchnormalization_20 = self.batchnormalization_20(conv2D_18)
        average_4 = Average()([batchnormalization_20,conv2D_28])
        conv2D_38 = self.conv2D_38(average_4)
        maxpool2D_2 = self.maxpool2D_2(average_4)
        transpoze2D_15 = self.transpoze2D_15(conv2D_38)
        add_2 = Add()([conv2D_47,maxpool2D_2])
        add_4 = Add()([conv2D_36,maxpool2D_2])
        conv2D_11 = self.conv2D_11(transpoze2D_15)
        conv2D_34 = self.conv2D_34(add_2)
        conv2D_50 = self.conv2D_50(add_4)
        transpoze2D_2 = self.transpoze2D_2(conv2D_11)
        activation_10 = self.activation_10(add_2)
        conv2D_32 = self.conv2D_32(activation_10)
        batchnormalization_10 = self.batchnormalization_10(conv2D_32)
        activation_5 = self.activation_5(batchnormalization_10)
        conv2D_27 = self.conv2D_27(activation_5)
        batchnormalization_19 = self.batchnormalization_19(conv2D_27)
        average_3 = Average()([batchnormalization_19,conv2D_50])
        activation_24 = self.activation_24(average_3)
        conv2D_26 = self.conv2D_26(activation_24)
        batchnormalization_13 = self.batchnormalization_13(conv2D_26)
        activation_16 = self.activation_16(batchnormalization_13)
        conv2D_13 = self.conv2D_13(activation_16)
        batchnormalization_17 = self.batchnormalization_17(conv2D_13)
        average_2 = Average()([batchnormalization_17,conv2D_50])
        activation_22 = self.activation_22(average_2)
        conv2D_29 = self.conv2D_29(activation_22)
        batchnormalization_15 = self.batchnormalization_15(conv2D_29)
        activation_23 = self.activation_23(batchnormalization_15)
        conv2D_24 = self.conv2D_24(activation_23)
        batchnormalization_12 = self.batchnormalization_12(conv2D_24)
        average_11 = Average()([batchnormalization_12,conv2D_50])
        conv2D_44 = self.conv2D_44(average_11)
        maxpool2D_3 = self.maxpool2D_3(average_11)
        transpoze2D_7 = self.transpoze2D_7(conv2D_44)
        add_3 = Add()([conv2D_34,maxpool2D_3])
        add_5 = Add()([conv2D_48,maxpool2D_3])
        conv2D_45 = self.conv2D_45(transpoze2D_7)
        activation_15 = self.activation_15(add_3)
        conv2D_6 = self.conv2D_6(add_3)
        conv2D_2 = self.conv2D_2(add_5)
        transpoze2D_8 = self.transpoze2D_8(conv2D_45)
        conv2D_43 = self.conv2D_43(activation_15)
        conv2D_41 = self.conv2D_41(transpoze2D_8)
        batchnormalization_7 = self.batchnormalization_7(conv2D_43)
        transpoze2D_14 = self.transpoze2D_14(conv2D_41)
        activation_17 = self.activation_17(batchnormalization_7)
        conv2D_19 = self.conv2D_19(activation_17)
        batchnormalization_16 = self.batchnormalization_16(conv2D_19)
        average_7 = Average()([batchnormalization_16,conv2D_2])
        activation_13 = self.activation_13(average_7)
        conv2D_21 = self.conv2D_21(activation_13)
        batchnormalization_14 = self.batchnormalization_14(conv2D_21)
        activation_14 = self.activation_14(batchnormalization_14)
        conv2D_15 = self.conv2D_15(activation_14)
        batchnormalization_5 = self.batchnormalization_5(conv2D_15)
        average_6 = Average()([batchnormalization_5,conv2D_2])
        activation_18 = self.activation_18(average_6)
        conv2D_14 = self.conv2D_14(activation_18)
        batchnormalization_26 = self.batchnormalization_26(conv2D_14)
        activation_19 = self.activation_19(batchnormalization_26)
        conv2D_5 = self.conv2D_5(activation_19)
        batchnormalization_3 = self.batchnormalization_3(conv2D_5)
        average_1 = Average()([batchnormalization_3,conv2D_2])
        add_6 = Add()([average_1,conv2D_6])
        conv2D_8 = self.conv2D_8(average_1)
        conv2D_35 = self.conv2D_35(average_1)
        activation_20 = self.activation_20(add_6)
        transpoze2D_4 = self.transpoze2D_4(conv2D_35)
        conv2D_22 = self.conv2D_22(activation_20)
        conv2D_12 = self.conv2D_12(transpoze2D_4)
        batchnormalization_11 = self.batchnormalization_11(conv2D_22)
        transpoze2D_12 = self.transpoze2D_12(conv2D_12)
        activation_21 = self.activation_21(batchnormalization_11)
        conv2D_37 = self.conv2D_37(transpoze2D_12)
        conv2D_10 = self.conv2D_10(activation_21)
        transpoze2D_5 = self.transpoze2D_5(conv2D_37)
        batchnormalization_21 = self.batchnormalization_21(conv2D_10)
        conv2D_42 = self.conv2D_42(transpoze2D_5)
        average_10 = Average()([batchnormalization_21,conv2D_8])
        transpoze2D_13 = self.transpoze2D_13(conv2D_42)
        activation_3 = self.activation_3(average_10)
        conv2D_20 = self.conv2D_20(activation_3)
        batchnormalization_9 = self.batchnormalization_9(conv2D_20)
        activation_4 = self.activation_4(batchnormalization_9)
        conv2D_23 = self.conv2D_23(activation_4)
        batchnormalization_2 = self.batchnormalization_2(conv2D_23)
        average_9 = Average()([batchnormalization_2,conv2D_8])
        activation_2 = self.activation_2(average_9)
        conv2D_7 = self.conv2D_7(activation_2)
        batchnormalization_23 = self.batchnormalization_23(conv2D_7)
        activation_12 = self.activation_12(batchnormalization_23)
        conv2D_25 = self.conv2D_25(activation_12)
        batchnormalization_25 = self.batchnormalization_25(conv2D_25)
        average_8 = Average()([batchnormalization_25,conv2D_8])
        conv2D_39 = self.conv2D_39(average_8)
        transpoze2D_11 = self.transpoze2D_11(conv2D_39)
        conv2D_4 = self.conv2D_4(transpoze2D_11)
        transpoze2D_3 = self.transpoze2D_3(conv2D_4)
        conv2D_9 = self.conv2D_9(transpoze2D_3)
        transpoze2D_6 = self.transpoze2D_6(conv2D_9)
        conv2D_46 = self.conv2D_46(transpoze2D_6)
        transpoze2D_1 = self.transpoze2D_1(conv2D_46)
        concatenate_1 = Concatenate()([transpoze2D_1,transpoze2D_2,transpoze2D_9,transpoze2D_10,transpoze2D_13,transpoze2D_14])

        return concatenate_1