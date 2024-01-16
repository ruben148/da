from keras.applications import ResNet50, InceptionResNetV2
from keras.layers import Input, ZeroPadding2D, UpSampling2D, Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Activation, Dropout
from keras.models import Model
import keras.backend as K

def build_resnet50_unet(input_shape, num_classes, weights_path, dropout_rate=0.25):
    resnet = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    resnet.load_weights(weights_path)

    s1 = resnet.get_layer("input_1").output             # input layer
    s2 = resnet.get_layer("conv1_relu").output          # first conv layer
    s3 = resnet.get_layer("conv2_block3_out").output    # 256x256
    s4 = resnet.get_layer("conv3_block4_out").output    # 128x128
    s5 = resnet.get_layer("conv4_block6_out").output    # 64x64

    # bridge
    b1 = resnet.get_layer("conv5_block3_out").output    # 32x32

    # decoder
    d1 = UpSampling2D()(b1)
    d1 = Concatenate()([d1, s5])
    d1 = Conv2D(512, (3, 3), padding="same")(d1)
    d1 = BatchNormalization()(d1)
    d1 = Activation("relu")(d1)
    d1 = Dropout(dropout_rate)(d1)

    d2 = UpSampling2D()(d1)
    d2 = Concatenate()([d2, s4])
    d2 = Conv2D(256, (3, 3), padding="same")(d2)
    d2 = BatchNormalization()(d2)
    d2 = Activation("relu")(d2)
    d2 = Dropout(dropout_rate)(d2)

    d3 = UpSampling2D()(d2)
    d3 = Concatenate()([d3, s3])
    d3 = Conv2D(128, (3, 3), padding="same")(d3)
    d3 = BatchNormalization()(d3)
    d3 = Activation("relu")(d3)
    d3 = Dropout(dropout_rate)(d3)

    d4 = UpSampling2D()(d3)
    d4 = Concatenate()([d4, s2])
    d4 = Conv2D(64, (3, 3), padding="same")(d4)
    d4 = BatchNormalization()(d4)
    d4 = Activation("relu")(d4)
    d4 = Dropout(dropout_rate)(d4)

    d5 = UpSampling2D()(d4)
    d5 = Concatenate()([d5, s1])
    d5 = Conv2D(64, (3, 3), padding="same")(d5)
    d5 = BatchNormalization()(d5)
    d5 = Activation("relu")(d5)
    d5 = Dropout(dropout_rate)(d5)

    # output
    output = Conv2D(num_classes, (1, 1), activation="softmax")(d5)

    model = Model(inputs=resnet.input, outputs=output)

    return model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    x = conv_block(x, num_filters)
    return x

def build_resnet50_unet_v2(input_shape, num_classes, weights_path):
    inputs = Input(input_shape)

    encoder = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    encoder.load_weights(weights_path)

    s1 = encoder.get_layer("input_1").output
    s2 = encoder.get_layer("conv1_relu").output
    s3 = encoder.get_layer("conv2_block3_out").output
    s4 = encoder.get_layer("conv3_block4_out").output
    b1 = encoder.get_layer("conv4_block6_out").output

    # s2 = ZeroPadding2D((1, 1))(s2)

    # decoder blocks
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    dropout = Dropout(0.3)(d4)
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(dropout)

    model = Model(inputs, outputs, name="ResNet50-UNet")
    return model

def build_inception_resnetv2_unet(input_shape, num_classes, weights_path):
    inputs = Input(input_shape)

    encoder = InceptionResNetV2(include_top=False, weights=None, input_tensor=inputs)
    encoder.load_weights(weights_path)
    s1 = encoder.get_layer("input_1").output           ## (512 x 512)

    s2 = encoder.get_layer("activation").output        ## (255 x 255)
    s2 = ZeroPadding2D(( (1, 0), (1, 0) ))(s2)         ## (256 x 256)

    s3 = encoder.get_layer("activation_3").output      ## (126 x 126)
    s3 = ZeroPadding2D((1, 1))(s3)                     ## (128 x 128)

    s4 = encoder.get_layer("activation_74").output      ## (61 x 61)
    s4 = ZeroPadding2D(( (2, 1),(2, 1) ))(s4)           ## (64 x 64)

    b1 = encoder.get_layer("activation_161").output     ## (30 x 30)
    b1 = ZeroPadding2D((1, 1))(b1)                      ## (32 x 32)

    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)
    
    dropout = Dropout(0.3)(d4)
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(dropout)

    model = Model(inputs, outputs, name="InceptionResNetV2-UNet")
    return model

def dice_loss_aux(y_true, y_pred, class_weights):
    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')

    dice_loss = 0.0
    for class_index in range(len(class_weights)):
        y_true_class = y_true_f[..., class_index]
        y_pred_class = y_pred_f[..., class_index]
        numerator = 2 * K.sum(y_true_class * y_pred_class)
        denominator = K.sum(y_true_class + y_pred_class)

        dice_coeff = (numerator + K.epsilon()) / (denominator + K.epsilon())

        dice_loss += (1 - dice_coeff) * class_weights[class_index]

    return dice_loss / len(class_weights)


def weighted_dice_loss(class_weights):
    def dice_loss(y_true, y_pred):
        return dice_loss_aux(y_true, y_pred, class_weights)
    return dice_loss

def dice_loss(y_true, y_pred):
    y_true_f = K.cast(y_true, 'float32')
    y_pred_f = K.cast(y_pred, 'float32')
    
    numerator = 2 * K.sum(y_true_f * y_pred_f)
    denominator = K.sum(y_true_f + y_pred_f)
    
    return 1 - (numerator + K.epsilon()) / (denominator + K.epsilon())