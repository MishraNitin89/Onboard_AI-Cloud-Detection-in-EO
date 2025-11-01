import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

def upsample_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.concatenate([x, skip])
    x = conv_block(x, filters)
    return x

def build_mobilenetv3_unet(input_shape=(256,256,4)):
    base_model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, include_top=False, weights='imagenet')
    skips = [
        base_model.get_layer('expanded_conv/project').output,
        base_model.get_layer('expanded_conv_3/project').output,
        base_model.get_layer('expanded_conv_6/project').output,
    ]
    x = base_model.output
    x = conv_block(x, 256)
    x = upsample_block(x, skips[2], 128)
    x = upsample_block(x, skips[1], 64)
    x = upsample_block(x, skips[0], 32)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(base_model.input, outputs)
    return model
