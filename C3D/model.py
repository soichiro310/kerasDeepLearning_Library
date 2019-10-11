from keras.models import *
from keras.layers import *

def C3D(seq_length=40, image_size=80,channel=3,nb_class=2, activation='softmax'):
    input_ = Input(shape=(seq_length,image_size,image_size,channel))    
    x = Conv3D(64, (3, 3, 3), activation='relu',border_mode='same'
               ,subsample=(1, 1, 1),input_shape=(seq_length,image_size,image_size,channel))(input_)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid')(x)
    # 2nd layer group
    x = Conv3D(128, (3, 3, 3), activation='relu',border_mode='same',subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),border_mode='valid')(x)
    # 3rd layer group
    x = Conv3D(256, (3, 3, 3), activation='relu',border_mode='same',subsample=(1, 1, 1))(x)
    x = Conv3D(256, (3, 3, 3), activation='relu',border_mode='same',subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),border_mode='valid')(x)
    # 4th layer group
    x = Conv3D(512, (3, 3, 3), activation='relu',border_mode='same',subsample=(1, 1, 1))(x)
    x = Conv3D(512, (3, 3, 3), activation='relu',border_mode='same',subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),border_mode='valid')(x)

    # 5th layer group
    x = Conv3D(512, (3, 3, 3), activation='relu',border_mode='same',subsample=(1, 1, 1))(x)
    x = Conv3D(512, (3, 3, 3), activation='relu',border_mode='same',subsample=(1, 1, 1))(x)
    x = ZeroPadding3D(padding=(0, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),border_mode='valid')(x)
    x = Flatten()(x)

    # FC layers group
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_ = Dense(nb_class, activation=activation)(x)
    
    return input_, output_

if __name__ == '__main__':
    input_, output_ = C3D()
    model = Model(inputs=input_, outputs=output_)
    model.summary()
