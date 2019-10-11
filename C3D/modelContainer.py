from keras.models import *
from keras.layers import *
from keras.engine.network import Network

def C3D(input_shape=(20,224,224,3),nb_class=1, activation='tanh'):
    input_ = Input(shape=input_shape)    
    x = Conv3D(64, (3, 3, 3), activation='relu',border_mode='same'
               ,subsample=(1, 1, 1),input_shape=input_shape)(input_)
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
    
    return Network(input_, output_, name='C3D')

if __name__ == '__main__':
    input_shape=(20,224,224,3)
    c3d = C3D(input_shape)
    input1 = Input(shape=(input_shape))
    input2 = Input(shape=(input_shape))
    x1 = c3d(input1)
    x2 = c3d(input2)
    
    pred = Subtract()([x1, x2])
    model = Model(inputs=[input1, input2], outputs=pred)
    model.summary()
