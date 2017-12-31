from keras.layers import Input, BatchNormalization, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D
from keras.layers import concatenate
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.models import Model


def get_model(model_name, input_dim, n_feature=0, learning_rate=0.0001, n_class=1):
    return eval(model_name)(input_dim, n_feature, learning_rate, n_class)


def nn1(input_dim, n_feature, learning_rate=0.0001, n_class=1):
    # create a NN model
    nn_input = Input(shape=(input_dim,))

    x = Dense(1024)(nn_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(n_class, activation='linear')(x)

    # this is the model we will train
    model = Model(inputs=nn_input, outputs=predictions)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=RMSprop(lr=learning_rate), loss='mean_squared_error')

    return model

