from keras.optimizers import Adadelta
from keras.layers import Input, Dense, Dropout
from keras.layers import Activation, BatchNormalization
from keras.models import Model
from keras.regularizers import l1


def get_model(model_name, input_dim, encoder_dim, **kwargs):
    return eval(model_name)(input_dim, encoder_dim, **kwargs)


def mlp2(input_dim, encoder_dim, learning_rate=1e-4, dropout=0.3):
    inputs = Input(shape=(input_dim,))
    encoded = Dense(1024)(inputs)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)
    encoded = Dropout(dropout)(encoded)
    encoded = Dense(1024)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)
    encoded = Dropout(dropout)(encoded)
    encoded = Dense(1024)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)
    encoded = Dropout(dropout)(encoded)

    encoded = Dense(encoder_dim)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)

    decoded = Dense(1024)(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)
    decoded = Dropout(dropout)(decoded)
    decoded = Dense(1024)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)
    decoded = Dropout(dropout)(decoded)
    decoded = Dense(1024)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)
    decoded = Dropout(dropout)(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adadelta(lr=learning_rate), loss='binary_crossentropy')

    encoder = Model(inputs, encoded)

    return autoencoder, encoder


def mlp1(input_dim, encoder_dim, learning_rate=1e-4):
    inputs = Input(shape=(input_dim,))
    encoded = Dense(1024, activation='relu')(inputs)
    encoded = Dense(512, activation='relu')(encoded)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(encoder_dim, activation='relu')(encoded)

    decoded = Dense(256, activation='relu')(encoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(1024, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adadelta(lr=learning_rate), loss='binary_crossentropy')

    encoder = Model(inputs, encoded)

    return autoencoder, encoder


