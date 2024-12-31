from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop


def NNModel(input_shape, action_space, learning_rate, layer_sizes=[512, 256, 64]):
    X_input = Input(input_shape)
    X = X_input

    for size in layer_sizes:
        if size > 0:
            X = Dense(size, activation="relu", kernel_initializer='random_normal', bias_initializer='zeros')(X)

    X = Dense(action_space, activation="linear", kernel_initializer='random_normal', bias_initializer='zeros')(X)

    model = Model(inputs=X_input, outputs=X, name='model')
    model.compile(loss="mean_squared_error",
                  optimizer=RMSprop(learning_rate=learning_rate, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])

    model.summary()
    return model