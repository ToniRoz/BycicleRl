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


# this might not be as useful right now but with one or multiple model classes the code could be cleaner
class NN_Model:
    def __init__(self, input_shape, action_space, learning_rate, layer_sizes=[512, 256, 64]):
        X_input = Input(input_shape)
        X = X_input

        for size in layer_sizes:
            if size > 0:
                X = Dense(size, activation="relu", kernel_initializer='random_normal', bias_initializer='zeros')(X)

        X = Dense(action_space, activation="linear", kernel_initializer='random_normal', bias_initializer='zeros')(X)

        self.model = Model(inputs=X_input, outputs=X, name='model')
        self.model.compile(loss="mean_squared_error",
                    optimizer=RMSprop(learning_rate=learning_rate, rho=0.95, epsilon=0.01),
                    metrics=["accuracy"])

        self.model.summary()

    def predict(self, state):
        return self.model.predict(state)

    def fit(self, state, target):
        return self.model.fit(state, target, verbose=0)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)