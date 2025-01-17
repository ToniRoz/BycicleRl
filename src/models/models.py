from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
import os

"""
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
"""


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

    def fit(self, state, target, batch_size, verbose):
        return self.model.fit(state, target, batch_size=batch_size ,verbose=verbose)

    def save(self, dir_path, filename):
        """
        Save the model to the specified directory with the given filename.
        """
        os.makedirs(dir_path, exist_ok=True)  # Create the directory if it doesn't exist
        full_path = os.path.join(dir_path, filename)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        self.model.save(full_path)
        print(f"Model saved to {full_path}")

    def load(self, dir_path, filename):
        """
        Load the model from the specified directory with the given filename.
        """
        full_path = os.path.join(dir_path, filename)
        if os.path.exists(full_path):
            self.model = load_model(full_path)
            print(f"Model loaded from {full_path}")
        else:
            raise FileNotFoundError(f"The file {full_path} does not exist.")

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
