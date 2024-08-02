import tensorflow as tf


class Model:

    @staticmethod
    def create_model_mlp():
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(784,)),
                tf.keras.layers.Dense(units=128, activation='relu'),
                tf.keras.layers.Dense(10, activation="softmax")
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        return model

    @staticmethod
    def create_model_cnn(shape):
        model = tf.keras.models.Sequential([

            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy']
                      )

        return model
