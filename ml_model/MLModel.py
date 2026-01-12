import tensorflow as tf


class Model:
    @staticmethod
    def create_model(cm):
        if cm.model_type == "MLP":
            return Model.create_model_mlp(cm.shape)
        else:
            return Model.create_model_cnn(cm.shape)


    @staticmethod
    def create_model_mlp(shape):
        model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(shape[0]*shape[1]),),

                tf.keras.layers.Dense(units=192, activation='relu'),
                tf.keras.layers.Dense(units=96, activation='relu'),
                tf.keras.layers.ReLU(),


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

            tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                                   kernel_initializer='he_normal'),
            tf.keras.layers.LayerNormalization(epsilon=1e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                   kernel_initializer='he_normal'),
            tf.keras.layers.LayerNormalization(epsilon=1e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(2),


            tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                                   kernel_initializer='he_normal'),
            tf.keras.layers.LayerNormalization(epsilon=1e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                   kernel_initializer='he_normal'),
            tf.keras.layers.LayerNormalization(epsilon=1e-5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(2),


            tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                                   kernel_initializer='he_normal'),
            tf.keras.layers.LayerNormalization(epsilon=1e-5),
            tf.keras.layers.ReLU(),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(128),
            tf.keras.layers.ReLU(),


            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy']
                      )

        return model
