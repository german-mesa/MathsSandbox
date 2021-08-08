import numpy as np

from tensorflow import keras


def main():
    xs = np.linspace(1, 6, num=6)
    ys = 0.5 * (xs + 1)

    model = keras.Sequential([
        keras.layers.Dense(units=1, input_shape=[1])
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss=keras.losses.mse)

    model.fit(x=xs, y=ys, epochs=500, verbose=1)

    print(f"A house with 7 bedrooms is about {model.predict([7.0])[0]} hundreds of thousands")


if __name__ == '__main__':
    main()
