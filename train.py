import tensorflow as tf
import argparse
import os
from tensorflow.python.lib.io import file_io
from utils import request_deploy_api


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", default=64, type=float)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--deploy", default=False, type=bool)
    return parser.parse_args()


def load_data():
    return tf.keras.datasets.mnist.load_data()


def normalize(data):
    return data / 225.0


def get_model(args):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(args.units, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=args.optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train():

    args = get_args()
    model = get_model(args)

    (train_x, train_y), (test_x, test_y) = load_data()
    train_x, test_x = normalize(train_x), normalize(test_x)

    print("Training...")
    model.fit(train_x, train_y, validation_split=0.2, epochs=10)

    loss, acc = model.evaluate(test_x, test_y)
    print(f"model test-loss={loss:.4f} test-acc={acc:.4f}")
    if args.deploy:
        deploy_model(model, args)


def arg_to_str(args):
    return "_".join([f"{x[0]}_{x[1]}" for x in vars(args).items()][:-1])


def deploy_model(model, args):
    gcp_bucket = os.getenv("GCS_BUCKET")
    bucket_path = os.path.join(gcp_bucket, "mnist")
    save_path = f"{arg_to_str(args)}.h5"
    print(f"saving model {save_path}")
    model.save(save_path)

    gs_path = os.path.join(bucket_path, save_path)
    with file_io.FileIO(save_path, mode="rb") as input_file:
        with file_io.FileIO(gs_path, mode="wb+") as output_file:
            output_file.write(input_file.read())
    print(f"model save success!")

    request_deploy_api(gs_path)
    print(f"Trigger Deploy success!")


if __name__ == "__main__":
    train()
