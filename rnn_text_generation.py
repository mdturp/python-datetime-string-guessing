"""Predict the python datetime format string using an RNN"""

import datetime
from enum import unique

import tensorflow as tf
import numpy as np

import supported_datetime_formats as sdf

STARTING_DATE = datetime.date(2002, 12, 4)
RNG = np.random.default_rng(1)
LABEL_CLASSES = [s["format"] for s in sdf.formats]


def create_dataset(n_samples):
    x = []
    y = []
    y_integer = []
    for _ in range(n_samples):
        days = int(RNG.integers(-100000, 100000))
        seconds = int(RNG.integers(-100000, 100000))
        delta = datetime.timedelta(days=days, seconds=seconds)
        new_day = STARTING_DATE + delta
        for idx, l in enumerate(LABEL_CLASSES):
            x.append(new_day.strftime(l))
            y_integer.append(idx)
            y.append(l)
    idx_shuffle = RNG.shuffle(list(range(len(x))))

    return (np.squeeze(np.array(x)[idx_shuffle]),
            np.squeeze(np.array(y)[idx_shuffle]),
            np.squeeze(np.array(y_integer)[idx_shuffle]))


def text_from_ids(ids, chars_from_ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def main():
    x_train, y_train_label, y_train_idx = create_dataset(10000)

    print("-------------------------")
    print(f"{x_train[0]} ----> {y_train_label[0]}")
    print("-------------------------")


if __name__ == "__main__":
    main()
