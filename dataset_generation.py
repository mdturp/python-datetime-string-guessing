from cProfile import label
import datetime
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

import supported_datetime_formats as sdf

rng = np.random.default_rng(1)

label_classes = [s["format"] for s in sdf.formats]

# label_classes = [
#     "%d/%m/%y",
#     "%d. %B %Y",
#     "%b %d, %Y %I:%M:%S %p",
#     "%b %d, %Y",
#     "%B %d, %Y %I:%M:%S %p",
#     "%B %d, %Y",
#     "%a %b %d %H:%M:%S %Y",
#     "%A %b %d %H:%M:%S %Y",
#     "%a %b %d %Y %H:%M:%S ",
#     "%A %d-%b %y %H:%M:%S",
#     ]
starting_date = datetime.date(2002, 12, 4)

MAX_TOKENS = 250
OUTPUT_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 16


def create_dataset(n_samples):
    x_text = []
    x = []
    y = []
    y_integer = []
    for _ in range(n_samples):
        days = int(rng.integers(-100000, 100000))
        seconds = int(rng.integers(-100000, 100000))
        delta = datetime.timedelta(days=days, seconds=seconds)
        new_day = starting_date + delta
        for idx, l in enumerate(label_classes):
            x_text.append(new_day)
            x.append(new_day.strftime(l))
            y_integer.append(idx)
            y.append(l)
    idx_shuffle = rng.shuffle(list(range(len(x))))

    return (np.squeeze(np.array(x)[idx_shuffle]),
            np.squeeze(np.array(x_text)[idx_shuffle]),
            np.squeeze(np.array(y)[idx_shuffle]),
            np.squeeze(np.array(y_integer)[idx_shuffle]))


x_train, x_train_text, y_train_label, y_train = create_dataset(10000)
x_test, x_test_text, y_test_label, y_test = create_dataset(1000)

print("-------------")
print(f"{x_train[0]} ------> {y_train_label[0]}")
print("------------")

vectorization_layer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    standardize=None,
    split="character",
    output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
    pad_to_max_tokens=True,
    output_mode="int")


def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return vectorization_layer(text)


vectorization_layer.adapt(x_train)

x_train = vectorization_layer(x_train)
x_test_1 = vectorization_layer(x_test)
print(x_test[0])
print(x_test_1[0])

vocab = vectorization_layer.get_vocabulary()
import json
with open("./model_weights/vocabulary.json", "w") as f:
    json.dump(vocab, f)

# def vectorize_text_by_my_self(x):
#     output = []
#     x_list = list(x)
#     for idx in range(OUTPUT_SEQUENCE_LENGTH):
#         token = 0
#         if idx < len(x_list):
#             token = vocab.index(x_list[idx])
#         output.append(token)
#     return output

# print(vectorize_text_by_my_self(x_test[0]))

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Embedding(MAX_TOKENS + 1, EMBEDDING_DIM),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(len(label_classes))])

# print(model.summary())

# model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(
#                 from_logits=True),
#               optimizer='adam',
#               metrics=tf.metrics.SparseTopKCategoricalAccuracy(k=2))

# epochs = 1
# history = model.fit(
#     x_train, y_train,
#     validation_data=(x_test, y_test),
#     epochs=epochs)

# export_model = tf.keras.Sequential([
#   vectorization_layer,
#   model,
#   tf.keras.layers.Activation('sigmoid')
# ])

# tfjs.converters.save_keras_model(model, "./model_weights")
