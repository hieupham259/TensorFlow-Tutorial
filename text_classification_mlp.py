import os
import re
import sys
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers, losses

# Download IMDB dataset
def download_data():
    if os.path.exists("./aclImdb_v1"):
        return os.path.join("./aclImdb_v1", "aclImdb", "train"), os.path.join("./aclImdb_v1", "aclImdb", "test")
    
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir=".", cache_subdir="")
    dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb_v1")
    train_dir = os.path.join(dataset_dir, "aclImdb", "train")
    test_dir = os.path.join(dataset_dir, "aclImdb", "test")
    remove_dir = os.path.join(train_dir, "unsup")
    if os.path.exists(remove_dir):
        shutil.rmtree(remove_dir)

# Datasets
def setup_dataset(train_dir, test_dir):
    batch_size = 32
    seed = 42
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=seed,
    )
    amount_batches = tf.data.experimental.cardinality(raw_train_ds)
    print(f'Number of batches in the training dataset: {amount_batches}')

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=seed,
    )
    amount_batches = tf.data.experimental.cardinality(raw_val_ds)
    print(f'Number of batches in the validation dataset: {amount_batches}')

    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size,
    )
    amount_batches = tf.data.experimental.cardinality(raw_test_ds)
    print(f'Number of batches in the test dataset: {amount_batches}')

    return raw_train_ds, raw_val_ds, raw_test_ds

# Text preprocessing

def custom_standardization(input_data: tf.Tensor) -> tf.Tensor:
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


def build_mlp_model(
    raw_train_ds: tf.data.Dataset,
    raw_val_ds: tf.data.Dataset,
    raw_test_ds: tf.data.Dataset,
):
    max_features = 20000  # vocab size for TF-IDF
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode="tf-idf",
    )
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        return vectorize_layer(text), label

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    # MLP model on TF-IDF
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(max_features,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer="adam",
        metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)],
    )

    return model, vectorize_layer, train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_dir, test_dir = download_data()
    raw_train_ds, raw_val_ds, raw_test_ds = setup_dataset(train_dir, test_dir)
    model, vectorize_layer, train_ds, val_ds, test_ds = build_mlp_model(
        raw_train_ds, raw_val_ds, raw_test_ds
    )
    print(model.summary())

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    loss, accuracy = model.evaluate(test_ds)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    # Export model that takes raw text
    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation("sigmoid"),
    ])
    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["accuracy"],
    )

    examples = tf.constant(
        [
            "The movie was great!",
            "The movie was okay.",
            "The movie was terrible...",
        ]
    )
    preds = export_model.predict(examples)
    print("Predictions:", preds)
