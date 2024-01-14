#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install -q git+https://github.com/tensorflow/docs
#pip install imutils
#from tensorflow_docs.vis import embed
from tensorflow import keras
from imutils import paths
from sklearn.metrics import confusion_matrix  #混同行列
from sklearn.metrics import accuracy_score, precision_score #適合率
from sklearn.metrics import recall_score, f1_score #再現率,F1スコア
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os
import warnings
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224
BATCH_SIZE = 512
EPOCHS = 200

MAX_SEQ_LENGTH = 200
NUM_FEATURES = 1024

train_df = pd.read_csv("/home/arai/SoccerAct10/train/train-1000.csv")
test_df = pd.read_csv("/home/arai/SoccerAct10/test/test-100.csv")
train_df= train_df.sample(frac=1, random_state=0)
train_df= train_df.sample(frac=1, random_state=0)
print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

train_df.head(10)

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    
    feature_extractor = keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        pooling=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    
    preprocess_input = keras.applications.xception.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    #outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(1024, activation="relu")(outputs)
    #outputs = keras.layers.Flatten()(outputs)
    return keras.Model(inputs, outputs, name="feature_extractor")

import warnings

feature_extractor = build_feature_extractor()

feature_extractor.summary()

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["event"])
)
print(label_processor.get_vocabulary())

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    #labels = df["event"].values
    #labels = label_processor(labels[..., None]).numpy()
    labels = pd.get_dummies(df["event"])

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


train_data, train_labels = prepare_all_videos(train_df, "/home/arai/SoccerAct10/train")
test_data, test_labels = prepare_all_videos(test_df, "/home/arai/SoccerAct10/test")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

# Utility for our sequence model.
def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
    x = keras.layers.GRU(1024, return_sequences=None)(
        frame_features_input, mask=mask_input
    )
    #x = keras.layers.LSTM(512, return_sequences=True)(
        #frame_features_input, mask=mask_input
    #)
    #x = keras.layers.LSTM(512)(x)
    x = keras.layers.Dropout(0.5)(x)
    #x = keras.layers.Dense(16, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    #rnn_model.compile(
        #loss="categorical_crossentropy", optimizer=SGD(learning_rate=0.0001, momentum=0.9), metrics=["accuracy"]
    #)
    return rnn_model


# Utility for running experiments.
def run_experiment():
    filepath = "train/cp.ckpt"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    seq_model.compile(
        loss="categorical_crossentropy",  optimizer=SGD(learning_rate=0.0001, momentum=0.9),  metrics=["accuracy"]
    )
    #optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999),
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        #validation_split=0.3,
        validation_data=([test_data[0], test_data[1]], test_labels),
        #batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    
    metrics = ['loss', 'accuracy']  # 使用する評価関数を指定

    plt.figure(figsize=(10, 5))  # グラフを表示するスペースを用意

    for i in range(len(metrics)):

        metric = metrics[i]

        plt.subplot(1, 2, i+1)  # figureを1×2のスペースに分け、i+1番目のスペースを使う
        plt.title(metric)  # グラフのタイトルを表示
    
        plt_train = history.history[metric]  # historyから訓練データの評価を取り出す
        plt_test = history.history['val_' + metric]  # historyからテストデータの評価を取り出す
    
        plt.plot(plt_train, label='training')  # 訓練データの評価をグラフにプロット
        plt.plot(plt_test, label='test')  # テストデータの評価をグラフにプロット
        plt.legend()  # ラベルの表示
    
    plt.show()  # グラフの表示
    
    return history, seq_model


_, sequence_model = run_experiment()

pred = sequence_model.predict(test_data)
pred = np.argmax(pred, axis=1)
labels = test_df["event"].values
labels = label_processor(labels[..., None]).numpy()

print("accuracy score:",accuracy_score(labels, pred))
print("precision score:",precision_score(labels, pred,average='micro'))
print("recall score:",recall_score(labels, pred,average='micro'))
print("f1 score:",f1_score(labels, pred,average='micro'))
cm = confusion_matrix(labels, pred)
print(cm)
ax= plt.subplot()
sns.heatmap(cm, annot=True, cmap='Blues', ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['corner', 'foul','freekick','goalkick','longpass','ontarget','penalty','shortpass','substitution','throw-in']); ax.yaxis.set_ticklabels(['corner', 'foul','freekick','goalkick','longpass','ontarget','penalty','shortpass','substitution','throw-in']);
#plt.figure(figsize=(15, 10))
#sns.heatmap(cm, annot=True, cmap='Blues')
plt.savefig('/home/arai/sklearn_confusion_matrix.png')
#plt.show()

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("home/arai/video2/test/", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames


def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("animation.gif")


# test_video = np.random.choice(test_df["video_name"].values.tolist())
test_video ="Clearence_1_562_2015-02-21.mp4"
print(f"Test video path: {test_video}")
test_frames = sequence_prediction(test_video)
to_gif(test_frames[:MAX_SEQ_LENGTH])

