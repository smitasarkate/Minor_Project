# Project by Prem, Date: 10-09-2021


# importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import urllib
import os
import json

import numpy as np
import imutils
from model import build_model, load_weights

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

import time


def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/Import-Prem/Streamlit_project/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


readme_text = st.markdown(get_file_content_as_string('instructions.md'))


# function for landmark identification
def landmark_detection():
    st.title('Landmark identification')
    st.subheader(
        "This project takes the input image and identifies the landmark in the image[only Asia's landmarks].")
    uploaded_file = st.file_uploader("Upload a image", type='jpg')

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.')
        my_bar = st.progress(0)
        TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
        LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
        IMAGE_SHAPE = (321, 321)

        classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                         input_shape=IMAGE_SHAPE +
                                                         (3,),
                                                         output_key="predictions:logits")])
        df = pd.read_csv(LABEL_MAP_URL)
        label_map = dict(zip(df.id, df.name))

        img = image.resize(IMAGE_SHAPE)
        img = np.array(img)/255.0

        img = img[np.newaxis, ...]
        prediction = classifier.predict(img)
        st.header(label_map[np.argmax(prediction)])
        my_bar.progress(100)


# function for object detection
def mask_detection():

    st.title('Realtime Mask Detection')
    st.subheader("Once you Press the start button You can see a real time window which detects a human wearing mask or not with a confidence interval.")
    st.write("Note - Press 'q' to exit the window")
    if st.button("Start mask detection"):

        def detect_and_predict_mask(frame, faceNet, maskNet):
            # grab the dimensions of the frame and then construct a blob
            # from it
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                         (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the face detections
            faceNet.setInput(blob)
            detections = faceNet.forward()
            print(detections.shape)

            # initialize our list of faces, their corresponding locations,
            # and the list of predictions from our face mask network
            faces = []
            locs = []
            preds = []

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the detection
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 224x224, and preprocess it
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)

                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

            # only make a predictions if at least one face was detected
            if len(faces) > 0:
                # for faster inference we'll make batch predictions on *all*
                # faces at the same time rather than one-by-one predictions
                # in the above `for` loop
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)

            # return a 2-tuple of the face locations and their corresponding
            # locations
            return (locs, preds)

        # load our serialized face detector model from disk
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model from disk
        maskNet = load_model("mask_detector.model")

        # initialize the video stream
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(
                    label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()


def Music_Generation():
    st.title('Music Generation')
    st.write('Step 1 -> Generate abc notation')
    st.write(
        'Step 2 -> Copy the abc notation and paste it on this [link](https://www.abcjs.net/abcjs-editor.html)')
    st.write('Step 3 -> Play and enjoy the music generated by neural networks :/)')
    st.text('Note - More the epoch, the more quality of music; Increasing lenght and epoch may result in slower output')
    epochs = st.number_input('Epochs', min_value=1,
                             max_value=100, value=80, step=10)
    lenght = st.number_input(
        'Lenght of music', min_value=1, max_value=1200, value=512, step=1)
    seed = st.text_input("Starting character", '')
    DATA_DIR = './data'
    MODEL_DIR = './model'

    def build_sample_model(vocab_size):
        model = Sequential()
        model.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))
        for i in range(3):
            model.add(LSTM(256, return_sequences=(i != 2), stateful=True))
            model.add(Dropout(0.2))

        model.add(Dense(vocab_size))
        model.add(Activation('softmax'))
        return model

    def sample(epoch, header, num_chars):
        with open(os.path.join(DATA_DIR, 'char_to_idx.json')) as f:
            char_to_idx = json.load(f)
        idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
        vocab_size = len(char_to_idx)

        model = build_sample_model(vocab_size)
        load_weights(epoch, model)
        model.save(os.path.join(MODEL_DIR, 'model.{}.h5'.format(epoch)))

        sampled = [char_to_idx[c] for c in header]
        print(sampled)

        for i in range(num_chars):
            batch = np.zeros((1, 1))
            if sampled:
                batch[0, 0] = sampled[-1]
            else:
                batch[0, 0] = np.random.randint(vocab_size)
            result = model.predict_on_batch(batch).ravel()
            sample = np.random.choice(range(vocab_size), p=result)
            sampled.append(sample)

        return ''.join(idx_to_char[c] for c in sampled)
    st.write(sample(epochs, seed, lenght), unsafe_allow_html=True)


st.sidebar.title("What to do")
app_mode = st.sidebar.selectbox("Choose the app mode",
                                ["Show Instruction", "Landmark identification", "Mask detection", "Music Generation", "Show the source code", "About"])

if app_mode == "Show Instructions":
    st.sidebar.success('Select a operation to perform')

elif app_mode == "Landmark identification":
    readme_text.empty()
    landmark_detection()

elif app_mode == "Show the source code":
    readme_text.empty()
    st.code(get_file_content_as_string("streamlit_app.py"))

elif app_mode == "Mask detection":
    readme_text.empty()
    mask_detection()

elif app_mode == "Music Generation":
    readme_text.empty()
    Music_Generation()

elif app_mode == "About":
    readme_text.empty()
    st.markdown(get_file_content_as_string('about.md'))
