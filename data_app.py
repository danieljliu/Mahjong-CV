import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import *
import time
import cv2
import av


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

        # noinspection PyUnresolvedReferences
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write('Device:', device)

    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 43)
    net.load_state_dict(torch.load('mahjong_resnet.pt', map_location=torch.device(device)))
    return net.eval();


def load_image(image_file):
    img = Image.open(image_file)
    return img


def img_to_tensor(image_file):
    """load image, returns cuda tensor"""
    loader = transforms.Compose([transforms.ToTensor()])

    image = Image.open(image_file)
    image = image.convert('RGB')
    image = loader(image)
    image = image.unsqueeze(0)
    return image


def convert_to_label(idx):
    label_df = pd.read_csv('label.csv')
    idx = idx.item()
    select = label_df.loc[label_df['label-index'] == idx].values[0][0]
    st.write(select)


def classify_image(image):
    net = load_model()
    image = img_to_tensor(image)
    with torch.no_grad():
        logits = net.forward(image)
    ps = torch.exp(logits)
    _, predTest = torch.max(ps, 1)
    # print(torch.argmax(ps))
    convert_to_label(torch.argmax(ps))


def load_yolo5():
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_full.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_symbols.pt')
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    #model = torch.hub.load('ultralytics/yolov5', 'custom', path='mahjong_resnet.pt')
    return model


def classify_yolo5(model, image):
    results = model(image)
    st.image(results.render()[0])
    st.write(results)


def main():
    st.title("Mahjong Tile Classifier")
    menu = ["Image", "Computer Vision"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Image':
        model = load_yolo5()

        st.subheader('Image')
        image_file = st.file_uploader("Upload Mahjong Tile", type=['png', 'jpg', 'jpeg'])
        pic = st.camera_input("Take a picture")
        if pic:
            #st.image(load_image(pic))
            start_time = time.time()
            classify_yolo5(model, load_image(pic))
            end_time = time.time()
            st.write('Classification took ', end_time - start_time, 'seconds')


        if image_file:
            file_details = {'filename': image_file.name, 'filetype': image_file.type, 'filesize': image_file.size}
            st.write(file_details)
            st.image(load_image(image_file))
            start_time = time.time()
            classify_image(image_file)
            end_time = time.time()
            st.write('Classification took ', end_time - start_time, 'seconds')

    if choice == 'Computer Vision':
        st.subheader('Computer Vision')
        webrtc_streamer(key="example", video_processor_factory=VideoProcessor)


if __name__ == '__main__':
    main()
