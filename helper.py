# Import Library
from ultralytics import YOLO
import streamlit as st
import cv2
import time
import threading
import PIL
import io
import base64
from gtts import gTTS
import pygame
from pytube import YouTube

from tempfile import NamedTemporaryFile
from streamlit_webrtc import (
    VideoTransformerBase,
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorFactory,
)

# Local File
import settings
import turn


def load_model(model_path):
    model = YOLO(model_path)
    return model


def showDetectFrame(conf, model, st_frame, image, caption=None):
    # Predict the objects in the image using the YOLOv8 model
    res = model.predict(image, conf=conf)
    # Get the results
    boxes = res[0].boxes
    labels = res[0].names  # Assuming res[0].names provides the class names
    res_plotted = res[0].plot()

    detected_labels = []
    for box in boxes:
        label = labels[int(box.cls)]
        if label not in detected_labels:
            detected_labels.append(label)

    st_frame.image(res_plotted, caption=caption, channels="BGR", use_column_width=True)

    # Function to generate audio from detected labels using gtts and streamlit
    def get_audio_bytes():
        text = (
            " ".join(detected_labels)
            if detected_labels
            else "Tidak ada objek yang terdeteksi"
        )

        # Generate audio with gTTS and save it to a BytesIO buffer
        tts = gTTS(text, lang="id")
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)  # Move the cursor to the start of the buffer

        return audio_buffer

    # Generate audio and play it using HTML for autoplay
    audio_buffer = get_audio_bytes()
    audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """
    st.components.v1.html(audio_html, height=0)


def play_youtube(conf, model):
    source_youtube = st.text_input("Silahkan Masukan Link YouTube")
    st.markdown(
        """
        <style>
            #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.block-container.st-emotion-cache-gh2jqd.ea3mdgi5 > div > div > div > div:nth-child(5) > div > div.st-ak.st-as.st-ar.st-am.st-av.st-aw.st-ax.st-ay.st-az.st-b0.st-b1.st-b2.st-ij.st-b4.st-b5.st-an.st-ao.st-ap.st-aq.st-ae.st-af.st-ag.st-ef.st-ai.st-aj.st-fa.st-fb.st-fc.st-fd.st-fe.st-ik.st-il > div {
                background-color: #262730;
                border-radius: 5px;
            }
        </style>""",
        unsafe_allow_html=True,
    )

    if st.button("Deteksi"):
        with st.spinner("Sedang Mendeteksi Objek..."):
            try:
                yt = YouTube(source_youtube)
                stream = yt.streams.filter(file_extension="mp4", res=720).first()
                vid_cap = cv2.VideoCapture(stream.url)

                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        showDetectFrame(
                            conf, model, st_frame, image, caption="Deteksi Video"
                        )
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.error("Ada Kesalahan Saat Memproses Link: " + str(e))


class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf
        self.last_detected_labels = set()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        res = self.model.predict(img, show=False, conf=self.conf)
        res_plotted = res[0].plot()

        detected_labels = set()
        for box in res[0].boxes:
            label = res[0].names[int(box.cls)]
            detected_labels.add(label)

        if detected_labels != self.last_detected_labels:
            self.last_detected_labels = detected_labels
            threading.Thread(target=self.speak_labels, args=(detected_labels,)).start()

        return res_plotted

    def speak_labels(self, detected_labels):
        text = (
            " ".join(detected_labels)
            if detected_labels
            else "Tidak ada objek yang terdeteksi"
        )

        tts = gTTS(text, lang="id")
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)  # Move the cursor to the start of the buffer

        pygame.mixer.init()
        pygame.mixer.music.load(audio_buffer, "mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.quit()


def live(conf, model):
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": turn.get_ice_servers(),
            "iceTransportPolicy": "relay",
        },
        video_transformer_factory=lambda: VideoTransformer(model, conf),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def process_uploaded_video(conf, model):
    uploaded_video = st.file_uploader(
        "Silahkan Upload Video", type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            temp_video_path = temp_file.name

        with open(temp_video_path, "rb") as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)

        if st.button("Deteksi"):
            with st.spinner("Sedang Mendeteksi Objek..."):
                try:
                    vid_cap = cv2.VideoCapture(temp_video_path)
                    st_frame = st.empty()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            showDetectFrame(
                                conf, model, st_frame, image, caption="Deteksi Video"
                            )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    source_vid = st.selectbox(
        "Silahkan Pilih Video yang Sudah Disediakan", settings.VIDEOS_DICT.keys()
    )

    with open(settings.VIDEOS_DICT.get(source_vid), "rb") as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.button("Deteksi Video"):
        with st.spinner("Sedang Mendeteksi Objek..."):
            try:
                vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        showDetectFrame(
                            conf, model, st_frame, image, caption="Deteksi Video"
                        )
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.error("Ada Kesalahan Saat Proses Video: " + str(e))


def take_picture(conf, model):
    picture = st.camera_input("Silahkan Mengambil Gambar")

    if picture:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(picture.read())
            temp_pict_path = temp_file.name

        if st.button("Deteksi Foto"):
            with st.spinner("Sedang Mendeteksi Objek..."):
                try:
                    vid_cap = cv2.VideoCapture(temp_pict_path)
                    st_frame = st.empty()
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            showDetectFrame(
                                conf, model, st_frame, image, caption="Deteksi Gambar"
                            )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error("Error loading video: " + str(e))


def up_picture(conf, model):
    source_img = st.file_uploader(
        "Silahkan Mengupload Gambar", type=("jpg", "jpeg", "png")
    )

    def proses():
        if source_img is not None:
            st_frame = st.empty()
            uploaded_image = PIL.Image.open(source_img)
            showDetectFrame(
                conf,
                model,
                st_frame,
                uploaded_image,
                caption="Hasil Deteksi Gambar",
            )

    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                st.image(
                    default_image_path, caption="Gambar Awal", use_column_width=True
                )
            else:
                st.image(source_img, caption="Gambar Awal", use_column_width=True)
                if st.button("Deteksi", help="Klik tombol ini untuk deteksi"):
                    with st.spinner("Sedang Mendeteksi Objek..."):
                        time.sleep(2)
                        with col2:
                            proses()

        except Exception as ex:
            st.error("Ada Kesalahan Saat Membaca File")
            st.error(ex)

    with col2:
        if source_img is None:
            default_image_path_result = str(settings.DEFAULT_DETECT_IMAGE)
            st.image(
                default_image_path_result,
                use_column_width=True,
                caption="Hasil Deteksi",
            )


def vid_help():
    html_temp_about1 = """
        <div>
            <h6 style="color: white">
                Untuk Mempermudah Penggunaan, Silahkan Menonton Tutorial Berikut 😉
            </h6>
        </div>
    """
    st.markdown(html_temp_about1, unsafe_allow_html=True)

    yt_url = "https://youtu.be/qN_ZyDgk3GU?si=QSVJw67gKpi2msyj"
    yt = YouTube(yt_url)
    st.video(yt_url)
