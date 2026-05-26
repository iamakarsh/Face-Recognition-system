from pathlib import Path
from html import escape
import sys
from tempfile import NamedTemporaryFile

import cv2
import streamlit as st
from PIL import Image

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

    WEBRTC_AVAILABLE = True
except ImportError:
    av = None
    RTCConfiguration = None
    VideoProcessorBase = object
    webrtc_streamer = None
    WEBRTC_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_CONFIDENCE, FACE_MODEL_PATH, IMAGE_EXTENSIONS, MODEL_OPTIONS, OUTPUT_DIR, VIDEO_EXTENSIONS
from app.detector import YoloDetector
from app.utils import bgr_to_rgb, pil_to_bgr, unique_output_path


st.set_page_config(page_title="Vision Detection Studio", page_icon=":camera:", layout="wide")

st.markdown(
    """
    <style>
    html { scroll-behavior: smooth; }
    .stApp {
        background:
            radial-gradient(circle at 12% 12%, rgba(77, 163, 255, 0.13), transparent 28rem),
            radial-gradient(circle at 92% 8%, rgba(62, 205, 137, 0.11), transparent 24rem),
            #0b0f16;
    }
    .block-container { padding-top: 1.4rem; max-width: 1260px; }
    h1 { margin-bottom: 0.2rem; }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.4rem;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #d7dce6;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, #171d27 0%, #121721 100%);
        border: 1px solid #303747;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.2);
    }
    [data-testid="stSidebar"] {
        background: #0f141d;
        border-right: 1px solid #242b39;
    }
    .hero {
        border: 1px solid #303747;
        border-radius: 8px;
        padding: 22px 24px;
        background:
            linear-gradient(135deg, rgba(20, 26, 36, 0.96) 0%, rgba(29, 36, 48, 0.95) 54%, rgba(22, 42, 34, 0.95) 100%);
        margin-bottom: 18px;
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.24);
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 760;
        line-height: 1.1;
        margin: 0 0 6px 0;
    }
    .hero-copy {
        color: #b8bfcc;
        font-size: 1rem;
        margin: 0;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }
    .chip {
        border: 1px solid #3a4253;
        border-radius: 999px;
        color: #dce3ef;
        background: rgba(255, 255, 255, 0.045);
        padding: 6px 10px;
        font-size: 0.82rem;
        line-height: 1;
    }
    .section-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #eef2f8;
        margin: 10px 0 6px 0;
    }
    .empty-state {
        border: 1px dashed #3a4253;
        border-radius: 8px;
        padding: 22px;
        color: #aeb6c4;
        background: rgba(255, 255, 255, 0.03);
        margin-top: 12px;
    }
    .history-item {
        border: 1px solid #303747;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        background: #121721;
    }
    .history-title {
        color: #f0f4fb;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .history-meta {
        color: #a8adba;
        font-size: 0.82rem;
        margin-top: 3px;
    }
    .small-muted { color: #a8adba; font-size: 0.95rem; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #10151e;
        border: 1px solid #283142;
        border-radius: 8px;
        padding: 6px;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        padding: 8px 18px;
        color: #b7bfce;
    }
    .stTabs [aria-selected="true"] {
        background: #202938;
        color: #ffffff;
    }
    .stButton > button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 8px;
        border: 1px solid #384256;
        background: #202938;
        color: #f4f7fb;
    }
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stFileUploader"] section {
        border-radius: 8px;
    }
    img {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Vision Detection Studio</div>
        <p class="hero-copy">Custom face detection, pretrained object detection, and live pose keypoints in one local workspace.</p>
        <div class="chip-row">
            <span class="chip">Live camera</span>
            <span class="chip">Image analysis</span>
            <span class="chip">Video processing</span>
            <span class="chip">YOLOv8</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

model_mode = st.sidebar.selectbox("Detection model", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[model_mode]
variant_label = st.sidebar.selectbox("Model quality", list(selected_model["variants"].keys()))
model_reference = selected_model["variants"][variant_label]
confidence_default = 0.45 if model_mode == "Person / Object Detection" else DEFAULT_CONFIDENCE
confidence = st.sidebar.slider("Confidence", 0.05, 0.95, confidence_default, 0.05)
image_size = st.sidebar.select_slider("Image size", options=[320, 480, 640, 800, 960, 1280], value=640)
max_detections = st.sidebar.slider("Max detections", 1, 300, 100, 1)
metric_label = selected_model["metric_label"]
short_label = selected_model["short_label"]

st.sidebar.info(selected_model["description"])

if model_mode == "Face Detection" and not FACE_MODEL_PATH.exists():
    st.error(f"Face model not found at {FACE_MODEL_PATH}")
    st.stop()


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_detector(model_reference_value: str) -> YoloDetector:
    return YoloDetector(model_reference_value)


detector = load_detector(str(model_reference))

class_ids = None
if model_mode == "Person / Object Detection":
    st.sidebar.divider()
    focus_mode = st.sidebar.radio("Detection focus", ["People only", "All COCO classes", "Custom classes"])
    available_labels = list(detector.names.values())

    if focus_mode == "People only":
        class_ids = detector.class_ids_for_names(["person"])
    elif focus_mode == "Custom classes":
        default_labels = ["person"] if "person" in available_labels else available_labels[:1]
        selected_labels = st.sidebar.multiselect("Classes", available_labels, default=default_labels)
        class_ids = detector.class_ids_for_names(selected_labels)

detector.configure(
    confidence=confidence,
    image_size=image_size,
    max_detections=max_detections,
    class_ids=class_ids,
)

st.sidebar.caption("Higher image size and larger models usually improve accuracy, but they run slower.")

st.sidebar.divider()
st.sidebar.markdown("**Current profile**")
st.sidebar.caption(f"Mode: {model_mode}")
st.sidebar.caption(f"Model: {variant_label}")
st.sidebar.caption(f"Confidence: {confidence:.2f}")
st.sidebar.caption(f"Image size: {image_size}")


class DetectionVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        annotated, count, _ = detector.draw_detections(image)
        cv2.putText(
            annotated,
            f"{metric_label}: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def add_history(source: str, count: int, class_counts: dict[str, int]) -> None:
    labels = ", ".join(f"{label}: {value}" for label, value in sorted(class_counts.items())[:4])
    if not labels:
        labels = "No labels"
    signature = (source, model_mode, variant_label, count, labels)
    if st.session_state.history and st.session_state.history[0].get("signature") == signature:
        return
    st.session_state.history.insert(
        0,
        {
            "signature": signature,
            "source": source,
            "mode": model_mode,
            "model": variant_label,
            "count": count,
            "labels": labels,
        },
    )
    st.session_state.history = st.session_state.history[:8]


def render_history() -> None:
    st.markdown('<div class="section-title">Recent detections</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown(
            '<div class="empty-state">Run a camera snapshot, image upload, or video upload and the latest results will appear here.</div>',
            unsafe_allow_html=True,
        )
        return

    with st.container(height=280):
        for item in st.session_state.history:
            source = escape(str(item["source"]))
            mode = escape(str(item["mode"]))
            model = escape(str(item["model"]))
            labels = escape(str(item["labels"]))
            st.markdown(
                f"""
                <div class="history-item">
                    <div class="history-title">{source} - {item["count"]} detections</div>
                    <div class="history-meta">{mode} - {model}</div>
                    <div class="history-meta">{labels}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_results(original: Image.Image, annotated_bgr, count: int, class_counts: dict[str, int], source_label: str) -> None:
    add_history(source_label, count, class_counts)
    stats = st.columns(3)
    stats[0].metric(metric_label, count)
    stats[1].metric("Mode", short_label)
    stats[2].metric("Image size", image_size)

    left, right = st.columns(2)
    left.image(original, caption=source_label, use_container_width=True)
    right.image(bgr_to_rgb(annotated_bgr), caption="Detection result", use_container_width=True)

    image_rgb = bgr_to_rgb(annotated_bgr)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if ok:
        st.download_button(
            "Download result",
            data=encoded.tobytes(),
            file_name="vision-detection-result.jpg",
            mime="image/jpeg",
        )

    if class_counts:
        with st.expander("Detected labels", expanded=model_mode != "Face Detection"):
            st.dataframe(
                [{"label": label, "count": value} for label, value in sorted(class_counts.items())],
                use_container_width=True,
                hide_index=True,
            )


overview_left, overview_right = st.columns([2.2, 1])
with overview_left:
    st.markdown('<div class="section-title">Active detection profile</div>', unsafe_allow_html=True)
    profile_cols = st.columns(4)
    profile_cols[0].metric("Mode", short_label)
    profile_cols[1].metric("Quality", variant_label.split(" - ")[0])
    profile_cols[2].metric("Confidence", f"{confidence:.2f}")
    profile_cols[3].metric("Image size", image_size)

with overview_right:
    st.markdown('<div class="section-title">Tuning tip</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="empty-state">For fewer false positives, raise confidence and use People only. For body motion, switch to Pose / Body Keypoints.</div>',
        unsafe_allow_html=True,
    )

camera_tab, image_tab, video_tab = st.tabs(["Camera", "Image", "Video"])

with camera_tab:
    camera_mode = st.radio("Camera mode", ["Live stream", "Snapshot"], horizontal=True)

    if camera_mode == "Live stream":
        st.markdown('<div class="small-muted">Start the stream and allow camera access in your browser.</div>', unsafe_allow_html=True)
        if WEBRTC_AVAILABLE:
            webrtc_streamer(
                key=f"vision-live-{model_reference}-{confidence}-{image_size}-{max_detections}-{class_ids}",
                video_processor_factory=DetectionVideoProcessor,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        else:
            st.warning("Live streaming needs streamlit-webrtc. Run `pip install -r requirements.txt`, then restart Streamlit.")

    else:
        st.markdown('<div class="small-muted">Use the browser camera for a quick detection snapshot.</div>', unsafe_allow_html=True)
        camera_image = st.camera_input("Camera")
        if camera_image:
            image = Image.open(camera_image)
            frame = pil_to_bgr(image)
            annotated, count, class_counts = detector.draw_detections(frame)
            show_results(image, annotated, count, class_counts, "Camera photo")
        else:
            st.markdown('<div class="empty-state">Take a snapshot to see side-by-side detection results here.</div>', unsafe_allow_html=True)

with image_tab:
    uploaded_image = st.file_uploader("Upload an image", type=sorted(ext.strip(".") for ext in IMAGE_EXTENSIONS))
    if uploaded_image:
        image = Image.open(uploaded_image)
        frame = pil_to_bgr(image)
        annotated, count, class_counts = detector.draw_detections(frame)
        show_results(image, annotated, count, class_counts, "Original image")
    else:
        st.markdown('<div class="empty-state">Drop an image here to test face, object, or pose detection.</div>', unsafe_allow_html=True)

with video_tab:
    st.markdown('<div class="small-muted">Video processing can take a little while depending on length and model.</div>', unsafe_allow_html=True)
    uploaded_video = st.file_uploader("Upload a video", type=sorted(ext.strip(".") for ext in VIDEO_EXTENSIONS))
    if uploaded_video:
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
            temp_video.write(uploaded_video.getbuffer())
            temp_video_path = Path(temp_video.name)

        output_path = unique_output_path(OUTPUT_DIR, ".mp4")
        with st.spinner("Running detection on video..."):
            max_detections, class_counts = detector.detect_video_file(temp_video_path, output_path)
        temp_video_path.unlink(missing_ok=True)
        add_history("Video upload", max_detections, class_counts)

        stats = st.columns(3)
        stats[0].metric(f"Max {metric_label.lower()}", max_detections)
        stats[1].metric("Mode", short_label)
        stats[2].metric("Image size", image_size)

        st.video(str(output_path))
        st.download_button(
            "Download processed video",
            data=output_path.read_bytes(),
            file_name=f"processed-{Path(uploaded_video.name).stem}.mp4",
            mime="video/mp4",
        )

        if class_counts:
            with st.expander("Detected labels across frames", expanded=model_mode != "Face Detection"):
                st.dataframe(
                    [{"label": label, "count": value} for label, value in sorted(class_counts.items())],
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.markdown('<div class="empty-state">Upload a short video to process it frame by frame and download the annotated result.</div>', unsafe_allow_html=True)

render_history()
