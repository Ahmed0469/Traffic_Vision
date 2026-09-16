import shutil
import tempfile
import threading
from pathlib import Path

import cv2
import streamlit as st

from src.object_detection import VideoProcessingCancelled, process_video

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

st.set_page_config(page_title="Traffic Vision | Smart Road Analysis", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap');
    :root { --cream: #e8d7ae; --green: #35d47b; }
    .stApp { background: #090a09; color: #d4d0c7; background-image: linear-gradient(#ffffff04 1px, transparent 1px), linear-gradient(90deg, #ffffff04 1px, transparent 1px); background-size: 72px 72px; }
    header[data-testid="stHeader"] { background: #090a09; }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Instrument Serif', Georgia, serif !important; font-weight: 400 !important; color: var(--cream) !important; }
    h1 { font-size: clamp(3.4rem, 9vw, 7.8rem) !important; line-height: .86 !important; letter-spacing: -.035em !important; }
    h2 { font-size: 2.35rem !important; }
    .eyebrow { color: var(--cream); font: .75rem 'DM Sans', sans-serif; letter-spacing: .15em; text-transform: uppercase; }
    .topbar { height: 76px; border-bottom: 1px solid #232521; display: flex; align-items: center; justify-content: space-between; color: var(--cream); font: 1.45rem 'Instrument Serif', Georgia, serif; }
    .availability { width: 9px; height: 9px; border-radius: 50%; background: var(--green); box-shadow: 0 0 16px var(--green); }
    .hero { min-height: 390px; padding: clamp(4rem, 10vw, 8rem) 0 3rem; border-bottom: 1px solid #232521; position: relative; }
    .hero::after { content: '01'; position: absolute; right: 0; bottom: 1rem; color: #41433d; font: .68rem 'DM Sans', sans-serif; letter-spacing: .16em; }
    .hero p { color: #7e8079; font-size: clamp(1rem, 1.7vw, 1.25rem); line-height: 1.55; max-width: 670px; }
    .hero strong { color: #d9d6ce; font-weight: 500; }
    .panel { min-height: 440px; background: #0d0f0d; border: 1px solid #232521; padding: clamp(1.5rem, 4vw, 3.25rem); }
    .panel h2 { margin-top: 0; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: #0d0f0d; border-color: #232521; border-radius: 0; padding: 1.5rem; }
    [data-testid="stFileUploader"] { border: 1px dashed #45483f; background: #11130f; padding: 1rem; }
    [data-testid="stFileUploader"] section { background: transparent; }
    .stButton > button, .stDownloadButton > button { border-radius: 0; border: 1px solid var(--cream); background: var(--cream); color: #151612; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: #f1e7cb; background: #f1e7cb; color: #151612; }
    .stop-button button { border-color: #4b4d47; background: transparent; color: #aaa99f; }
    .stProgress > div > div > div > div { background-color: var(--green); }
    .warning { color: #c7a866; background: #231e12; border-left: 2px solid #c7a866; padding: .7rem .8rem; font-size: .78rem; line-height: 1.45; }
    .status { color: #777a72; font-size: .75rem; letter-spacing: .05em; }
    .percent { color: var(--cream); font: 1.4rem 'Instrument Serif', Georgia, serif; text-align: right; }
    .footer { border-top: 1px solid #232521; color: #555851; display: flex; justify-content: space-between; padding: 1.5rem 0 0; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state():
    if "job" not in st.session_state:
        st.session_state.job = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def reset_for_new_upload():
    st.session_state.job = None
    st.session_state.uploader_key += 1
    st.rerun()


def run_job(job):
    temporary_directory = Path(tempfile.mkdtemp(prefix="traffic_vision_"))
    input_path = temporary_directory / f"input{job['suffix']}"
    output_path = temporary_directory / "annotated_result.mp4"
    input_path.write_bytes(job["input_bytes"])

    def update_progress(processed, total):
        with job["lock"]:
            job["processed"] = processed
            job["total"] = total

    try:
        process_video(input_path, output_path, update_progress, job["cancel_event"].is_set)
        with job["lock"]:
            job["state"] = "complete"
            job["result_bytes"] = output_path.read_bytes()
    except VideoProcessingCancelled:
        with job["lock"]:
            job["state"] = "cancelled"
    except Exception as error:
        with job["lock"]:
            job["state"] = "error"
            job["error"] = str(error)
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


def start_job(uploaded_file, width, height):
    warning = ""
    if width > 1024 or height > 1024:
        warning = f"This video is {width}x{height}. It may take more time to process; try a smaller video with lower dimensions for faster results."
    job = {
        "state": "processing",
        "input_bytes": uploaded_file.getvalue(),
        "suffix": Path(uploaded_file.name).suffix.lower(),
        "name": Path(uploaded_file.name).stem,
        "processed": 0,
        "total": 0,
        "warning": warning,
        "cancel_event": threading.Event(),
        "lock": threading.Lock(),
        "result_bytes": None,
        "error": "",
    }
    st.session_state.job = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()


@st.fragment(run_every=0.7)
def render_processing(job):
    with job["lock"]:
        state = job["state"]
        processed = job["processed"]
        total = job["total"]
    if state == "processing":
        percent = round(processed / total * 100) if total else 0
        status = f"Processing frame {processed} of {total}..." if total else "Starting analysis..."
        status_col, percent_col = st.columns([4, 1])
        with status_col:
            st.markdown(f'<div class="status">{status}</div>', unsafe_allow_html=True)
        with percent_col:
            st.markdown(f'<div class="percent">{percent}%</div>', unsafe_allow_html=True)
        st.progress(percent)
        st.markdown('<div class="stop-button">', unsafe_allow_html=True)
        if st.button("Stop analysis", key="stop-analysis"):
            job["cancel_event"].set()
        st.markdown('</div>', unsafe_allow_html=True)
    elif state == "complete":
        st.rerun()
    elif state == "cancelled":
        st.rerun()


initialize_state()
job = st.session_state.job

st.markdown('<div class="topbar"><span>Traffic Vision</span><span class="availability" title="System ready"></span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><div class="eyebrow">&gt; Traffic intelligence / 01</div><h1>Road footage, <em>made legible.</em></h1><p>Detect <strong>helmets</strong> and extract <strong>number plates</strong> with an annotated result you can preview right here.</p></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([0.82, 1.18], gap="small")

with left:
    with st.container(border=True):
        st.subheader("Analyze a video")
        uploaded_file = st.file_uploader(
            "Drop your video here or choose a file from your device",
            type=sorted(ALLOWED_EXTENSIONS),
            help="Maximum 100 MB. Larger dimensions may take longer to process.",
            key=f"video-uploader-{st.session_state.uploader_key}",
            disabled=job is not None and job["state"] == "processing",
        )

        if uploaded_file is not None and (job is None or job["state"] not in {"processing", "complete"}):
            if uploaded_file.size > MAX_UPLOAD_BYTES:
                st.error("The video must be smaller than 100 MB.")
            else:
                probe_path = Path(tempfile.mktemp(suffix=Path(uploaded_file.name).suffix))
                probe_path.write_bytes(uploaded_file.getvalue())
                cap = cv2.VideoCapture(str(probe_path))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                readable = cap.isOpened() and width > 0 and height > 0
                cap.release()
                probe_path.unlink(missing_ok=True)
                if not readable:
                    st.error("The uploaded file is not a readable video.")
                else:
                    if width > 1024 or height > 1024:
                        st.markdown(f'<div class="warning">This video is {width}x{height}. It may take more time to process; try a smaller video with lower dimensions for faster results.</div>', unsafe_allow_html=True)
                    if st.button("Start analysis", type="primary"):
                        start_job(uploaded_file, width, height)
                        st.rerun()

        if job is not None and job["state"] == "processing":
            render_processing(job)
        elif job is not None and job["state"] == "cancelled":
            st.info("Analysis stopped. Choose another video.")
            if st.button("Use another video", key="new-video-cancelled"):
                reset_for_new_upload()
        elif job is not None and job["state"] == "error":
            st.error(f"Video processing failed: {job['error']}")
            if st.button("Use another video", key="new-video-error"):
                reset_for_new_upload()
        elif job is not None and job["state"] == "complete":
            if st.button("Analyze another video", key="new-video-complete"):
                reset_for_new_upload()

with right:
    with st.container(border=True):
        if job is not None and job["state"] == "complete":
            st.subheader("Annotated result")
            st.video(job["result_bytes"], format="video/mp4")
            st.download_button(
                "Download video",
                data=job["result_bytes"],
                file_name=f"{job['name']}_detections.mp4",
                mime="video/mp4",
            )
        else:
            st.subheader("Annotated result")
            st.markdown('<p class="status">Your processed video will appear here.</p>', unsafe_allow_html=True)

st.markdown('<div class="footer"><span>Traffic Vision · Computer vision for safer roads</span><span>01 / Analysis workspace</span></div>', unsafe_allow_html=True)
