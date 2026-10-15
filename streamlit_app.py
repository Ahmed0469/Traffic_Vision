import shutil
import tempfile
import threading
from pathlib import Path

import cv2
import streamlit as st

from src.object_detection import (
    VideoProcessingCancelled,
    load_video_frames,
    process_selected_frames,
    read_video_frame,
)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_SELECTED_FRAMES = 10
ALLOWED_EXTENSIONS = {".mp4", ".mov"}

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
    [data-testid="stDialog"] > div { background: #0d0f0d; border: 1px solid #45483f; border-radius: 0; }
    [data-testid="stDialog"] h2 { color: var(--cream) !important; }
    [data-testid="stDialog"] [data-testid="stMarkdownContainer"] p { color: #d4d0c7; }
    [data-testid="stDialog"] .stButton > button { background: var(--cream) !important; border-color: var(--cream) !important; color: #151612 !important; -webkit-text-fill-color: #151612 !important; font-size: 1rem !important; font-weight: 800 !important; }
    [data-testid="stDialog"] .stButton > button p,
    [data-testid="stDialog"] .stButton > button span { color: #151612 !important; -webkit-text-fill-color: #151612 !important; font-size: 1rem !important; font-weight: 800 !important; opacity: 1 !important; }
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
    if "video" not in st.session_state:
        st.session_state.video = None
    if "selected_frames" not in st.session_state:
        st.session_state.selected_frames = []
    if "active_frame" not in st.session_state:
        st.session_state.active_frame = 0
    if "processing_notice_upload_key" not in st.session_state:
        st.session_state.processing_notice_upload_key = None


def reset_for_new_upload():
    video = st.session_state.video
    if video is not None:
        shutil.rmtree(video["temporary_directory"], ignore_errors=True)
    st.session_state.video = None
    st.session_state.job = None
    st.session_state.selected_frames = []
    st.session_state.processing_notice_upload_key = None
    st.session_state.uploader_key += 1
    st.rerun()


def load_uploaded_video(uploaded_file):
    upload_key = getattr(uploaded_file, "file_id", f"{uploaded_file.name}:{uploaded_file.size}")
    current_video = st.session_state.video
    if (
        current_video is not None
        and current_video["upload_key"] == upload_key
        and "frames" in current_video
    ):
        return current_video, ""
    if current_video is not None:
        shutil.rmtree(current_video["temporary_directory"], ignore_errors=True)
    st.session_state.video = None
    st.session_state.job = None
    st.session_state.selected_frames = []
    st.session_state.active_frame = 0

    temporary_directory = Path(tempfile.mkdtemp(prefix="traffic_vision_"))
    input_path = temporary_directory / f"input{Path(uploaded_file.name).suffix.lower()}"
    input_bytes = uploaded_file.getvalue()
    input_path.write_bytes(input_bytes)
    cap = cv2.VideoCapture(str(input_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 0
    readable = cap.isOpened() and frame_count > 0 and width > 0 and height > 0
    cap.release()
    if not readable:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        st.session_state.video = None
        return None, "The uploaded file is not a readable video."

    video = {
        "upload_key": upload_key,
        "input_path": input_path,
        "temporary_directory": temporary_directory,
        "name": Path(uploaded_file.name).stem,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }
    try:
        video["frames"] = load_video_frames(input_path)
    except (OSError, ValueError) as error:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        return None, str(error)
    video["frame_count"] = len(video["frames"])
    st.session_state.video = video
    return video, ""


def acknowledge_processing_notice(upload_key):
    st.session_state.processing_notice_upload_key = upload_key
    st.rerun(scope="app")


@st.dialog("Optimized Processing Mode", dismissible=False)
def show_processing_notice(upload_key):
    st.markdown(
        "To keep analysis responsive and support consistent access across our shared cloud environment, "
        "Traffic Vision prepares video frames for review and runs detection only on the frames you select, up to ten. "
        "This workload-aware approach helps maintain predictable performance for everyone."
    )
    st.button(
        "SELECT UP TO 10 FRAMES",
        type="primary",
        use_container_width=True,
        on_click=acknowledge_processing_notice,
        args=(upload_key,),
    )


def run_job(job):
    def update_progress(processed, total):
        with job["lock"]:
            job["processed"] = processed
            job["total"] = total

    try:
        results = process_selected_frames(
            job["input_path"],
            job["frame_indices"],
            update_progress,
            job["cancel_event"].is_set,
        )
        encoded_results = []
        for frame_index, original, annotated in results:
            _, original_bytes = cv2.imencode(".jpg", original)
            _, annotated_bytes = cv2.imencode(".jpg", annotated)
            encoded_results.append(
                {
                    "frame_number": frame_index + 1,
                    "original": original_bytes.tobytes(),
                    "annotated": annotated_bytes.tobytes(),
                }
            )
        with job["lock"]:
            job["state"] = "complete"
            job["result_frames"] = encoded_results
    except VideoProcessingCancelled:
        with job["lock"]:
            job["state"] = "cancelled"
    except Exception as error:
        with job["lock"]:
            job["state"] = "error"
            job["error"] = str(error)


def start_job(video):
    job = {
        "state": "processing",
        "input_path": video["input_path"],
        "frame_indices": [item["index"] for item in st.session_state.selected_frames],
        "processed": 0,
        "total": 0,
        "cancel_event": threading.Event(),
        "lock": threading.Lock(),
        "result_frames": None,
        "error": "",
    }
    st.session_state.job = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()


def set_active_frame(frame_index):
    st.session_state.active_frame = frame_index


def add_active_frame(video):
    frame_index = st.session_state.active_frame
    selected_frames = st.session_state.selected_frames
    if any(item["index"] == frame_index for item in selected_frames):
        return
    if len(selected_frames) >= MAX_SELECTED_FRAMES:
        return

    frame = read_video_frame(video["input_path"], frame_index)
    success, encoded_frame = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError(f"Could not save frame {frame_index + 1}.")
    selected_frames.append({"index": frame_index, "image": encoded_frame.tobytes()})
    st.session_state.job = None


@st.fragment
def render_frame_navigator(video):
    frame_count = len(video["frames"])
    st.markdown("**Choose frames for detection**")
    current_frame = st.slider(
        "Frame timeline",
        min_value=0,
        max_value=frame_count - 1,
        key="active_frame",
        format="Frame %d",
    )

    control_columns = st.columns(4, gap="small")
    with control_columns[0]:
        st.button("⏮ First Frame", on_click=set_active_frame, args=(0,), use_container_width=True)
    with control_columns[1]:
        st.button(
            "◀ Prev Frame",
            on_click=set_active_frame,
            args=(max(0, current_frame - 1),),
            disabled=current_frame == 0,
            use_container_width=True,
        )
    with control_columns[2]:
        st.button(
            "▶ Next Frame",
            on_click=set_active_frame,
            args=(min(frame_count - 1, current_frame + 1),),
            disabled=current_frame == frame_count - 1,
            use_container_width=True,
        )
    with control_columns[3]:
        st.button(
            "⏭ Last Frame",
            on_click=set_active_frame,
            args=(frame_count - 1,),
            use_container_width=True,
        )

    st.image(
        video["frames"][current_frame],
        channels="RGB",
        caption=f"Frame {current_frame + 1} of {frame_count}",
        use_container_width=True,
    )
    selected_frames = st.session_state.selected_frames
    frame_selected = any(item["index"] == current_frame for item in selected_frames)
    try:
        if st.button(
            "📸 Select Frame for Detection",
            type="primary",
            disabled=frame_selected or len(selected_frames) >= MAX_SELECTED_FRAMES,
            key=f"select-frame-{video['upload_key']}-{current_frame}",
            use_container_width=True,
        ):
            add_active_frame(video)
            st.rerun()
    except (OSError, ValueError) as error:
        st.error(str(error))

    st.markdown(f"**Grabbed frames: {len(selected_frames)} / {MAX_SELECTED_FRAMES}**")
    if selected_frames:
        gallery_columns = st.columns(3, gap="small")
        for position, item in enumerate(selected_frames):
            with gallery_columns[position % len(gallery_columns)]:
                st.image(
                    video["frames"][item["index"]],
                    channels="RGB",
                    caption=f"Frame {item['index'] + 1}",
                    use_container_width=True,
                )
                if st.button(
                    f"Remove frame {item['index'] + 1}",
                    key=f"remove-selected-{video['upload_key']}-{item['index']}",
                    use_container_width=True,
                ):
                    selected_frames.pop(position)
                    st.session_state.job = None
                    st.rerun()

    if not selected_frames:
        st.caption("Select at least one frame. You can choose up to 10.")
    if st.button(
        "Analyze selected frames",
        type="primary",
        disabled=not selected_frames,
        key=f"analyze-frames-{video['upload_key']}",
        use_container_width=True,
    ):
        start_job(video)
        st.rerun()


@st.fragment(run_every=0.7)
def render_processing(job):
    with job["lock"]:
        state = job["state"]
        processed = job["processed"]
        total = job["total"]
    if state == "processing":
        percent = round(processed / total * 100) if total else 0
        status = f"Analyzing selected frame {processed} of {total}..." if total else "Starting analysis..."
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

left, right = st.columns([1.1, 0.9], gap="small")

with left:
    with st.container(border=True):
        st.subheader("Analyze a video")
        uploaded_file = st.file_uploader(
            "Drop your video here or choose a file from your device",
            type=sorted(ALLOWED_EXTENSIONS),
            help="Maximum 100 MB. Larger dimensions may take longer to process.",
            key=f"video-uploader-{st.session_state.uploader_key}",
            disabled=job is not None and job["state"] in {"processing", "complete"},
        )

        video = st.session_state.video
        video_error = ""
        if uploaded_file is not None:
            if uploaded_file.size > MAX_UPLOAD_BYTES:
                st.error("The video must be smaller than 100 MB.")
                upload_key = getattr(uploaded_file, "file_id", f"{uploaded_file.name}:{uploaded_file.size}")
                if video is not None and video["upload_key"] != upload_key:
                    shutil.rmtree(video["temporary_directory"], ignore_errors=True)
                    st.session_state.video = None
                    st.session_state.job = None
                    st.session_state.selected_frames = []
                    st.session_state.active_frame = 0
                    video = None
            else:
                upload_key = getattr(
                    uploaded_file,
                    "file_id",
                    f"{uploaded_file.name}:{uploaded_file.size}",
                )
                if video is not None and video["upload_key"] != upload_key:
                    shutil.rmtree(video["temporary_directory"], ignore_errors=True)
                    st.session_state.video = None
                    st.session_state.job = None
                    st.session_state.selected_frames = []
                    st.session_state.active_frame = 0
                    video = None
                if video is None:
                    with st.spinner("Decoding frames for instant navigation..."):
                        video, video_error = load_uploaded_video(uploaded_file)
        elif video is not None:
            shutil.rmtree(video["temporary_directory"], ignore_errors=True)
            st.session_state.video = None
            st.session_state.job = None
            st.session_state.selected_frames = []
            st.session_state.active_frame = 0
            st.session_state.processing_notice_upload_key = None
            video = None
        job = st.session_state.job

        if video_error:
            st.error(video_error)

        if video is not None:
            st.markdown(
                f'<div class="status">{video["width"]} × {video["height"]} · {video["frame_count"]} frames</div>',
                unsafe_allow_html=True,
            )
            if video["frame_count"] > 0 and (job is None or job["state"] != "processing"):
                render_frame_navigator(video)
                if st.session_state.processing_notice_upload_key != video["upload_key"]:
                    show_processing_notice(video["upload_key"])

        if job is not None and job["state"] == "processing":
            render_processing(job)
        elif job is not None and job["state"] == "cancelled":
            st.info("Analysis stopped. You can run the selected frames again or choose another video.")
            if st.button("Run selected frames again", key="retry-cancelled"):
                start_job(video)
                st.rerun()
            if st.button("Use another video", key="new-video-cancelled"):
                reset_for_new_upload()
        elif job is not None and job["state"] == "error":
            st.error(f"Frame processing failed: {job['error']}")
            if st.button("Run selected frames again", key="retry-error"):
                start_job(video)
                st.rerun()
            if st.button("Use another video", key="new-video-error"):
                reset_for_new_upload()
        elif job is not None and job["state"] == "complete":
            if st.button("Analyze another video", key="new-video-complete"):
                reset_for_new_upload()

with right:
    with st.container(border=True):
        if job is not None and job["state"] == "complete":
            st.subheader("Before and after")
            for result in job["result_frames"]:
                st.markdown(f"**Frame {result['frame_number']}**")
                original_column, detected_column = st.columns(2)
                with original_column:
                    st.caption("Before")
                    st.image(result["original"], use_container_width=True)
                with detected_column:
                    st.caption("After")
                    st.image(result["annotated"], use_container_width=True)
        else:
            st.subheader("Frame comparisons")
            st.markdown('<p class="status">Your selected frames will appear here after analysis.</p>', unsafe_allow_html=True)

st.markdown('<div class="footer"><span>Traffic Vision · Computer vision for safer roads</span><span>01 / Analysis workspace</span></div>', unsafe_allow_html=True)
