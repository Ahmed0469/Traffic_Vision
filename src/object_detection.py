import ssl
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np
import paddlex
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
helmet_classification_model = YOLO(str(PROJECT_ROOT / "Helmet_Classification_Model.pt"))
numberplate_detection_model = YOLO(str(PROJECT_ROOT / "NumberPlate_Detection_Model.pt"))
ssl._create_default_https_context = ssl._create_unverified_context
paddle_ocr_model = paddlex.create_model(model_name="PP-OCRv4_mobile_rec")


class VideoProcessingCancelled(Exception):
    pass


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = source_fps if np.isfinite(source_fps) and source_fps > 0 else 30.0
    expected_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_path.with_name(f"{output_path.stem}_opencv.mp4")
    writer = cv2.VideoWriter(
        str(intermediate_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("The output video writer could not be opened.")

    processed_frames = 0
    cleanup_intermediate = False
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            if cancel_callback is not None and cancel_callback():
                raise VideoProcessingCancelled()

            helmets_result = helmet_classification_model.predict(frame, classes=[2, 3], verbose=False)
            detections = helmets_result[0].plot()
            numberplate_result = numberplate_detection_model.predict(frame, verbose=False)

            for box in numberplate_result[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().round().astype(int)
                x1 = max(0, min(x1, frame_width - 1))
                y1 = max(0, min(y1, frame_height - 1))
                x2 = max(x1 + 1, min(x2, frame_width - 1))
                y2 = max(y1 + 1, min(y2, frame_height - 1))
                number_plate = frame[max(0, y1):y2, max(0, x1):x2]
                if number_plate.size == 0:
                    continue

                cv2.rectangle(detections, (x1, y1), (x2, y2), (0, 220, 255), 3)
                plate_text = " ".join(
                    str(result.get("rec_text", "")).strip()
                    for result in paddle_ocr_model.predict(number_plate, batch_size=1)
                ).strip()
                if plate_text:
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = max(0.5, min(1.2, min(frame_width, frame_height) / 500))
                    thickness = max(1, round(font_scale * 2))
                    (text_width, text_height), baseline = cv2.getTextSize(
                        plate_text, font, font_scale, thickness
                    )
                    while text_width > frame_width - 20 and font_scale > 0.4:
                        font_scale *= 0.9
                        thickness = max(1, round(font_scale * 2))
                        (text_width, text_height), baseline = cv2.getTextSize(
                            plate_text, font, font_scale, thickness
                        )

                    padding_x, padding_y = 10, 8
                    label_width = min(frame_width, text_width + padding_x * 2)
                    label_height = text_height + baseline + padding_y * 2
                    label_x = max(0, min(x1, frame_width - label_width))
                    label_y = y1 - label_height if y1 >= label_height else y2
                    label_y = max(0, min(label_y, frame_height - label_height))
                    cv2.rectangle(
                        detections,
                        (label_x, label_y),
                        (label_x + label_width, label_y + label_height),
                        (18, 28, 38),
                        cv2.FILLED,
                    )
                    cv2.putText(
                        detections,
                        plate_text,
                        (label_x + padding_x, label_y + padding_y + text_height),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

            writer.write(detections)
            processed_frames += 1
            if progress_callback is not None:
                progress_callback(processed_frames, expected_frames)
            if cancel_callback is not None and cancel_callback():
                raise VideoProcessingCancelled()
    except VideoProcessingCancelled:
        cleanup_intermediate = True
        raise
    except Exception:
        cleanup_intermediate = True
        raise
    finally:
        cap.release()
        writer.release()
        if cleanup_intermediate:
            intermediate_path.unlink(missing_ok=True)

    if processed_frames == 0:
        raise ValueError("No video frames were found in the uploaded file.")

    try:
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-i",
                str(intermediate_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Browser-compatible video encoding failed: {error.stderr[-500:]}") from error
    finally:
        intermediate_path.unlink(missing_ok=True)

    return processed_frames, expected_frames


if __name__ == "__main__":
    input_path = PROJECT_ROOT / "Data" / "traffic5.mp4"
    output_path = PROJECT_ROOT / "Data" / "traffic5_detections.mp4"
    processed, expected = process_video(input_path, output_path)
    print(f"Video processing completed: {processed}/{expected} frames written to {output_path}")