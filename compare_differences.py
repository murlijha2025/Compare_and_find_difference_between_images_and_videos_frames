"""Find and highlight the differences between two videos or two images.

Usage:
    python compare_differences.py video0.avi video1.avi
    python compare_differences.py before.png after.png -o results
    python compare_differences.py a.mp4 b.mp4 --no-display --min-area 100

Press 'q' in the preview window to stop early when comparing videos.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BOX_COLOR = (36, 255, 12)  # green (BGR)
CHANGE_COLOR = (0, 0, 255)  # red (BGR)


def find_differences(image_a, image_b, min_area=40):
    """Compare two same-sized BGR images.

    Returns (annotated_a, annotated_b, pixel_overlay, ssim_score):
      - annotated_a / annotated_b: copies with green boxes around changed regions
      - pixel_overlay: copy of image_b with changed pixels painted red
      - ssim_score: 1.0 means identical, lower means more different
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    # Structural similarity -> boxes around changed regions
    score, ssim_map = structural_similarity(gray_a, gray_b, full=True)
    ssim_map = (ssim_map * 255).astype("uint8")
    changed_regions = cv2.threshold(
        ssim_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    contours, _ = cv2.findContours(
        changed_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    annotated_a, annotated_b = image_a.copy(), image_b.copy()
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated_a, (x, y), (x + w, y + h), BOX_COLOR, 2)
            cv2.rectangle(annotated_b, (x, y), (x + w, y + h), BOX_COLOR, 2)

    # Pixel-level difference -> red overlay
    # (absdiff catches brighter AND darker pixels; cv2.subtract only catches one direction)
    pixel_diff = cv2.cvtColor(cv2.absdiff(image_a, image_b), cv2.COLOR_BGR2GRAY)
    changed_pixels = cv2.threshold(
        pixel_diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]
    overlay = image_b.copy()
    overlay[changed_pixels == 255] = CHANGE_COLOR

    return annotated_a, annotated_b, overlay, score


def compare_images(path_a, path_b, out_dir, min_area):
    image_a, image_b = cv2.imread(str(path_a)), cv2.imread(str(path_b))
    if image_a is None or image_b is None:
        sys.exit("Couldn't read one of the images.")
    if image_a.shape != image_b.shape:
        image_b = cv2.resize(image_b, (image_a.shape[1], image_a.shape[0]))

    before, after, overlay, score = find_differences(image_a, image_b, min_area)
    cv2.imwrite(str(out_dir / "before.png"), before)
    cv2.imwrite(str(out_dir / "after.png"), after)
    cv2.imwrite(str(out_dir / "diff.png"), overlay)
    print(f"Similarity (SSIM): {score:.3f}")


def compare_videos(path_a, path_b, out_dir, min_area, show):
    cap_a, cap_b = cv2.VideoCapture(str(path_a)), cv2.VideoCapture(str(path_b))
    if not (cap_a.isOpened() and cap_b.isOpened()):
        sys.exit("Couldn't open one of the videos.")

    size = (
        int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
        int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
    )
    fps = cap_a.get(cv2.CAP_PROP_FPS) or 20.0
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer_a = cv2.VideoWriter(str(out_dir / "before.avi"), fourcc, fps, size)
    writer_b = cv2.VideoWriter(str(out_dir / "after.avi"), fourcc, fps, size)

    scores = []
    try:
        while True:
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not (ok_a and ok_b):  # stop when either video ends
                break
            if (frame_b.shape[1], frame_b.shape[0]) != size:
                frame_b = cv2.resize(frame_b, size)

            before, after, _, score = find_differences(frame_a, frame_b, min_area)
            writer_a.write(before)
            writer_b.write(after)
            scores.append(score)

            if show:
                cv2.imshow("before", before)
                cv2.imshow("after", after)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        for resource in (cap_a, cap_b, writer_a, writer_b):
            resource.release()
        if show:
            cv2.destroyAllWindows()

    if scores:
        print(f"Compared {len(scores)} frames. Average similarity (SSIM): {np.mean(scores):.3f}")
    else:
        print("No frames were compared.")


def parse_args():
    parser = argparse.ArgumentParser(description="Highlight differences between two videos or images.")
    parser.add_argument("first", type=Path, help="first video or image")
    parser.add_argument("second", type=Path, help="second video or image (same type as the first)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"),
                        help="folder for results (default: output)")
    parser.add_argument("--min-area", type=float, default=40,
                        help="ignore changed regions smaller than this many pixels (default: 40)")
    parser.add_argument("--no-display", action="store_true",
                        help="don't open preview windows (videos only)")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.first, args.second):
        if not path.is_file():
            sys.exit(f"File not found: {path}")

    args.output.mkdir(parents=True, exist_ok=True)
    is_image = [p.suffix.lower() in IMAGE_EXTENSIONS for p in (args.first, args.second)]
    if all(is_image):
        compare_images(args.first, args.second, args.output, args.min_area)
    elif not any(is_image):
        compare_videos(args.first, args.second, args.output, args.min_area, not args.no_display)
    else:
        sys.exit("Give two images or two videos, not one of each.")
    print(f"Results saved in {args.output}/")


if __name__ == "__main__":
    main()
