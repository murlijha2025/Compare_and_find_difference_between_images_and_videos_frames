Compare Differences Between Videos or Images
Finds what changed between two videos or two images and highlights it. Changed regions get a green box, and for images it also saves a version with the changed pixels painted red. It also gives a similarity score (SSIM) so you can see how different they are at a glance.
Setup
You need Python 3.9+.
```bash
git clone https://github.com/murlijha2025/Compare_and_find_difference_between_images_and_videos_frames.git
cd Compare_and_find_difference_between_images_and_videos_frames
pip install -r requirements.txt
```
Usage
```bash
# two videos (preview windows open, press q to stop)
python compare_differences.py video0.avi video1.avi

# two images, results in a custom folder
python compare_differences.py before.png after.png -o results

# no preview windows (for servers or batch runs), ignore tiny changes
python compare_differences.py a.mp4 b.mp4 --no-display --min-area 100
```
Option	What it does
`first`, `second`	Two videos or two images (not one of each)
`-o`, `--output`	Folder for results (default: `output`)
`--min-area`	Ignore changed regions smaller than this many pixels (default: 40)
`--no-display`	Don't open preview windows (videos only)
Output
Saved in the output folder:
Images: `before.png` and `after.png` (green boxes around changes), and `diff.png` (changed pixels in red).
Videos: `before.avi` and `after.avi` with the boxes drawn on every frame.
The similarity score prints in the terminal. 1.0 means identical, lower means more different.
How it works
Converts both frames to grayscale.
Computes the structural similarity (SSIM) map between them.
Thresholds the map to find regions that changed, and draws a box around each one bigger than `--min-area`.
Separately, takes the pixel difference between the two images and marks the changed pixels in red.
Limitations
Both inputs need to be the same kind of file. If the sizes don't match, the second one is resized to the first.
Videos are compared frame by frame, so they need to be aligned. If one starts later, everything will show as different. Comparison stops when the shorter video ends.
Camera shake, lighting changes and compression noise show up as differences. Raising `--min-area` helps.
Dependencies
`opencv-python`, `scikit-image`, `numpy`
