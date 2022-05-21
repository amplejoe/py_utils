# py_utils
Util functions/classes for python projects.

# Installation

1. Clone to a local directory, e.g. `~/projects`
   ```
   mkdir -p ~/projects/modules && cd ~/projects/modules
   git clone git@github.com:amplejoe/py_utils.git
   ```
2. Include module directory in `PYTHONPATH`
    Windows
    Linux - edit ~/.bashrc and include:
    ```
    PYTHONPATH="${PYTHONPATH}:~/projects/modules"
    ```
3. Install requirements
    ```
    python -m pip install --user -r ~/projects/modules/py_utils/requirements.txt
    ```
# Usage


Import desired modules within your projects.

#### Utils (utils.py)

Simple utils class containing everyday useful Python functions. Example:
```
from py_utils import utils

utils.get_file_paths("/path/containing/images", ".jpg", ".png")
```

#### OpenCv Utils ([opencv_utils.py](opencv_utils.py))

[OpenCV](https://opencv.org/)-related helper functions. Example:

```
from py_utils import opencv_utils

img = opencv_utils.get_image("/path/to/image.jpg")
img_txt = opencv_utils.overlay_text(img, "Hello\nWorld!", scale=1.5)
opencv_utils.show_image(img_txt, title="Hello")
```

#### Random Color Generator ((random_colors)[random_colory.py])
Tool for generating random colors. Example:
```
from py_utils import random_colors

colors = get_colors('rgb', 10)
```

#### ColorLabeler ([ColorLabeler.py](ColorLabeler.py))

Class for handling mulit-colored masks in projects dealing with image segmentation problems.

#### Video Cutter ([VideoCutter.py](VideoCutter.py))

Helper class for cutting videos using [FFmpeg](https://ffmpeg.org/). Requires [FFmpeg](https://ffmpeg.org/) installation.

#### CNN Utils ([cnn_utils.py](cnn_utils.py))

CNN Helper functions. Requires a working [Detectron2](https://github.com/facebookresearch/detectron2) installation.

#### Script Creation Tool ([create_script.py](create_script.py))

Capable of generating boilerplate starting code for Python main executable scripts (`main`), classes (`classes`) as well as bash scripts (`bash`). Just execute the script via `python create_script.py` or list its usage information: `python create_script.py -h`.





