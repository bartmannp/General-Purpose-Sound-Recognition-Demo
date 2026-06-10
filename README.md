# ai4s_sed_demo

General purpose, real-time sound recognition demo:

![demo screenshot](assets/demo_screenshot.png)


The prediction is obtained by applying the audio tagging system on consecutive short audio segments. It is able to perform multiple updates per second on a moderate CPU. A sample video can be viewed at:

https://www.youtube.com/watch?v=7TEtDMzdLeY


**This is a newer version. The original version can be found at [this branch](https://github.com/yinkalario/General-Purpose-Sound-Recognition-Demo/tree/demo2019)**

# Authors

This demo has been developed and trained during our previous AudioSet work, check the following links for more info and models:

* Paper: https://arxiv.org/abs/1912.10211
* Repository: https://github.com/qiuqiangkong/audioset_tagging_cnn

If you use our work, please consider citing us:

[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).

Yin Cao, Qiuqiang Kong, Andres Fernandez, Christian Kroos, Turab Iqbal, Wenwu Wang, Mark Plumbley


---

# Installation

### Repo:

At the moment, no `pip` installation is available. Clone this repo into `<repo_root>` via:

```
https://github.com/bartmannp/General-Purpose-Sound-Recognition-Demo
```

### System dependencies (Debian/Ubuntu/Raspberry Pi OS):

Before installing Python packages, install the required system libraries:

```
sudo apt install portaudio19-dev python3-tk libsndfile1
```

* `portaudio19-dev` — required to build PyAudio
* `python3-tk` — required for the Tkinter GUI
* `libsndfile1` — required by the audio loading stack used by `librosa`

### Python dependencies:

This project supports **Python 3.11 and 3.12**. We recommend using [uv](https://docs.astral.sh/uv/) for environment and dependency management.

**Install uv** (if not already installed):

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install all dependencies and run the app via `uv` (it will create a virtual environment automatically):

```
uv run -m sed_demo MODEL_PATH='<model_location>'

On Raspberry Pi, install an ARM-compatible PyTorch wheel first if `uv` or `pip`
cannot resolve one automatically for your platform. A 64-bit Raspberry Pi OS
image is strongly recommended.
```

Alternatively, using conda/pip:

```
conda create -n panns python=3.12
conda activate panns
pip install -r requirements.txt

If PyTorch is not available from the default index on your Pi, install it from
the wheel source recommended for your Raspberry Pi OS and Python version, then
install the remaining packages.
```

A comprehensive list of working dependencies can be found in the [full_dependencies.txt](assets/full_dependencies.txt) file.

### pretrained model (CNN9, ~150MB):

Download the model into your preferred `<model_location>` via:

```
wget https://zenodo.org/record/3576599/files/Cnn9_GMP_64x64_300000_iterations_mAP%3D0.37.pth?download=1
```

Then specify the path when running the app using the `MODEL_PATH` flag (see sample command below).

More models can be found [here](https://zenodo.org/record/3576599) and [here](https://zenodo.org/record/3987831).


# RUN

Assuming the model has been downloaded and `<repo_root>` is the current directory:

```
uv run -m sed_demo MODEL_PATH='<model_location>'
```

Or, if using conda/pip:

```
python -m sed_demo MODEL_PATH='<model_location>'
```

Note that the terminal will print all available parameters and their values upon start. The syntax to alter them is the same as with `MODEL_PATH`, e.g. to change the number of classes displayed to 10, add `TOP_K=10`.

### Raspberry Pi notes

The demo can run on Raspberry Pi without code changes to the model itself, but
there are a few practical constraints:

* Use a **64-bit Raspberry Pi OS** image.
* Prefer a **Raspberry Pi 5** for smoother real-time inference. Older models
	may struggle to keep up.
* A USB microphone or other working input device must be available to PortAudio.
* The Tk GUI requires a desktop session. For console-only deployments, use the
	new headless mode.

Headless mode avoids importing Tkinter and prints the top predictions to the
terminal instead of opening a window:

```
uv run -m sed_demo MODEL_PATH='<model_location>' HEADLESS=True
```

By default, the app uses the system default input device and prints the chosen
device on startup. If you want to choose interactively at launch time, enable:

```
uv run -m sed_demo MODEL_PATH='<model_location>' HEADLESS=True SELECT_AUDIO_DEVICE=True
```

To pin a device without opening the selector, pass its PortAudio index:

```
uv run -m sed_demo MODEL_PATH='<model_location>' HEADLESS=True AUDIO_DEVICE_INDEX=2
```

You can tune headless output with:

* `INFERENCE_INTERVAL=0.25` to limit how often the model runs. Increasing this
	value lowers CPU usage on Raspberry Pi at the cost of slower UI/console
	updates.
* `HEADLESS_PRINT_INTERVAL=1.0` to control how often predictions are printed.
* `HEADLESS_MIN_CONFIDENCE=0.15` to suppress low-confidence results.
* `HEADLESS_LOG_PATH='sound-recognition.log'` to append timestamped output to a log file.

Each headless output line is timestamped, both in the terminal and in the
optional log file.


---

# Related links:

* https://research.google.com/audioset/dataset
* https://github.com/qiuqiangkong/audioset_tagging_cnn
* https://github.com/qiuqiangkong/panns_inference
* https://github.com/yinkalario/Sound-Event-Detection-AudioSet




