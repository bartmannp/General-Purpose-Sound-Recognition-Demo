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

Then specify the path when running the app using either:

* a YAML config file (recommended), or
* a CLI override (e.g. `MODEL_PATH='<model_location>'`).

More models can be found [here](https://zenodo.org/record/3576599) and [here](https://zenodo.org/record/3987831).


# RUN

Assuming the model has been downloaded and `<repo_root>` is the current directory,
the app looks for the default config in this order:

1. `options.default.yaml`
2. `assets/options.default.yaml`

```
uv run -m sed_demo
```

Or, if using conda/pip:

```
python -m sed_demo
```

To use your own YAML file, pass its path with `CONFIG_PATH`:

```
uv run -m sed_demo CONFIG_PATH='assets/options.example.yaml'
```

CLI values still override YAML values when needed:

```
uv run -m sed_demo CONFIG_PATH='assets/options.example.yaml' TOP_K=12
```

If a key is missing from your YAML file, the in-code default is used.
The terminal prints the final merged configuration on startup.

## Configuration Reference

Configuration merge priority (lowest to highest):

1. Code defaults in ConfDef
2. Main config file set by CONFIG_PATH
3. CLI overrides passed as KEY=VALUE
4. Selected specific-app config from SPECIFIC_APP_CONFS or SPECIFIC_APP_CONFS_N

Main config options:

Audio labels and model:

* ALL_LABELS_PATH: CSV path with all AudioSet labels used by the model.
* SUBSET_LABELS_PATH: Optional CSV path for label filtering. Use null to disable.
* LABEL_COLLECTIONS_PATH: Optional CSV path defining virtual grouped labels.
* LABEL_GAINS: Optional mapping label_name -> gain. 1.0 is neutral.
* COLLECTION_GAINS: Optional mapping collection_name -> gain. 1.0 is neutral.
* MODEL_PATH: Path to the pretrained model checkpoint.

Audio and frontend processing:

* SAMPLERATE: Audio sample rate used for capture and preprocessing.
* AUDIO_CHUNK_LENGTH: Number of audio samples read per stream chunk.
* RINGBUFFER_LENGTH: Audio ring buffer length in samples.
* MODEL_WINSIZE: STFT window size used by the model frontend.
* STFT_HOPSIZE: STFT hop size.
* STFT_WINDOW: STFT window type.
* N_MELS: Number of mel bins.
* MEL_FMIN: Lowest mel frequency.
* MEL_FMAX: Highest mel frequency.
* INFERENCE_INTERVAL: Seconds between model inferences.

Audio device behavior:

* AUDIO_DEVICE_INDEX: Optional fixed PortAudio device index.
* SELECT_AUDIO_DEVICE: If true, always opens interactive device selection on startup, even if AUDIO_DEVICE_INDEX is set.

Runtime mode and headless output:

* HEADLESS: If true, run without GUI and print predictions to console.
* HEADLESS_PRINT_INTERVAL: Maximum interval between printed lines in headless mode.
* HEADLESS_MIN_CONFIDENCE: Hide predictions below this confidence.
* HEADLESS_REDUCED_LOG_OUTPUT: If true, omit no-detection lines and emit a
	heartbeat line once per minute if no other line was printed.
* STOP_AFTER_MINUTES: Optional runtime limit in minutes. When set to a
	positive value, the app shuts down cleanly after that many minutes.
* HEADLESS_LOG_MAX_MINUTES: Optional max duration for one logfile in minutes.
	When set to a positive value, a new logfile is started after that duration.
Use null for unlimited logfile length.
* HEADLESS_LOG_PATH: Optional log file path for headless mode.
	Supported tokens: $year, $month, $day, $hour, $minute, $seconds, $timestamp.
	Example value: logs/sound-recognition_$timestamp.log

GUI and config selection:

* TOP_K: Number of top predictions to display.
* TITLE_FONTSIZE: GUI title font size.
* TABLE_FONTSIZE: GUI prediction table font size.
* CONFIG_PATH: Path to main YAML config file.
* SPECIFIC_APP_CONFS: Optional path to one specific-app override YAML.

Multiple specific-app configs:

You can define multiple specific-app configs in the main YAML:

* SPECIFIC_APP_CONFS_0: path/to/profile_a.yaml
* SPECIFIC_APP_CONFS_1: path/to/profile_b.yaml
* SPECIFIC_APP_CONFS_2: path/to/profile_c.yaml

If more than one SPECIFIC_APP_CONFS_N entry is present in the main YAML,
the app asks at startup which one to load. When no interactive input is
available, it automatically falls back to SPECIFIC_APP_CONFS_0.

### Example: main config with all common options

Example file: assets/options.default.yaml

ALL_LABELS_PATH: sed_demo/assets/audioset_labels.csv
SUBSET_LABELS_PATH: null
LABEL_COLLECTIONS_PATH: assets/requested_collections_medium.csv
LABEL_GAINS: {}
COLLECTION_GAINS: {}
MODEL_PATH: models/Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth
SAMPLERATE: 32000
AUDIO_CHUNK_LENGTH: 1024
RINGBUFFER_LENGTH: 64000
MODEL_WINSIZE: 1024
STFT_HOPSIZE: 512
STFT_WINDOW: hann
N_MELS: 64
MEL_FMIN: 50
MEL_FMAX: 14000
INFERENCE_INTERVAL: 0.25
AUDIO_DEVICE_INDEX: null
SELECT_AUDIO_DEVICE: true
HEADLESS: true
HEADLESS_PRINT_INTERVAL: 1.0
HEADLESS_MIN_CONFIDENCE: 0.15
HEADLESS_REDUCED_LOG_OUTPUT: false
STOP_AFTER_MINUTES: 120
HEADLESS_LOG_MAX_MINUTES: null
HEADLESS_LOG_PATH: logs/session_$timestamp.log
TOP_K: 6
TITLE_FONTSIZE: 28
TABLE_FONTSIZE: 22
CONFIG_PATH: assets/options.default.yaml
SPECIFIC_APP_CONFS_0: assets/profiles/home.yaml
SPECIFIC_APP_CONFS_1: assets/profiles/office.yaml

### Example: specific-app profile with gain tuning

Example file: assets/profiles/home.yaml

HEADLESS: true
LABEL_GAINS:
	Walk, footsteps: 1.2
	Toilet flush: 0.8
COLLECTION_GAINS:
	door bell: 1.3
	boiling water: 0.9

### Example run commands

Use the default main config:

uv run -m sed_demo

Use a different main config:

uv run -m sed_demo CONFIG_PATH=assets/options.example.yaml

Force one specific profile from CLI (highest priority):

uv run -m sed_demo CONFIG_PATH=assets/options.default.yaml SPECIFIC_APP_CONFS=assets/profiles/home.yaml

Override a single key from CLI:

uv run -m sed_demo CONFIG_PATH=assets/options.default.yaml TOP_K=12

### Aggregated label collections

Besides filtering to a subset of existing labels, the demo can now aggregate
several AudioSet labels into a virtual label by summing their probabilities
after model inference. This does **not** change model loading, retraining, or
probability normalization; it only changes how predictions are grouped for
display.

To enable this mode, pass `LABEL_COLLECTIONS_PATH` pointing to a CSV with four
columns:

```
collection_name,index,mid,display_name
```

Each repeated `collection_name` defines one aggregate label. An example file is
provided in [assets/example_label_collections.csv](assets/example_label_collections.csv).

Example:

```
uv run -m sed_demo MODEL_PATH='<model_location>' HEADLESS=True LABEL_COLLECTIONS_PATH='assets/example_label_collections.csv' TOP_K=3
```

`SUBSET_LABELS_PATH` can still be used at the same time. If both are set, the
collection members are first filtered by the subset and then summed inside each
collection.

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
* `HEADLESS_REDUCED_LOG_OUTPUT=True` to skip no-detection lines and keep one
	heartbeat line per minute when idle.
* `HEADLESS_COLLECTION_DETAIL_OUTPUT=True` to append per-member numeric values
	for collection outputs in parentheses.
* `STOP_AFTER_MINUTES=120` to stop the app automatically after two hours.
* `HEADLESS_LOG_MAX_MINUTES=60` to rotate logfile output every hour.
* `HEADLESS_LOG_PATH='sound-recognition.log'` to append timestamped output to a log file.

Each headless output line is timestamped, both in the terminal and in the
optional log file.

When `LABEL_COLLECTIONS_PATH` is active and
`HEADLESS_COLLECTION_DETAIL_OUTPUT=True`, collection lines can look like:

```
[2026-07-13 09:41:02] Speech aggregate: 1.30 (0.10, 0.30, 0.00, 0.90, 0.00, 0.00)
```

The number before parentheses is the collection score. Values in parentheses
are the accumulated contributions of collection members, ordered exactly as
listed in the collection CSV file.


---

# Related links:

* https://research.google.com/audioset/dataset
* https://github.com/qiuqiangkong/audioset_tagging_cnn
* https://github.com/qiuqiangkong/panns_inference
* https://github.com/yinkalario/Sound-Event-Detection-AudioSet




