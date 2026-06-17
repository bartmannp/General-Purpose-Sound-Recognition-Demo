#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module is the main entry point to the app. It contains the specific class
to run the app, and a way of feeding custom parameters through the CLI.

Usage example (ensure that python can ``import sed_demo``):

python -m sed_demo TOP_K=10 TABLE_FONTSIZE=25
"""


from threading import Thread
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
#
import torch
from omegaconf import OmegaConf
#
from sed_demo import AI4S_BANNER_PATH, SURREY_LOGO_PATH, CVSSP_LOGO_PATH, \
    EPSRC_LOGO_PATH, AUDIOSET_LABELS_PATH
from sed_demo.utils import load_csv_labels, load_label_collections
from sed_demo.models import Cnn9_GMP_64x64
from sed_demo.audio_loop import AsynchAudioInputStream
from sed_demo.inference import AudioModelInference, PredictionTracker


DEFAULT_CONFIG_PATH = os.path.join("assets", "options.default.yaml")
DEFAULT_CONFIG_CANDIDATES = (
    "options.default.yaml",
    DEFAULT_CONFIG_PATH,
)


def resolve_default_config_path():
  for candidate in DEFAULT_CONFIG_CANDIDATES:
    candidate_path = os.path.abspath(candidate)
    if os.path.exists(candidate_path):
      return candidate_path
  return os.path.abspath(DEFAULT_CONFIG_PATH)


def collect_specific_app_conf_entries(conf):
  """
  Return configured specific-app config entries as
  (order, key, path) tuples.
  Supported keys:
    - SPECIFIC_APP_CONFS
    - SPECIFIC_APP_CONFS_0, SPECIFIC_APP_CONFS_1, ...
  """
  if conf is None:
    return []

  entries = []
  for key in conf.keys():
    value = OmegaConf.select(conf, key)
    if value in (None, ""):
      continue

    if key == "SPECIFIC_APP_CONFS":
      entries.append((-1, key, str(value)))
      continue

    if key.startswith("SPECIFIC_APP_CONFS_"):
      suffix = key[len("SPECIFIC_APP_CONFS_"):]
      try:
        order = int(suffix)
      except ValueError:
        continue
      entries.append((order, key, str(value)))

  entries.sort(key=lambda item: (item[0], item[1]))
  return entries


def choose_specific_app_conf_interactively(entries):
  """
  Ask the user to choose one specific-app config from provided entries.
  """
  print("Multiple specific app configs are defined in the main config file:")
  for idx, (_, key, path) in enumerate(entries):
    print(f"  [{idx}] {key}: {path}")

  while True:
    selected = input(
      f"Select config to load [0-{len(entries) - 1}] (default 0): ").strip()
    if selected == "":
      return entries[0]
    try:
      selected_idx = int(selected)
    except ValueError:
      print("Please enter a valid integer index.")
      continue
    if 0 <= selected_idx < len(entries):
      return entries[selected_idx]
    print("Selected index is out of range.")


def build_runtime(model_path, all_labels, tracked_labels=None,
      label_collections=None,
          samplerate=32000, audio_chunk_length=1024,
          ringbuffer_length=40000, model_winsize=1024,
          stft_hopsize=512, stft_window="hann", n_mels=64,
          mel_fmin=50, mel_fmax=14000, input_device_index=None):
  """
  Build the shared audio/model inference components used by both GUI and
  headless runtimes.
  """
  audiostream = AsynchAudioInputStream(
    samplerate, audio_chunk_length, ringbuffer_length, input_device_index)
  num_audioset_classes = len(all_labels)
  model = Cnn9_GMP_64x64(num_audioset_classes)
  checkpoint = torch.load(model_path,
              map_location=lambda storage, loc: storage,
              weights_only=False)
  model.load_state_dict(checkpoint["model"])
  inference = AudioModelInference(
    model, model_winsize, stft_hopsize, samplerate, stft_window,
    n_mels, mel_fmin, mel_fmax)
  tracker = PredictionTracker(
    all_labels, allow_list=tracked_labels,
    label_collections=label_collections)
  return audiostream, inference, tracker


def wait_for_next_inference(last_inference_at, inference_interval):
  if inference_interval <= 0:
    return time.monotonic()

  now = time.monotonic()
  remaining = inference_interval - (now - last_inference_at)
  if remaining > 0:
    time.sleep(remaining)
    return time.monotonic()
  return now


def create_gui_app(top_banner_path, logo_paths, model_path, all_labels,
           tracked_labels=None, label_collections=None, samplerate=32000,
           audio_chunk_length=1024, ringbuffer_length=40000,
           model_winsize=1024, stft_hopsize=512,
           stft_window="hann", n_mels=64, mel_fmin=50,
           mel_fmax=14000, inference_interval=0.25, top_k=5, title_fontsize=22,
           table_fontsize=18, input_device_index=None):
  from sed_demo.gui import DemoFrontend

  class DemoApp(DemoFrontend):
    """
    Tk frontend backed by the real-time audio inference runtime.
    """

    BG_COLOR = "#fff8fa"
    BUTTON_COLOR = "#ffcc99"
    BAR_COLOR = "#ffcc99"

    def __init__(self):
      super().__init__(top_k, top_banner_path, logo_paths,
               title_fontsize=title_fontsize,
               table_fontsize=table_fontsize)
      runtime = build_runtime(
        model_path, all_labels, tracked_labels, label_collections,
        samplerate, audio_chunk_length, ringbuffer_length,
        model_winsize, stft_hopsize, stft_window,
        n_mels, mel_fmin, mel_fmax, input_device_index)
      self.audiostream, self.inference, self.tracker = runtime
      self.top_k = top_k
      self.inference_interval = inference_interval
      self.thread = None
      self.protocol("WM_DELETE_WINDOW", self.exit_demo)

    def inference_loop(self):
      last_inference_at = 0.0
      while self.is_running():
        last_inference_at = wait_for_next_inference(
          last_inference_at, self.inference_interval)
        dl_inference = self.inference(self.audiostream.read())
        top_preds = self.tracker(dl_inference, self.top_k)
        for label, bar, (clsname, pval) in zip(
            self.sound_labels, self.confidence_bars, top_preds):
          label["text"] = clsname
          bar["value"] = pval

    def start(self):
      self.audiostream.start()
      self.thread = Thread(target=self.inference_loop)
      self.thread.daemon = True
      self.thread.start()

    def stop(self):
      self.audiostream.stop()

    def exit_demo(self):
      if self.is_running():
        print("Waiting for threads to finish...")
        self.toggle_start()
      self.after(0, self.terminate_after_thread)

    def terminate_after_thread(self, wait_loop_ms=50):
      if self.thread is not None and self.thread.is_alive():
        self.after(wait_loop_ms, self.terminate_after_thread)
      else:
        print("Exiting...")
        self.audiostream.terminate()
        self.destroy()

  return DemoApp()


class HeadlessDemoApp:
  """
  Console runtime for low-overhead or non-desktop deployments.
  """

  def __init__(self, model_path, all_labels, tracked_labels=None,
      label_collections=None,
         samplerate=32000, audio_chunk_length=1024,
         ringbuffer_length=40000, model_winsize=1024,
         stft_hopsize=512, stft_window="hann", n_mels=64,
      mel_fmin=50, mel_fmax=14000, inference_interval=0.25, top_k=5,
         print_interval=1.0, min_confidence=0.15,
         log_path=None, input_device_index=None):
    runtime = build_runtime(
      model_path, all_labels, tracked_labels, label_collections,
      samplerate, audio_chunk_length, ringbuffer_length,
      model_winsize, stft_hopsize, stft_window,
      n_mels, mel_fmin, mel_fmax, input_device_index)
    self.audiostream, self.inference, self.tracker = runtime
    self.top_k = top_k
    self.inference_interval = inference_interval
    self.print_interval = print_interval
    self.min_confidence = min_confidence
    self.log_path = log_path
    self.log_handle = None

  def _timestamp(self):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  def _emit(self, message):
    line = f"[{self._timestamp()}] {message}"
    print(line)
    if self.log_handle is not None:
      self.log_handle.write(line + "\n")
      self.log_handle.flush()

  def _describe_device(self):
    device = self.audiostream.input_device_info
    return (
      f"Using input device #{int(device['index'])}: {device['name']} "
      f"({int(device['maxInputChannels'])} ch, "
      f"default rate {device['defaultSampleRate']:.0f} Hz)"
    )

  def _format_predictions(self, predictions):
    visible = [
      f"{clsname}: {pval:.2f}"
      for clsname, pval in predictions
      if pval >= self.min_confidence
    ]
    if not visible:
      return "No predictions above threshold"
    return " | ".join(visible)

  def run(self):
    if self.log_path:
      log_dir = os.path.dirname(os.path.abspath(self.log_path))
      if log_dir:
        os.makedirs(log_dir, exist_ok=True)
      self.log_handle = open(self.log_path, "a", encoding="utf-8")
      self._emit(f"Logging to {self.log_path}")
    self._emit("Headless mode active. Press Ctrl+C to stop.")
    self._emit(self._describe_device())
    last_output = None
    last_print = 0.0
    self.audiostream.start()
    try:
      last_inference_at = 0.0
      while True:
        last_inference_at = wait_for_next_inference(
          last_inference_at, self.inference_interval)
        predictions = self.tracker(
          self.inference(self.audiostream.read()), self.top_k)
        output = self._format_predictions(predictions)
        now = time.monotonic()
        if output != last_output or (now - last_print) >= self.print_interval:
          self._emit(output)
          last_output = output
          last_print = now
    except KeyboardInterrupt:
      self._emit("Stopping...")
    finally:
      self.audiostream.terminate()
      if self.log_handle is not None:
        self.log_handle.close()


def describe_audio_device(device_info):
  return (
    f"#{int(device_info['index'])}: {device_info['name']} "
    f"({int(device_info['maxInputChannels'])} ch, "
    f"default rate {device_info['defaultSampleRate']:.0f} Hz)"
  )


def select_audio_device(select_interactively=False):
  if not select_interactively:
    default_device = AsynchAudioInputStream.get_default_input_device()
    print(f"Using default input device {describe_audio_device(default_device)}")
    return int(default_device["index"])

  devices = AsynchAudioInputStream.get_input_devices()
  if not devices:
    raise RuntimeError("No input audio devices were found.")

  print("Available input devices:")
  for device in devices:
    print(f"  {describe_audio_device(device)}")

  valid_indexes = {int(device["index"]) for device in devices}
  while True:
    selected = input("Select input device index (press Enter for default): ").strip()
    if selected == "":
      default_device = AsynchAudioInputStream.get_default_input_device()
      print(f"Using default input device {describe_audio_device(default_device)}")
      return int(default_device["index"])
    try:
      selected_index = int(selected)
    except ValueError:
      print("Please enter a valid integer device index.")
      continue
    if selected_index in valid_indexes:
      chosen_device = next(
        device for device in devices if int(device["index"]) == selected_index)
      print(f"Using selected input device {describe_audio_device(chosen_device)}")
      return selected_index
    print("Selected device index is not in the available input device list.")


# ##############################################################################
# # OMEGACONF
# ##############################################################################
@dataclass
class ConfDef:
    """
    Check ``DemoApp`` docstring for details on the parameters. Defaults should
    work reasonably well out of the box.
    """
    ALL_LABELS_PATH: str = AUDIOSET_LABELS_PATH
    SUBSET_LABELS_PATH: Optional[str] = None
    LABEL_COLLECTIONS_PATH: Optional[str] = None
    MODEL_PATH: str = os.path.join(
        "models", "Cnn9_GMP_64x64_300000_iterations_mAP=0.37.pth")
    #
    SAMPLERATE: int = 32000
    AUDIO_CHUNK_LENGTH: int = 1024
    RINGBUFFER_LENGTH: int = int(32000 * 2)
    #
    MODEL_WINSIZE: int = 1024
    STFT_HOPSIZE: int = 512
    STFT_WINDOW: str = "hann"
    N_MELS: int = 64
    MEL_FMIN: int = 50
    MEL_FMAX: int = 14000
    INFERENCE_INTERVAL: float = 0.25
    AUDIO_DEVICE_INDEX: Optional[int] = None
    SELECT_AUDIO_DEVICE: bool = False
    HEADLESS: bool = False
    HEADLESS_PRINT_INTERVAL: float = 1.0
    HEADLESS_MIN_CONFIDENCE: float = 0.15
    HEADLESS_LOG_PATH: Optional[str] = None
    # frontend
    TOP_K: int = 6
    TITLE_FONTSIZE: int = 28
    TABLE_FONTSIZE: int = 22
    CONFIG_PATH: str = DEFAULT_CONFIG_PATH
    SPECIFIC_APP_CONFS: Optional[str] = None


def load_runtime_config():
  """
  Load runtime configuration with precedence:
  1) code defaults (lowest)
  2) main YAML config
  3) CLI KEY=VALUE overrides
  4) selected SPECIFIC_APP_CONFS file (highest)

  Supported specific-app keys:
    - SPECIFIC_APP_CONFS
    - SPECIFIC_APP_CONFS_0, SPECIFIC_APP_CONFS_1, ...
  If the main config file defines more than one specific-app entry, the
  user is prompted to choose which one to load.
  """
  defaults = OmegaConf.structured(ConfDef())
  defaults = OmegaConf.create(OmegaConf.to_container(defaults, resolve=False))
  cli_conf = OmegaConf.from_cli()

  config_path = OmegaConf.select(cli_conf, "CONFIG_PATH")
  if config_path is None:
    config_path = resolve_default_config_path()

  yaml_conf = OmegaConf.create()
  if config_path:
    if not os.path.isabs(config_path):
      config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
      raise FileNotFoundError(
        f"Configuration file not found: {config_path}. "
        "Set CONFIG_PATH=<path_to_yaml> or create the default file."
      )
    yaml_conf = OmegaConf.load(config_path)

  # Explicitly layer defaults -> main YAML -> CLI so defaults are always
  # lowest priority and main YAML reliably overrides code defaults.
  merged = OmegaConf.merge(defaults, yaml_conf)
  merged = OmegaConf.merge(merged, cli_conf)

  cli_specific_entries = collect_specific_app_conf_entries(cli_conf)
  yaml_specific_entries = collect_specific_app_conf_entries(yaml_conf)

  selected_specific_entry = None
  if cli_specific_entries:
    # CLI-provided selection wins.
    selected_specific_entry = cli_specific_entries[0]
  elif len(yaml_specific_entries) > 1:
    selected_specific_entry = choose_specific_app_conf_interactively(
      yaml_specific_entries)
  elif len(yaml_specific_entries) == 1:
    selected_specific_entry = yaml_specific_entries[0]

  if selected_specific_entry is not None:
    _, selected_key, specific_conf_path = selected_specific_entry
    if not os.path.isabs(specific_conf_path):
      specific_conf_path = os.path.abspath(specific_conf_path)
    if not os.path.exists(specific_conf_path):
      raise FileNotFoundError(
        f"Specific app configuration file not found: {specific_conf_path}. "
        f"Set {selected_key}=<path_to_yaml> to a valid file."
      )
    specific_conf = OmegaConf.load(specific_conf_path)
    merged = OmegaConf.merge(merged, specific_conf)
    merged.SPECIFIC_APP_CONFS = specific_conf_path

  merged.CONFIG_PATH = config_path
  return merged


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
if __name__ == '__main__':

  if sys.version_info[:2] not in ((3, 11), (3, 12)):
    print(
      "WARNING: This project is standardized on Python 3.11/3.12. "
      f"Current interpreter: {sys.version.split()[0]}"
    )

  CONF = load_runtime_config()
  print("\n\nCONFIGURATION:")
  print(OmegaConf.to_yaml(CONF), end="\n\n\n")

  _, _, all_labels = load_csv_labels(CONF.ALL_LABELS_PATH)
  if CONF.SUBSET_LABELS_PATH is None:
    subset_labels = None
  else:
    _, _, subset_labels = load_csv_labels(CONF.SUBSET_LABELS_PATH)
  if CONF.LABEL_COLLECTIONS_PATH is None:
    label_collections = None
  else:
    label_collections = load_label_collections(CONF.LABEL_COLLECTIONS_PATH)
  logo_paths = [SURREY_LOGO_PATH, CVSSP_LOGO_PATH, EPSRC_LOGO_PATH]
  if CONF.AUDIO_DEVICE_INDEX is not None:
    audio_device_index = CONF.AUDIO_DEVICE_INDEX
    try:
      selected_device = AsynchAudioInputStream.get_input_devices()
      matched_device = next(
        device for device in selected_device
        if int(device["index"]) == int(audio_device_index))
      print(f"Using configured input device {describe_audio_device(matched_device)}")
    except StopIteration as exc:
      raise RuntimeError(
        f"Configured AUDIO_DEVICE_INDEX={audio_device_index} is not available."
      ) from exc
  else:
    audio_device_index = select_audio_device(CONF.SELECT_AUDIO_DEVICE)

  if CONF.HEADLESS:
    demo = HeadlessDemoApp(
      CONF.MODEL_PATH, all_labels, subset_labels, label_collections,
      CONF.SAMPLERATE, CONF.AUDIO_CHUNK_LENGTH, CONF.RINGBUFFER_LENGTH,
      CONF.MODEL_WINSIZE, CONF.STFT_HOPSIZE, CONF.STFT_WINDOW,
      CONF.N_MELS, CONF.MEL_FMIN, CONF.MEL_FMAX,
      CONF.INFERENCE_INTERVAL,
      CONF.TOP_K, CONF.HEADLESS_PRINT_INTERVAL,
      CONF.HEADLESS_MIN_CONFIDENCE, CONF.HEADLESS_LOG_PATH,
      audio_device_index)
    demo.run()
  else:
    try:
      demo = create_gui_app(
        AI4S_BANNER_PATH, logo_paths, CONF.MODEL_PATH,
        all_labels, subset_labels, label_collections,
        CONF.SAMPLERATE, CONF.AUDIO_CHUNK_LENGTH, CONF.RINGBUFFER_LENGTH,
        CONF.MODEL_WINSIZE, CONF.STFT_HOPSIZE, CONF.STFT_WINDOW,
        CONF.N_MELS, CONF.MEL_FMIN, CONF.MEL_FMAX,
        CONF.INFERENCE_INTERVAL,
        CONF.TOP_K, CONF.TITLE_FONTSIZE, CONF.TABLE_FONTSIZE,
        audio_device_index)
    except ImportError as exc:
      raise RuntimeError(
        "Tkinter GUI dependencies are unavailable. Install the GUI system "
        "packages or run with HEADLESS=True."
      ) from exc

    demo.mainloop()
