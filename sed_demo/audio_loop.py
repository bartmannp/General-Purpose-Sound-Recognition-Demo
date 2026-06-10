#!/usr/bin python
# -*- coding:utf-8 -*-


"""
This module contains functionality to handle the real-time input audio process.
"""


import numpy as np
import pyaudio


# ##############################################################################
# # RING BUFFER
# ##############################################################################
class RingBuffer():
    """
    A 1D ring buffer using numpy arrays, designed to efficiently handle
    real-time audio buffering. Modified from
    https://scimusing.wordpress.com/2013/10/25/ring-buffers-in-pythonnumpy/
    """
    def __init__(self, length, dtype=np.float32):
        """
        :param int length: Number of samples in this buffer
        """

        self._length = length
        self._buf = np.zeros(length, dtype=dtype)
        self._bufrange = np.arange(length)
        self._idx = 0  # the oldest location

    def update(self, arr):
        """
        Adds 1D array to ring buffer. Note that ``len(arr)`` must be anything
        smaller than ``self.length``, otherwise it will error.
        """
        len_arr = len(arr)
        assert len_arr < self._length, "RingBuffer too small for this update!"
        idxs = (self._idx + self._bufrange[:len_arr]) % self._length
        self._buf[idxs] = arr
        self._idx = idxs[-1] + 1  # this will be the new oldest location

    def read(self):
        """
        Returns a copy of the whole ring buffer, unwrapped in a way that the
        first element is the oldest, and the last is the newest.
        """
        idxs = (self._idx + self._bufrange) % self._length  # read from oldest
        result = self._buf[idxs]
        return result


# ##############################################################################
# # AUDIO INPUT STREAM (ASYNCH LOOP)
# ##############################################################################
class AsynchAudioInputStream:
    """
    This class features an asynchronous process that holds a ring buffer
    and updates it in real time with reads from a system microphone.
    """

    IN_CHANNELS = 1
    PYAUDIO_DTYPE = pyaudio.paFloat32
    NP_DTYPE = np.float32

    def __init__(self, samplerate=32000, chunk_length=1024,
                 ringbuffer_length=62*1024, input_device_index=None):
        """
        """
        self.sr = samplerate
        self.chunk = chunk_length
        self.rb_length = ringbuffer_length
        self.input_device_index = input_device_index
        # setup recording stream
        self.pa = pyaudio.PyAudio()
        self.input_device_info = self._get_selected_device_info()
        self.stream = self.pa.open(format=self.PYAUDIO_DTYPE,
                                   channels=self.IN_CHANNELS,
                                   rate=samplerate,
                                   input=True,  # record
                                   output=False,  # playback
                                   input_device_index=input_device_index,
                                   frames_per_buffer=chunk_length,
                                   stream_callback=self.callback,
                                   start=False)
        # setup audio buffer
        self.rb = RingBuffer(ringbuffer_length, self.NP_DTYPE)

    def _get_selected_device_info(self):
        if self.input_device_index is None:
            return self.pa.get_default_input_device_info()
        return self.pa.get_device_info_by_index(self.input_device_index)

    @staticmethod
    def get_input_devices():
        pa = pyaudio.PyAudio()
        try:
            devices = []
            for index in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(index)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append(info)
            return devices
        finally:
            pa.terminate()

    @staticmethod
    def get_default_input_device():
        pa = pyaudio.PyAudio()
        try:
            return pa.get_default_input_device_info()
        finally:
            pa.terminate()

    def read(self):
        """
        Returns the current reading from the ring buffer, unwrapped so
        that the first element is the oldest.
        """
        return self.rb.read()

    def start(self):
        """
        Starts updating the ring buffer with readings from the microphone.
        """
        self.stream.start_stream()

    def stop(self):
        """
        Stops updating the ring buffer (but doesn't delete its contents).
        """
        self.stream.stop_stream()

    def terminate(self):
        """
        Close the input stream (can't record afterwards).
        """
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def __enter__(self):
        """
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        """
        self.terminate()

    def callback(self, in_data, frame_count, time_info, status):
        """
        This function is automatically called by ``self.p`` every time there is
        new recorded data. By convention it returns the buffer plus a flag.

        :param in_data: Recorded data as bytestring as ``cls.PYAUDIO_DTYPE``
        :param frame_count: Number of samples in recorded data (``self.chunk``)
        :param time_info: unused
        :param status: unused
        """
        in_arr = np.frombuffer(in_data, dtype=np.float32)
        self.rb.update(in_arr)
        return (in_arr, pyaudio.paContinue)
