import pyaudio
from audioread.rawread import RawAudioFile
from librosa.core import resample, to_mono
from librosa import util
import wave
from io import BytesIO
from wave import Wave_write
import librosa
import numpy as np


def real_time_sound():
    CHUNK = 1000
    FORMAT = pyaudio.paInt16  # 16bit编码格式
    CHANNELS = 1  # 单声道
    RATE = 20000  # 22.05khz采样频率

    try:
        p = pyaudio.PyAudio()
        # 创建音频流
        stream = p.open(format=FORMAT,  # 音频流wav格式
                        channels=CHANNELS,  # 单声道
                        rate=RATE,  # 22.05khz采样频率
                        input=True,
                        frames_per_buffer=CHUNK)

        print("Start Recording...")

        # 录制音频数据
        byte_stream = BytesIO()
        with Wave_write(byte_stream) as ww:
            ww.setnchannels(CHANNELS)
            ww.setframerate(RATE)
            ww.setsampwidth(p.get_sample_size(FORMAT))
            while True:
                data = stream.read(CHUNK)

                ww.writeframes(data)
                bytes_io = BytesIO(byte_stream.getvalue())
                y, sr = stream_to_np(bytes_io, sr=20000)
                mel = librosa.feature.melspectrogram(y, sr=20000, n_fft=2205, hop_length=1102, n_mels=512)
                yield mel
    except Exception:
        raise
    finally:
        # 录制完成
        stream.stop_stream()
        stream.close()
        p.terminate()


def stream_to_np(bytes_io, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type='kaiser_best'):
    """
    重写了librosa.load函数，把文件参数改成bytesIO类型，并把audioread.audio_open替换为自定义的RawAudioStream类，
    因为前者需要文件路径作为参数。
    :param bytes_io:
    :param sr:
    :param mono:
    :param offset:
    :param duration:
    :param dtype:
    :param res_type:
    :return:
    """
    y = []

    with RawAudioStream(bytes_io) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

        n = 0

        for frame in input_file:
            frame = util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

        if y:
            y = np.concatenate(y)

            if n_channels > 1:
                y = y.reshape((-1, n_channels)).T
                if mono:
                    y = to_mono(y)

            if sr is not None:
                y = resample(y, sr_native, sr, res_type=res_type)

            else:
                sr = sr_native

        # Final cleanup for dtype and contiguity
        y = np.ascontiguousarray(y, dtype=dtype)

        return y, sr


class RawAudioStream(RawAudioFile):
    """
    继承rawread模块中RawAudioFile类，并把构造函数中文件路径参数替换为bytesIO类型，wave.open同样可以打开。
    """
    def __init__(self, data):

        self._file = wave.open(data)

        self._needs_byteswap = False
        self._check()
        return

    def close(self):
        self._file.close()


class MyWaveWriter(Wave_write):
    """
    继承wave模块中Wave_write类，Wave_write默认以文件路径作为参数，如果我们从麦克风传来的数据再通过一次IO写入到硬盘上，那么IO延迟会
    严重影响数据的实时性，并且对机器资源造成浪费。所以我们重写Wave_write类，把文件写入到硬盘文件改成写入到内存的bytesIO中，从而提升
    读写速度。
    """
    def __init__(self, b):
        self._i_opened_the_file = None
        if isinstance(b, BytesIO):
            self._i_opened_the_file = b
        try:
            self.initfp(b)
        except Exception:
            if self._i_opened_the_file:
                b.close()
            raise
