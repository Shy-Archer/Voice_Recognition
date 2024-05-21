import sys
import numpy as np
import librosa

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
def preprocess_audio(audio_file):
    try:
        s, sr = librosa.load(audio_file, sr=None, mono=True)
        return s, sr
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku")
        return None, None


def calculate_hps(spectrum, num_harmonics):
    hps = np.copy(spectrum[:len(spectrum) // num_harmonics])
    for i in range(2, num_harmonics + 1):
        hps *= spectrum[i - 1::i][:len(hps)]
    return hps


def detect_f0(audio_file):
    samples, sample_rate = preprocess_audio(audio_file)

    if samples is None or sample_rate is None:
        return -1
    if len(samples) == 0:
        return 0
    T = samples / sample_rate
    samples = samples / np.max(np.abs(samples))

    spectrum = np.fft.fft(samples)
    num_harmonics = 5
    hps = calculate_hps(np.abs(spectrum), num_harmonics)

    min_frequency = 80
    max_frequency = 270
    min = int(min_frequency / (44100 / len(spectrum)))
    max = int(max_frequency / (44100 / len(spectrum)))
    hps[:min] = 0
    hps[max:] = 0

    max_index = np.argmax(hps)
    f0 = max_index * (sample_rate / len(samples))
    return f0


def gender_recognition(audio_file):
    f0 = detect_f0(audio_file)

    if f0==-1:
        return

    elif f0 > 80 and f0 <= 160:
        return "M"
    else:
        return "K"


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].endswith(".wav"):
        print("Use Syntax: python Voice_Recognition.py plik.wav")
    else:
        file_path = sys.argv[1]
        result = gender_recognition(file_path)
        if result != None:
            print(result)
