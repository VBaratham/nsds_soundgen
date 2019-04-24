import random
import itertools

from scipy.signal import hanning
import scipy.io.wavfile
import numpy as np
import wave
import struct

fs = 96000 # wav sampling frequency

def silence(sil):
    return np.zeros(int(fs*sil))

def single_tone(freq, dur=0.1, sil=0.4, ramp=0.005):
    tone = np.sin(2*np.pi*freq * np.arange(0, dur, 1.0/fs))
    ramplen = int(ramp*fs)
    hann = hanning(2*ramplen)
    window = np.ones(int(dur*fs))
    window[:ramplen] = hann[:ramplen]
    window[-ramplen:] = hann[-ramplen:]
    return np.concatenate([tone*window, silence(sil)])

    
def generate_tone(n_simul, which, ratio, basefreq):
    freqs = [basefreq, basefreq*ratio, basefreq*ratio*ratio]
    tones = [single_tone(freq) for freq in freqs]

    if n_simul == 1:
        return tones[which]
    elif n_simul == 2 and which == 0:
        return (tones[1] + tones[2]) / 2.0
    elif n_simul == 2 and which == 1:
        return (tones[0] + tones[2]) / 2.0
    elif n_simul == 2 and which == 2:
        return (tones[0] + tones[1]) / 2.0
    elif n_simul == 3:
        return (tones[0] + tones[1] + tones[2]) / 3.0
    else:
        raise ValueError("n_simul must be 1, 2, or 3")


BASE_FREQS = [
    # 32.0,
    64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
    # 4096.0
]
RATIOS = [2.0/1.0, 3.0/2.0, 4.0/3.0,
          # 5.0/4.0, 6.0/5.0, 5.0/3.0, 8.0/5.0, 9.0/8.0,
          16.0/9.0, 16.0/15.0, 45.0/32.0
]
def iter_params():
    simul_12 = itertools.product(
        (1, 2,), # n_simul
        (0, 1, 2), # which
        RATIOS,
        BASE_FREQS,
    )
    simul_3 = itertools.product(
        (3,), # n_simul
        (None,), # which
        RATIOS,
        BASE_FREQS,
    )

    return itertools.chain(simul_12, simul_3)


def generate_block(n_presentations=10):
    params = list(iter_params())
    allparams = []
    alltones = [silence(1.0)]
    for i in range(n_presentations):
        random.shuffle(params)
        allparams.append(np.vstack([np.array(p, dtype=np.float32) for p in params]))
        alltones.extend(generate_tone(*p) for p in params)
    return np.vstack(allparams), np.concatenate(alltones)


def save_wav(filename, audio):
    nchannels = 1
    sampwidth = 2
    comptype = 'NONE'
    compname = 'not compressed'
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.set_params((nchannels, sampwidth, fs, len(audio), comptype, compname))

        for sample in audio:
            wav_file.writeframes(struct.pack('h', int(sample)))
    


def main():

    # plt.plot(single_tone(100.0))
    # plt.show()

    # ABOVE HERE, DEBUGGING
    # BELOW, THE REAL THING
    
    params, audio = generate_block()
    scipy.io.wavfile.write('vyassa_test.wav', fs, audio)
    np.savetxt('vyassa_test_params.csv', params)

if __name__ == '__main__':
    main()
