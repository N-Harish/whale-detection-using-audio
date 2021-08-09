# Whale detection using audio

## **Run the following command to convert audio from .aiff to .wav using ffmpeg docker**

```
1) docker build -t audioconverter -f AudioConverter.dockerfile .
2) docker run --rm -v $pwd:/srv -it audioconverter bash
3) cd /srv
4) audioconvert convert test/ test_wav/ -o .wav
```

## Technologies used

* Librosa (audio processing)
* sklearn (Machine learning)
* h5py (opening h5 files)
* mlxtend (stacking classifier)

