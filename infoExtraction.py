import librosa
import soundfile as sf
import io

def audio_info(audio):
    try:
        # Check if input is a file-like object or a file path
        if isinstance(audio, io.BytesIO):
            audio.seek(0)  # Reset buffer position
            audio, sr = sf.read(audio, dtype='float32')
        else:
            audio, sr = librosa.load(audio, sr=None, mono=False)

        # Determine number of channels
        num_channels = 1 if len(audio.shape) == 1 else audio.shape[0]

        # Calculate duration
        duration = librosa.get_duration(y=audio, sr=sr)

        # Prepare the result dictionary
        info = {
            "sample_rate": sr,
            "num_channels": num_channels,
            "duration": duration,
            "shape": audio.shape,
        }
        
        return info
    except Exception as e:
        return {"error": f"Failed to process audio: {str(e)}"}