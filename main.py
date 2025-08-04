from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nemo.collections.asr as nemo_asr
from model import Process
import torch
import logging
import io
import soundfile as sf
import numpy as np
from infoExtraction import audio_info
from pydub import AudioSegment  # For audio format conversion

app = FastAPI()
logger = logging.getLogger("uvicorn.error")
model_name = "ai4bharat/indictrans2-en-indic-1B"
voice_to_text_model = "ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIVATE_MODEL = "TAMIL_VOICE_TO_TEXT" #"ENGLISH_TO_TAMIL"
# DEVICE = "cpu"


def convert_audio_to_wav(file: UploadFile):
    """Convert various audio formats to WAV with a consistent sample rate."""
    try:
        audio_data = file.file.read()
        audio_io = io.BytesIO(audio_data)

        # Use pydub to handle different audio formats
        audio = AudioSegment.from_file(audio_io)

        # Convert to mono and set sample rate to 16k if necessary
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Export to a BytesIO object as WAV
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        return wav_io
    except Exception as e:
        raise RuntimeError(f"Audio conversion failed: {str(e)}")



class MessageInput(BaseModel):
    msg: str



@app.on_event("startup")
async def load_model():
    logger.info(f"Loading model on {DEVICE}...")
    print(ACTIVATE_MODEL == "ENGLISH_TO_TAMIL", ACTIVATE_MODEL, "ENGLISH_TO_TAMIL")
    if(ACTIVATE_MODEL == "ENGLISH_TO_TAMIL"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        model_kwargs = {
            "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
            "trust_remote_code": True,
        }



        # Use flash attention only if on CUDA
        if DEVICE == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs).to(DEVICE)

        app.state.tokenizer = tokenizer
        app.state.model = model
        logger.info("Model loaded successfully.")

    elif ACTIVATE_MODEL == "TAMIL_VOICE_TO_TEXT":

        model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.freeze()  # inference mode
        model = model.to(device)  # transfer model to device
        app.state.model = model
        logger.info("Model loaded successfully.")



@app.post("/translate")
def translate(msg: MessageInput, request: Request):

    process = Process([msg.msg])
    batch = process.getBatch()

    tokenizer = request.app.state.tokenizer
    model = request.app.state.model

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens into text
    with tokenizer.as_target_tokenizer():
        decoded_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = process.translate(decoded_tokens)

    return {"data": translations[0]}


@app.post("/voice_text")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        # Convert uploaded file to WAV format with 16k sample rate
        model = request.app.state.model
        try:
            wav_io = convert_audio_to_wav(file)
            audio, sample_rate = sf.read(wav_io, dtype='float32')
            # print(f"Audio loaded: Sample rate = {sample_rate}, Data length = {len(audio)}")
            audio_info(audio)
        except Exception as e:
            return {"error": f"Audio processing failed: {str(e)}"}

        # Ensure the audio is in float32 format and not empty
        if audio.size == 0:
            return {"error": "Audio file is empty or corrupted"}

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Transcribe the audio
        try:
            ctc_text = model.transcribe([audio], batch_size=1, logprobs=False, language_id='ta')[0]
            return {"transcription": ctc_text}
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}"}

    except Exception as e:
        return {"error": f"File processing failed: {str(e)}"}
