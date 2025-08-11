from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import nemo.collections.asr as nemo_asr
from model import Process
import torch
import logging
import io
import soundfile as sf
import numpy as np
from infoExtraction import audio_info
from pydub import AudioSegment  # For audio format conversion
import os

app = FastAPI()
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
logger = logging.getLogger("uvicorn.error")

english_to_tamil = "ai4bharat/indictrans2-en-indic-1B"
voice_to_text_model = "ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large"
text_to_proper_text = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ACTIVATE_MODEL = "ENGLISH_TO_TAMIL"
# ACTIVATE_MODEL = "TAMIL_VOICE_TO_TEXT"
ACTIVATE_MODEL = "LLAMA3_INSTRUCT"

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

    if ACTIVATE_MODEL == "ENGLISH_TO_TAMIL":
        tokenizer = AutoTokenizer.from_pretrained(english_to_tamil, trust_remote_code=True)
        model_kwargs = {
            "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
            "trust_remote_code": True,
        }

        # Use flash attention only if on CUDA
        if DEVICE == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForSeq2SeqLM.from_pretrained(english_to_tamil, **model_kwargs)
        # model.resize_token_embeddings(len(tokenizer))
        model.to(DEVICE)
        model.eval()

        # model.freeze()
        app.state.tokenizer = tokenizer
        app.state.model = model
        logger.info("Model loaded successfully.")

    if ACTIVATE_MODEL == "TAMIL_VOICE_TO_TEXT":

        model = nemo_asr.models.ASRModel.from_pretrained(voice_to_text_model)

        model = model.to(DEVICE)  # transfer model to device
        model.eval()
        model.freeze()  # inference mode
        app.state.model = model
        logger.info("Model loaded successfully.")

    if ACTIVATE_MODEL == "LLAMA3_INSTRUCT":
        logger.info("Loading Llama 3 model...")

        llama3_tokenizer = AutoTokenizer.from_pretrained(text_to_proper_text, token=hf_token)
        llama3_model = AutoModelForCausalLM.from_pretrained(
            text_to_proper_text,
            token=hf_token,
            device_map="auto" if DEVICE == "cuda" else None,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        llama3_model.eval()
        app.state.llama3_tokenizer = llama3_tokenizer
        app.state.llama3_model = llama3_model

        logger.info("Llama 3 model loaded successfully.")


@app.get("/")
async def check():
    return {"msg": "Running..."}

@app.post("/translate")
async def translate(msg: MessageInput, request: Request):

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

    with torch.inference_mode():
        generated_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=3,
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
            with torch.no_grad():
                with torch.cuda.amp.autocast():  # 🚀 Use FP16 where possible
                    ctc_text = model.transcribe([audio], batch_size=1, logprobs=False, language_id='ta')[0]
            return {"transcription": ctc_text}
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}"}

    except Exception as e:
        return {"error": f"File processing failed: {str(e)}"}


@app.post("/proper_text")
async def properText(msg: MessageInput, request: Request):
    tokenizer = request.app.state.llama3_tokenizer
    model = request.app.state.llama3_model

    # Improved system message with clearer instructions
    system_msg = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a text simplifier. Rewrite the given text to be easily understood by someone with basic education:
    1. Use simple, common words
    2. Keep sentences short (10-15 words max)
    3. Break complex ideas into multiple sentences
    4. Avoid jargon and technical terms
    5. Maintain the original meaning <|eot_id|>"""

    # Properly formatted user message
    user_text = f"<|start_header_id|>user<|end_header_id|>\n{msg.msg}<|eot_id|>"

    # Combine messages
    prompt = system_msg + user_text + "<|start_header_id|>assistant result: <|end_header_id|>\n"

    # Tokenize with attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512  # Prevent overly long inputs
    ).to(DEVICE)

    # Improved generation parameters
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,  # More precise than max_length
            num_beams=3,
            early_stopping=True,
            do_sample=True,
            temperature=0.5,  # Slightly higher for more variety
            top_p=0.95,
            repetition_penalty=1.1,  # Reduce repetition
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode while cleaning up special tokens
    full_response = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
    rewritten_text = full_response.split("assistant result:")[-1]
    return {"generated_text": rewritten_text.strip()}

