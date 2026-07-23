import io

import librosa
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from ray import serve

from omnivoice.serving import voice_registry
from omnivoice.serving.model_runtime import get_model
from omnivoice.serving.schemas import TTSRequest, VoiceDto, VoicesResponse

app = FastAPI(title="OmniVoice serving with Ray Serve")

# response_format -> (soundfile subtype/format, media type). Only "wav" is
# implemented for now; other OpenAI formats (mp3, opus, aac, flac, pcm) need
# a transcoder we don't have wired up yet.
_SUPPORTED_FORMATS = {
    "wav": ("WAV", "audio/wav"),
}


@app.post("/v1/audio/speech")
def text_to_speech(request: TTSRequest):
    if request.response_format not in _SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"response_format={request.response_format!r} not supported yet, "
                f"only {list(_SUPPORTED_FORMATS)} are implemented"
            ),
        )

    voice_clone_prompt = None
    if request.ref_voice is not None:
        registered = voice_registry.get_voice(request.ref_voice)
        if registered is None:
            raise HTTPException(
                status_code=404,
                detail=f"voice {request.ref_voice!r} not found, clone it via POST /v1/voices first",
            )
        voice_clone_prompt = registered.prompt

    model = get_model()
    audios = model.generate(
        text=request.input,
        language=request.language,
        instruct=request.instructions,
        voice_clone_prompt=voice_clone_prompt,
        duration=request.duration,
        speed=request.speed,
        num_step=request.num_step,
        guidance_scale=request.guidance_scale,
        t_shift=request.t_shift,
        denoise=request.denoise,
        postprocess_output=request.postprocess_output,
        layer_penalty_factor=request.layer_penalty_factor,
        position_temperature=request.position_temperature,
        class_temperature=request.class_temperature,
        audio_chunk_duration=request.audio_chunk_duration,
        audio_chunk_threshold=request.audio_chunk_threshold,
        pad_duration=request.pad_duration,
        fade_duration=request.fade_duration,
        normalize_text=request.normalize_text,
    )

    audio = audios[0]
    sample_rate = model.sampling_rate
    if request.sample_rate is not None and request.sample_rate != sample_rate:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=request.sample_rate
        )
        sample_rate = request.sample_rate

    sf_format, media_type = _SUPPORTED_FORMATS[request.response_format]
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=sf_format)
    return Response(content=buffer.getvalue(), media_type=media_type)


@app.get("/v1/voices")
def get_voices():
    return VoicesResponse(
        voices=[
            VoiceDto(name=v.name, ref_text=v.prompt.ref_text, language=v.language)
            for v in voice_registry.list_voices()
        ]
    )


@app.post("/v1/voices")
async def clone_voice(
    name: str = Form(..., description="Unique name to register the voice under."),
    ref_audio: UploadFile = File(..., description="Reference audio file."),
    ref_text: str | None = Form(
        None, description="Transcript of ref_audio. Auto-transcribed via ASR if omitted."
    ),
    ref_language: str | None = Form("Vietnamese"),
):
    try:
        voice_registry.validate_name(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if voice_registry.get_voice(name) is not None:
        raise HTTPException(status_code=409, detail=f"voice {name!r} already exists")

    audio_bytes = await ref_audio.read()
    try:
        waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not decode ref_audio: {e}")
    if waveform.ndim == 2:
        waveform = waveform.T  # (frames, channels) -> (channels, frames)

    model = get_model()
    prompt = model.create_voice_clone_prompt(ref_audio=(waveform, sr), ref_text=ref_text)
    registered = voice_registry.register_voice(name, prompt, ref_language)

    return VoiceDto(name=registered.name, ref_text=prompt.ref_text, language=registered.language)
