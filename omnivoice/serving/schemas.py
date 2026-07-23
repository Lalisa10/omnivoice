from typing import Optional

from pydantic import BaseModel


class VoiceDto(BaseModel):
    name: str   # Unique
    ref_text: Optional[str]
    language: Optional[str] = "Vietnamese"


class TTSRequest(BaseModel):
    input: str
    ref_voice: Optional[str] = None  # Name of a previously cloned voice
    instructions: Optional[str] = None
    language: Optional[str] = "Vietnamese"
    response_format: Optional[str] = "wav"
    # Output sample rate in Hz. The model always generates at its native
    # rate (24000 Hz); if set to something else, the server resamples
    # the waveform before writing the response.
    sample_rate: Optional[int] = None

    # Duration / pacing. duration overrides speed when both are set.
    duration: Optional[float] = None
    speed: Optional[float] = 1.0

    # Generation config, mirrors OmniVoiceGenerationConfig defaults.
    num_step: Optional[int] = 32
    guidance_scale: Optional[float] = 2.0
    t_shift: Optional[float] = 0.1
    denoise: Optional[bool] = True
    postprocess_output: Optional[bool] = True
    layer_penalty_factor: Optional[float] = 5.0
    position_temperature: Optional[float] = 5.0
    class_temperature: Optional[float] = 0.0
    audio_chunk_duration: Optional[float] = 15.0
    audio_chunk_threshold: Optional[float] = 30.0
    pad_duration: Optional[float] = 0.1
    fade_duration: Optional[float] = 0.1

    # Opt-in text normalization (numbers, dates, currency -> spoken form).
    normalize_text: Optional[bool] = False


class VoicesResponse(BaseModel):
    voices: list[VoiceDto]
