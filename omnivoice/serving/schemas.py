from typing import Optional

from pydantic import BaseModel, Field


class VoiceDto(BaseModel):
    name: str   # Unique
    ref_text: Optional[str]
    language: Optional[str] = "Vietnamese"


class TTSRequest(BaseModel):
    # Bounds keep one HTTP request from monopolizing the GPU: without them a
    # schema-valid request can generate for hours.
    input: str = Field(..., min_length=1, max_length=5000)
    ref_voice: Optional[str] = None  # Name of a previously cloned voice
    instructions: Optional[str] = None
    language: Optional[str] = "Vietnamese"
    response_format: Optional[str] = "wav"
    # Output sample rate in Hz. The model always generates at its native
    # rate (24000 Hz); if set to something else, the server resamples
    # the waveform before writing the response.
    sample_rate: Optional[int] = Field(None, ge=8000, le=48000)

    # Duration / pacing. duration overrides speed when both are set.
    duration: Optional[float] = Field(None, gt=0, le=300)
    speed: Optional[float] = Field(1.0, ge=0.25, le=4.0)

    # Generation config, mirrors OmniVoiceGenerationConfig defaults.
    num_step: Optional[int] = Field(32, ge=1, le=128)
    guidance_scale: Optional[float] = Field(2.0, ge=0, le=10)
    t_shift: Optional[float] = Field(0.1, gt=0, le=1)
    denoise: Optional[bool] = True
    postprocess_output: Optional[bool] = True
    layer_penalty_factor: Optional[float] = Field(5.0, ge=0, le=100)
    position_temperature: Optional[float] = Field(5.0, ge=0, le=100)
    class_temperature: Optional[float] = Field(0.0, ge=0, le=10)
    audio_chunk_duration: Optional[float] = Field(15.0, ge=5, le=30)
    audio_chunk_threshold: Optional[float] = Field(30.0, ge=10, le=120)
    pad_duration: Optional[float] = Field(0.1, ge=0, le=5)
    fade_duration: Optional[float] = Field(0.1, ge=0, le=5)

    # Opt-in text normalization (numbers, dates, currency -> spoken form).
    normalize_text: Optional[bool] = False


class VoicesResponse(BaseModel):
    voices: list[VoiceDto]
