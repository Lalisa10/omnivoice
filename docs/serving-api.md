# OmniVoice HTTP serving API

HTTP layer defined in [omnivoice/serving/api_server.py](../omnivoice/serving/api_server.py),
deployed via [k8s/deployment.yaml](../k8s/deployment.yaml) + [k8s/service.yaml](../k8s/service.yaml).
Base URL below assumes the Service is reachable at `omnivoice-serve.omnivoice.svc.cluster.local:8000`
(in-cluster) or port-forwarded to `localhost:8000`:

```bash
kubectl -n omnivoice port-forward svc/omnivoice-serve 8000:8000
BASE_URL=http://localhost:8000
```

Interactive OpenAPI docs are also served at `$BASE_URL/docs` (FastAPI default).

## Known limitations of the current serving code

- **`ref_text` is effectively required** when cloning a voice. The model
  supports auto-transcribing `ref_audio` via a Whisper ASR model, but
  [omnivoice/serving/model_runtime.py](../omnivoice/serving/model_runtime.py)
  loads `OmniVoice.from_pretrained()` without `load_asr=True`, so the ASR
  model is never loaded server-side. Omitting `ref_text` will raise an error
  processing the request.
- **Only `response_format=wav`** is implemented; other OpenAI-style formats
  (`mp3`, `opus`, `aac`, `flac`, `pcm`) return `400`.
- **Cold start**: the model loads lazily on the *first* request that needs
  it (see the probe comment in `k8s/deployment.yaml`), so the first call
  after a pod starts (or restarts) will be much slower than subsequent ones.
- Voices are stored on disk (`OMNIVOICE_VOICE_DIR`), not in the response —
  clone once via `POST /v1/voices`, then reference by `name` afterwards.

---

## `POST /v1/audio/speech` — synthesize speech

Generates a WAV file from text. Three modes, chosen by which fields you set:

- **Voice clone**: set `ref_voice` to a name previously registered via
  `POST /v1/voices`.
- **Voice design**: set `instructions` (a free-text description, e.g.
  `"female, calm, British accent"`), no `ref_voice`.
- **Auto**: neither `ref_voice` nor `instructions` — the model picks a voice.

### Fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `input` | string | **required** | Text to synthesize. |
| `ref_voice` | string \| null | `null` | Name of a voice previously cloned via `POST /v1/voices`. Mutually exclusive in practice with `instructions`. |
| `instructions` | string \| null | `null` | Voice-design prompt (accent, gender, tone, ...). Used when `ref_voice` is not set. |
| `language` | string \| null | `"Vietnamese"` | Target language name (e.g. `"English"`, `"Vietnamese"`). |
| `response_format` | string | `"wav"` | Only `"wav"` is implemented today; anything else returns `400`. |
| `sample_rate` | int \| null | `null` (model native, 24000 Hz) | If set and different from the model's native rate, the server resamples before returning. |
| `duration` | float \| null | `null` | Target output duration in seconds. Overrides `speed` if both are set. |
| `speed` | float \| null | `1.0` | Speech rate multiplier. |
| `num_step` | int \| null | `32` | Diffusion denoising steps. Higher = slower, potentially higher quality. |
| `guidance_scale` | float \| null | `2.0` | Classifier-free guidance scale. |
| `t_shift` | float \| null | `0.1` | Diffusion time-shift parameter. |
| `denoise` | bool \| null | `true` | Whether to apply the denoising pass. |
| `postprocess_output` | bool \| null | `true` | Apply output post-processing (e.g. trimming). |
| `layer_penalty_factor` | float \| null | `5.0` | Penalty applied across codebook layers during decoding. |
| `position_temperature` | float \| null | `5.0` | Sampling temperature over token positions. |
| `class_temperature` | float \| null | `0.0` | Sampling temperature over token classes. |
| `audio_chunk_duration` | float \| null | `15.0` | Chunk size (seconds) used for long-text chunked generation. |
| `audio_chunk_threshold` | float \| null | `30.0` | Text/duration threshold above which chunked generation kicks in. |
| `pad_duration` | float \| null | `0.1` | Silence padding (seconds) added around chunks. |
| `fade_duration` | float \| null | `0.1` | Crossfade duration (seconds) between chunks. |
| `normalize_text` | bool \| null | `false` | Opt-in text normalization (numbers, dates, currency -> spoken form). |

Response: raw `audio/wav` bytes (`200`), or `400` if `response_format` is
unsupported.

### curl — minimal (auto voice)

```bash
curl -sS -X POST "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input": "Xin chào, đây là OmniVoice."}' \
  -o out.wav
```

### curl — voice design, all optional fields shown

```bash
curl -sS -X POST "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Xin chào, đây là OmniVoice.",
    "instructions": "female, calm, British accent",
    "language": "English",
    "response_format": "wav",
    "sample_rate": 24000,
    "duration": null,
    "speed": 1.0,
    "num_step": 32,
    "guidance_scale": 2.0,
    "t_shift": 0.1,
    "denoise": true,
    "postprocess_output": true,
    "layer_penalty_factor": 5.0,
    "position_temperature": 5.0,
    "class_temperature": 0.0,
    "audio_chunk_duration": 15.0,
    "audio_chunk_threshold": 30.0,
    "pad_duration": 0.1,
    "fade_duration": 0.1,
    "normalize_text": false
  }' \
  -o out.wav
```

### curl — voice clone (using a previously registered voice)

```bash
curl -sS -X POST "$BASE_URL/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hôm nay trời đẹp quá.",
    "ref_voice": "my-voice",
    "language": "Vietnamese",
    "speed": 1.05
  }' \
  -o out.wav
```

### Errors

| Status | Cause |
|---|---|
| `400` | `response_format` other than `"wav"`. |
| `404` | `ref_voice` set but no voice with that name is registered. |

---

## `GET /v1/voices` — list cloned voices

No parameters.

```bash
curl -sS "$BASE_URL/v1/voices"
```

```json
{
  "voices": [
    {"name": "my-voice", "ref_text": "Đây là giọng nói mẫu.", "language": "Vietnamese"}
  ]
}
```

---

## `POST /v1/voices` — clone a voice from reference audio

`multipart/form-data` request.

### Fields

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | string (form) | **required** | — | Unique voice name. Letters, digits, `_`, `-` only. `409` if it already exists. |
| `ref_audio` | file | **required** | — | Reference audio file (any format `soundfile` can decode). |
| `ref_text` | string (form) | effectively required* | `null` | Transcript of `ref_audio`. *Auto-transcription via ASR is documented but not enabled in this deployment (see Known limitations above) — omitting it will error. |
| `ref_language` | string (form) | optional | `"Vietnamese"` | Stored alongside the voice, returned by `GET /v1/voices`. |

### curl — all fields

```bash
curl -sS -X POST "$BASE_URL/v1/voices" \
  -F "name=my-voice" \
  -F "ref_audio=@/path/to/reference.wav;type=audio/wav" \
  -F "ref_text=Đây là giọng nói mẫu." \
  -F "ref_language=Vietnamese"
```

```json
{"name": "my-voice", "ref_text": "Đây là giọng nói mẫu.", "language": "Vietnamese"}
```

### Errors

| Status | Cause |
|---|---|
| `400` | Invalid `name` (bad characters), or `ref_audio` could not be decoded. |
| `409` | A voice with that `name` is already registered. |
