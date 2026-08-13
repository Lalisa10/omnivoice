# OmniVoice HTTP serving API

The Ray Serve application is defined in
[omnivoice/serving/api_server.py](../omnivoice/serving/api_server.py) and configured by
[serve_config.yaml](../serve_config.yaml) when run locally. Kubernetes supplies the
same settings through [k8s/serve-configmap.yaml](../k8s/serve-configmap.yaml), mounted
by [k8s/deployment.yaml](../k8s/deployment.yaml), and exposes the application through
[k8s/service.yaml](../k8s/service.yaml). Apply the ConfigMap before the Deployment.
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
- **Cold start**: the model loads eagerly when the Ray GPU replica starts.
  Kubernetes doesn't route traffic until `/healthz` can reach that replica,
  but rollout still needs enough startup time for model loading.
- Voices are stored on disk (`OMNIVOICE_VOICE_DIR`), not in the response —
  clone once via `POST /v1/voices`, then reference by `name` afterwards.

## Run and tune Ray Serve

Run the same application used by the container:

```bash
uv run serve run serve_config.yaml --blocking
```

The HTTP ingress runs on CPU and forwards synthesis work to one `SpeechModel`
replica that reserves one GPU. Individual HTTP requests are dynamically batched
before one vectorized `OmniVoice.generate()` call. The public HTTP API remains
request-per-audio; batching is internal.

Application logs default to `INFO`, including a line for every dynamic batch.
Set `OMNIVOICE_LOG_LEVEL` to another standard Python logging level if needed.
For example, a successful two-item batch logs
`Processing dynamic batch: items=2 compatible_sub_batches=1`.

The balanced RTX 5060 8 GB profile in `serve_config.yaml` uses:

```yaml
user_config:
  max_batch_size: 2
  batch_wait_timeout_s: 0.02
```

Ray applies these two `user_config` values through `reconfigure`, so they can be
changed and redeployed without changing Python code. Keep `max_batch_size: 1`
as the no-batching baseline. On an 8 GB GPU, benchmark before raising it above
2; compatible requests are padded to the longest generated sequence and VRAM
usage grows accordingly.

Requests with different scalar generation settings or different clone modes
are isolated into compatible sub-batches. A failed sub-batch is retried one
item at a time so a malformed request doesn't fail its neighbors. CUDA OOM also
falls back to single-item inference; a single-item OOM returns HTTP 503.

### Load benchmark

After the service is ready, run:

```bash
uv run python -m omnivoice.serving.load_test \
  --requests 20 --concurrency 4 --warmup 2
```

The command reports successful requests, errors, requests/s, generated audio
seconds/s, and mean/p50/p95/p99 latency. Compare identical runs with
`max_batch_size: 1` and `2`; useful concurrency values are 1, 2, 4, and 8.
For the balanced profile, target at least 10% more throughput at concurrency 2+
while keeping p95 latency below 1.5x the baseline. If p95 is too high, reduce
the wait timeout from 20 ms to 10 ms and then 0 ms. If batch size 2 still causes
OOM after fallback, use batch size 1.

Reference measurements on an RTX 5060 Laptop GPU (8 GB), using the default
short Vietnamese input, concurrency 4, and identical warmups:

| Steps | Batch size | Requests/s | p95 latency | Errors |
|---:|---:|---:|---:|---:|
| 8 | 1 | 3.530 | 1.255 s | 0/20 |
| 8 | 2 | 4.941 | 1.140 s | 0/20 |
| 32 | 1 | 1.001 | 4.186 s | 0/12 |
| 32 | 2 | 1.792 | 2.422 s | 0/12 |

These are local reference numbers rather than a capacity guarantee. Batch size
2 improved throughput by about 40% at 8 steps and 79% at the production-default
32 steps for this workload.

For Kubernetes, edit the matching values in `k8s/serve-configmap.yaml`, then apply
the ConfigMap and restart the Deployment so the Ray Serve process reads the new
configuration:

```bash
kubectl apply -f k8s/serve-configmap.yaml
kubectl -n omnivoice rollout restart deployment/omnivoice-serve
kubectl -n omnivoice rollout status deployment/omnivoice-serve
```

Ray exposes its standard Serve metrics plus these application metrics:
`omnivoice_batch_items_total`, `omnivoice_sub_batches_total`,
`omnivoice_batch_fallbacks_total`, `omnivoice_batch_size`, and
`omnivoice_inference_seconds`.

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
