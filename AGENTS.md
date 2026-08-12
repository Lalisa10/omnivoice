# AGENTS.md

Hướng dẫn cho Claude Code (và người mới) khi làm việc trong repo này.

## Bối cảnh dự án (quan trọng)

Đây là bản fork của [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice), một mô hình
TTS zero-shot đa ngôn ngữ (600+ ngôn ngữ) theo kiến trúc *diffusion language model*.

Mục tiêu của fork này (khác với upstream):
1. **Nghiên cứu luồng hoạt động** của mô hình OmniVoice (đọc/hiểu inference & training).
2. **Customize để serving quy mô lớn cho production** — trước mắt với **Ray**
   (Ray Serve), sau này có thể là **Triton Inference Server**.

Vì vậy khi sửa/thêm code, ưu tiên hướng tới serving throughput cao, tách rõ phần
"model logic" khỏi phần "orchestration/serving", và tránh phá vỡ API `generate()`
vốn là đơn vị phục vụ.

## Cài đặt & môi trường

- Quản lý deps bằng **uv** (khuyến nghị) hoặc pip. Python >= 3.10.
  ```bash
  uv sync                 # cài core deps (bao gồm torch CUDA cu128 trên Linux/Windows)
  uv sync --extra eval    # thêm deps cho đánh giá (WER/SIM/UTMOS)
  ```
- PyTorch CUDA được ghim ở `torch==2.8.0`/`torchaudio==2.8.0` (xem `[tool.uv]` trong
  [pyproject.toml](pyproject.toml)). Có hỗ trợ Apple Silicon (MPS) và Intel Arc (XPU) —
  `flash_attn` không có trên XPU/MPS, model tự fallback sang SDPA.
- **Chạy mọi lệnh trong WSL** (đã thống nhất). Distro: Ubuntu-24.04 (WSL2). Project
  truy cập tại `/mnt/d/Projects/OmniVoice`. Claude Code được mở trực tiếp từ terminal
  WSL nên **không cần** bọc lệnh qua `wsl.exe bash -lc '...'` nữa — chạy lệnh bash bình
  thường (kể cả các script trong [examples/](examples/)) ngay trong shell hiện tại.
- GPU: **NVIDIA RTX 5060 Laptop 8GB** (CUDA passthrough OK trong WSL). 8GB sát với
  inference `float16` → batch nhỏ khi test cục bộ, Ray Serve nên 1 replica/GPU.
- Hiệu năng I/O: chạy trên `/mnt/d` từ WSL chậm hơn ext4 native → để dataset/WebDataset
  shards lớn trong filesystem WSL (`~/...`), code giữ ở `/mnt/d`.
- Nếu tải model từ HuggingFace bị chặn: `export HF_ENDPOINT="https://hf-mirror.com"`.

## Lệnh thường dùng

```bash
# Web demo (Gradio)
omnivoice-demo --ip 0.0.0.0 --port 8001

# Inference 1 item
omnivoice-infer --model k2-fsa/OmniVoice --text "..." --output out.wav \
    [--ref_audio ref.wav --ref_text "..."] [--instruct "female, british accent"]

# Batch inference đa GPU (đọc JSONL, ghi WAV)
omnivoice-infer-batch --model k2-fsa/OmniVoice --test_list test.jsonl --res_dir results/

# Training / finetune / eval (Linux/WSL)
bash examples/run_emilia.sh      # train from scratch (3 stage: check → tokenize → train)
bash examples/run_finetune.sh    # finetune từ checkpoint
bash examples/run_eval.sh        # WER / speaker-sim / UTMOS
```

Repo chưa có test suite hay linter được cấu hình — không có lệnh test/lint chuẩn.

## Kiến trúc mã nguồn

Package chính: [omnivoice/](omnivoice/)

- **[omnivoice/models/omnivoice.py](omnivoice/models/omnivoice.py)** — trái tim của dự án.
  - `OmniVoice(PreTrainedModel)`: bọc một **LLM backbone** (mặc định `Qwen/Qwen3-0.6B`)
    làm thân, cộng `audio_embeddings` + `audio_heads` cho **8 codebook** audio
    (vocab 1025 = 1024 + 1 mask token).
  - Audio được token hoá bằng **HiggsAudioV2TokenizerModel** (24 kHz).
  - `from_pretrained()` (ghi đè): tải model + text tokenizer + audio tokenizer +
    `RuleDurationEstimator`, và tùy chọn Whisper ASR (`load_asr=True`) để tự phiên âm
    ref audio. Có `train=True` để bỏ qua phần inference-only.
  - `generate(...) -> list[np.ndarray]`: **đơn vị phục vụ chính**. 3 chế độ:
    voice clone (`ref_audio`/`ref_text` hoặc `voice_clone_prompt`), voice design
    (`instruct`), auto (không có gì). Giải mã lặp theo kiểu masked-diffusion
    (`num_step`), có `_generate_chunked` cho văn bản dài và `_generate_iterative`.
  - `forward(...)`: tính loss cho training.
- **[omnivoice/cli/](omnivoice/cli/)** — entry point (`infer.py`, `infer_batch.py`,
  `demo.py`, `train.py`); khai báo ở `[project.scripts]`.
  - `infer_batch.py` hiện dùng `ProcessPoolExecutor`, **1 worker/GPU**, model nạp trong
    `process_init` qua biến global `worker_model`, batching theo `--batch_duration`.
    Đây là chỗ tham khảo/thay thế khi chuyển sang Ray Serve (mỗi replica = 1 worker giữ model).
- **[omnivoice/data/](omnivoice/data/)** — dataset/collator/batching/processor (WebDataset).
- **[omnivoice/training/](omnivoice/training/)** — `config.py` (`TrainingConfig` load từ JSON),
  `builder.py`, `trainer.py`, `checkpoint.py`. Train đa GPU qua `accelerate`.
- **[omnivoice/eval/](omnivoice/eval/)** — WER, speaker similarity, MOS (UTMOS).
- **[omnivoice/utils/](omnivoice/utils/)** — `audio.py`, `text.py`, `duration.py`,
  `voice_design.py`, `lang_map.py`, `common.py` (`get_best_device`, `str2bool`, ...).
- **[omnivoice/scripts/](omnivoice/scripts/)** — tiền xử lý dữ liệu (denoise, trích audio
  token, JSONL→WebDataset).

## Định dạng dữ liệu

- **Batch inference / test list**: JSONL, mỗi dòng 1 object. Bắt buộc `id`, `text`;
  tùy chọn `ref_audio`, `ref_text`, `instruct`, `language_id`, `duration`, `speed`.
- **Training manifest**: JSONL, bắt buộc `id`, `audio_path`, `text`; tùy chọn `language_id`.

## Lưu ý khi hướng tới serving (Ray/Triton)

- Giữ `OmniVoice.generate()` là biên giới ổn định; wrap nó trong Ray Serve deployment
  thay vì gọi trực tiếp CLI.
- Model nặng và stateful (giữ tokenizers + estimator trên GPU) → nạp **1 lần / replica**,
  không nạp lại mỗi request; tham khảo mẫu `process_init` trong `infer_batch.py`.
- Batching động của Ray Serve có thể thay cho `--batch_duration` batching thủ công;
  `generate()` đã nhận list input nên hỗ trợ batch sẵn.
- `dtype=torch.float16` là mặc định cho inference. RTF thấp tới ~0.025.

## Quy ước

- Header license Apache-2.0 ở đầu file `.py` (theo upstream) — giữ khi thêm file mới
  vào package gốc.
- Khi đồng bộ với upstream, hạn chế sửa trực tiếp `models/omnivoice.py` theo cách gây
  xung đột merge lớn; ưu tiên đặt code serving/customize ở module/thư mục riêng.
