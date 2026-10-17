# Đặc tả triển khai toàn dự án: Vietnamese ASR → English Translation trên Kaggle T4 ×2

**Phiên bản 2.0 — ngày 17/10/2026 — ngôn ngữ: tiếng Việt.**

**Mục đích:** tài liệu giao việc cho agent để xây dựng source, chuẩn bị dữ liệu, fine-tune, đánh giá và đóng gói dự án. Đọc toàn bộ trước khi triển khai. Đây là **đặc tả cần hiện thực**, chưa phải codebase đã huấn luyện hoặc notebook đã chạy thành công trên Kaggle. Các lệnh `python -m vi_asr ...` bên dưới là giao diện agent phải xây, không phải công cụ đã tồn tại. Git, pip, torchrun và API thư viện được phân biệt với CLI của nhóm.

**Thiết kế đã chốt:** ViP-VL/ChunkFormer + CTC; frozen PhoBERT hướng dẫn bằng token–frame pooling; tùy chọn temporal prior mềm; VinAI Translate vi→en giữ cố định. Hướng dẫn này hợp nhất và cập nhật thiết kế cũ: **không dùng một vector PhoBERT toàn câu làm cache chính; không lấy cấu hình single-GPU cũ làm cấu hình T4 ×2.**

**Tính khoa học:** lựa chọn kiến trúc và hyperparameter trong tài liệu là đề xuất phải kiểm chứng. Không tự tạo WER/BLEU, thời gian chạy, VRAM, hash, revision hoặc tuyên bố cải thiện. Nếu chưa chạy, ghi `not_run` kèm nguyên nhân và bước kiểm tra còn thiếu.

## Mục lục

1. Phạm vi và quyết định kiến trúc
2. Pipeline training, inference và tầng mô hình
3. Repo nào clone, repo nào chỉ tham khảo
4. Source audit và patch cần làm
5. Checkpoint, revisions và provenance
6. Cấu trúc source và sản phẩm bàn giao
7. Công việc local và Kaggle
8. Môi trường, dependencies và runtime lock
9. Chuẩn bị ViMD và kiểm soát leakage
10. Audio frontend, augmentation và CTC tokenizer
11. PhoBERT token cache
12. Token–frame pooling và temporal prior
13. Loss, masks và API model
14. Fine-tuning schedule và khởi tạo công bằng
15. Hiểu đúng T4 ×2, batch và bộ nhớ
16. DDP sampler và gradient accumulation
17. DDP training loop, AMP và lỗi phân tán
18. Validation phân tán và checkpoint/resume
19. Cấu hình YAML chuẩn
20. Baseline, ablation và ngân sách
21. Các cổng kiểm tra trước full training
22. Chuẩn bị ZIP và dữ liệu trên Kaggle
23. Notebook Kaggle theo từng cell
24. Metric, thống kê và lỗi ASR
25. VinAI, đánh giá cascade và dữ liệu tham chiếu
26. Demo upload/microphone và audio dài
27. Profiling, OOM và ước lượng thời gian
28. Tái lập, báo cáo và tiêu chí bàn giao
29. Tài liệu tham khảo
30. Prompt hoàn chỉnh giao agent

## 1. Phạm vi và quyết định kiến trúc

### 1.1 Đầu ra bắt buộc của đề bài

- Nhận WAV/MP3/FLAC hoặc microphone; resample tự động, hỗ trợ audio dài bằng chia đoạn.
- Xuất transcript tiếng Việt và bản dịch tiếng Anh có cùng `segment_id`; có thể kèm timestamp đoạn.
- Có baseline, thực nghiệm đối chứng, phân tích lỗi, demo Gradio/Streamlit, README và khả năng chạy lại.
- Đánh giá WER/CER, BLEU/chrF++, RTF/latency; dữ liệu test không được dùng chọn cấu hình.

### 1.2 Thành phần và vai trò

| Thành phần | Quyết định | Có cập nhật khi train ASR? |
|---|---|---|
| Acoustic encoder | `khanhld/vip-vl-base-vie`, ChunkFormer pretrained | Có |
| Transcript head | Linear CTC mới + SentencePiece BPE train-only | Có |
| Text teacher | `vinai/phobert-base`, frozen, token representations | Không; chạy trước để cache |
| Alignment | Cosine scorer + softmax + weighted audio pooling | Có |
| Temporal information | Gaussian prior yếu theo vị trí chuỗi; ablation bật/tắt | Không có tham số học ở bản đầu |
| Dialect/region head | Nhánh phụ 3 vùng, mở rộng sau core | Chỉ khi bật thí nghiệm riêng |
| MT | `vinai/vinai-translate-vi2en-v2` | Không |
| External baseline | Whisper-small multilingual, zero-shot trên ViMD | Không |
| ASR data | Official ViMD splits ở revision cố định | Train/valid/test riêng |

**Core nghiên cứu:** B1 = ViP-VL+CTC; P1 = B1+token pooling; P2 = P1+temporal prior. Giữ nhánh vùng miền riêng để kiểm tra đóng góp, không đưa đồng thời mọi cải tiến rồi kết luận alignment tốt.

Không huấn luyện SSL từ đầu, không joint-train VinAI cùng ASR, không đưa direct speech translation/RNN-T/GCN/kNN thành yêu cầu core. Đổi backbone là thay đổi thí nghiệm cần ghi lại, không phải sửa lỗi kỹ thuật thông thường.

### 1.3 Tên phương pháp và giới hạn phát biểu

Tên mô tả phù hợp: **Token–Frame Attention Pooling with a Soft Temporal Prior for Text-Guided Vietnamese CTC ASR**.

Đơn vị alignment là **subword token PhoBERT**, không mặc nhiên là từ. Temporal prior khuyến khích tập trung theo thứ tự tương đối; **không bảo đảm căn chỉnh đơn điệu, liên tục hoặc timestamp chính xác**. Không gọi là word-level forced alignment nếu chưa có bước gộp từ và đánh giá boundary.

Nếu mới báo cáo WER theo vùng mà chưa có cơ chế học theo vùng, mô tả là *multi-dialect evaluation*. Chỉ dùng tên *dialect-aware learning* cho thí nghiệm có cơ chế tương ứng và ablation.

## 2. Pipeline training, inference và tầng mô hình

### 2.1 Pipeline huấn luyện

```text
Audio → mono 16 kHz → 80-d fbank → checkpoint global CMVN
                                         ↓
                                 ViP-VL encoder
                                         ↓ H[B,T,512]
                   ┌─────────────────────┼──────────────────────┐
                   ↓                     ↓                      ↓ tùy chọn
          Linear(512,V)          Token–frame pooling       masked mean
                   ↓                     ↑                      ↓
                CTC loss        frozen PhoBERT tokens      Linear(512,3)
                   ↑                     ↓                      ↓
           ASR target IDs         token cosine loss         region CE
```

Một encoder forward/microbatch. Nhánh CTC giữ chuỗi frame H. Text tạo queries cho nhánh phụ, **không truyền reference tokens/length vào đường nhận dạng dùng để suy luận**.

### 2.2 Pipeline triển khai

```text
File / microphone → decode/resample → VAD/segment queue
   → fbank/CMVN → ViP-VL → CTC greedy → transcript vi
   → chuẩn hóa/chia câu cho MT → VinAI Translate → English text
   → UI hiển thị theo segment_id
```

Inference loại PhoBERT, scoring projections, pooling loss và region head. Export vẫn phải chứa cấu hình frontend, CMVN và tokenizer. Không chạy VinAI trong training loop ASR.

### 2.3 Tầng và tensor shapes

| Tầng | Cấu hình | Output |
|---|---|---|
| Frontend | 16 kHz, 80 mel, frame length 25 ms, shift 10 ms | `[B,F,80]` |
| Subsampling | `dw_striding` 8×, dùng length/mask thật từ source | `[B,T,512]` |
| Encoder | 12 blocks, hidden 512, 8 heads, FFN 2048, conv kernel 15 | `[B,T,512]` |
| CTC head | `Linear(512,V)`, V từ tokenizer artifact | `[B,T,V]` |
| PhoBERT frozen | last hidden states; bỏ special/padding | `[B,N,768]` |
| Scoring keys | `Linear(512,256)` + L2 normalize | `[B,T,256]` |
| Scoring queries | `Linear(768,256)` + L2 normalize | `[B,N,256]` |
| Attention | cosine + optional prior + frame softmax | `[B,N,T]` |
| Audio pooling | `A @ H`, values chỉ từ audio | `[B,N,512]` |
| Teacher prediction | `Linear(512,768)` | `[B,N,768]` |

Tái sử dụng nguyên backbone; mỗi encoder block giữ cấu trúc macaron FFN, attention, convolution, FFN và residual/normalization của source. Không đổi sang Transformer thuần. Kiến trúc/checkpoint: [ViP-VL model card](https://huggingface.co/khanhld/vip-vl-base-vie).

`T≈F/8` chỉ để ước lượng; không tính CTC lengths bằng chia nguyên tùy ý. Ví dụ source đã đối chiếu cho 1.000 input frames có thể trả 124 output frames. Hidden state có ngữ cảnh rộng, không đại diện một lát audio độc lập đúng 80 ms.

## 3. Repo nào clone, repo nào chỉ tham khảo

### 3.1 Bảng quyết định

| Repo / artifact | Có clone? | Tái sử dụng gì? | Chỉnh ở đâu? |
|---|---|---|---|
| [khanld/chunkformer](https://github.com/khanld/chunkformer) | **Bắt buộc**, pin SHA | Encoder và dependencies, CMVN, frontend tham khảo | Wrapper của nhóm; upstream patch tối thiểu có diff |
| [VinAIResearch/PhoBERT](https://github.com/VinAIResearch/PhoBERT) | Không bắt buộc | Usage/preprocessing/citation; weights qua Transformers | `teacher.py`, `teacher_cache.py` của nhóm |
| [VinAIResearch/VinAI_Translate](https://github.com/VinAIResearch/VinAI_Translate) | Không bắt buộc | Cách load/generate và lưu ý benchmark | `translation.py` của nhóm |
| [nguyen-dv/ViMD_Dataset](https://github.com/nguyen-dv/ViMD_Dataset) | Đọc; clone tùy chọn | Schema, split, citation | Data adapter/manifests của nhóm |
| [vncorenlp/VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) | Không cần clone toàn bộ | Word segmenter qua wrapper + assets | Preparation script của nhóm |
| [openai/whisper](https://github.com/openai/whisper) | Không cần nếu dùng pinned package | Baseline multilingual transcription | Baseline adapter của nhóm |
| [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | Không cần nếu dùng pinned package/assets | VAD phục vụ demo audio dài/mic | Segmenter adapter của nhóm |
| MFGCN paper | Không có repo tác giả đã xác minh trong hồ sơ hiện tại | Ý tưởng text-conditioned weighted pooling | Alignment module tự hiện thực, ghi adaptation |
| Deng et al. ICASSP 2022 | Không bắt buộc clone reimplementation | Text-guided CTC và token mapping | Tham khảo nguyên lý, không nhận bên thứ ba là official |

Không clone Transformers/PyTorch/fairseq/WeNet chỉ để ghép file. Dùng package và pin version; chỉ fork khi có lỗi cụ thể cần patch. Repo PhoBERT/VinAI chủ yếu cung cấp hướng dẫn/checkpoint; không giả định clone là có đầy đủ pipeline train mà nhóm cần.

### 3.2 Clone backbone

Các lệnh Bash dưới đây chạy tại root project mới trong Linux/WSL hoặc Kaggle terminal, không đặt trong `sources/` của thư mục đồng bộ:

```bash
git clone https://github.com/khanld/chunkformer.git third_party/chunkformer
git -C third_party/chunkformer checkout --detach 24b12583947d2338bb39f005637ee43a64c364e7
git -C third_party/chunkformer rev-parse HEAD
```

SHA trên là snapshot đã đối chiếu trong tài liệu trước, không phải tuyên bố đó là `main` mới nhất. Agent kiểm tra fetch được, lưu SHA và hash source. Không dùng shallow clone rồi tự chuyển sang main khi commit không có. Có thể tải archive đúng SHA nếu Git bị chặn; giữ LICENSE và metadata nguồn.

### 3.3 Cách vendor source

Giữ package nội bộ `chunkformer/` đủ dependencies; không copy vài file rồi thiếu imports bắc cầu. Giữ nguyên LICENSE/header. Các patch lưu `third_party/patches/0001-*.patch`, kèm lý do, SHA nguồn, test và trạng thái áp dụng.

Model/head/trainer/data của nhóm nằm trong `src/vi_asr/`, không nhồi vào upstream. Ghi `THIRD_PARTY.md`: file dùng nguyên trạng, file sửa, code tự viết, weights tải sẵn và citations tương ứng.

## 4. Source audit và patch cần làm

### 4.1 File bắt buộc đọc trong ChunkFormer

| File | Cần kiểm tra |
|---|---|
| `chunkformer/modules/encoder.py` | Constructor, `forward_encoder`, masks, full/chunk context, LayerDrop, checkpoint execution |
| `modules/encoder_layer.py` | FFN/attention/conv/residual; arguments và cache tensors |
| `modules/subsampling.py` | Length/mask qua striding |
| `modules/attention.py`, `embedding.py` | Chunk relative attention và positional encoding |
| `modules/convolution.py` | Conv normalization, context và padding |
| `modules/ctc.py`, `asr_model.py` | Loss reduction, blank ID, hybrid decoder defaults |
| `modules/cmvn.py`, `utils/cmvn.py` | Checkpoint CMVN và buffers |
| `dataset/processor.py` | Waveform scale trước fbank, dither, frame settings |
| `utils/init_model.py` | Factory có dựng attention decoder ngoài nhu cầu không |
| `chunkformer/__init__.py` | Import eager lớp inference và dependency kéo theo |
| `examples/asr/ctc/run.sh`, `examples/ssl/vipvl/` | Recipe chỉ dùng tham khảo; không chạy SSL recipe để fine-tune CTC |
| `pyproject.toml`, `LICENSE` | Python/dependency và license ở revision thực tế |

Xuất `docs/source_audit.md` có class/function, signature, input/output shapes, mask semantics, reuse/change, đường kiểm tra và kết luận. Đọc source không đồng nghĩa đã xác nhận forward bằng checkpoint trên GPU.

### 4.2 Thin wrapper encoder + CTC

Factory upstream có thể tạo attention decoder và ASRModel có CTC/attention mixing. Để core là pure CTC, dựng `ChunkFormerEncoder` đúng config + CMVN, thêm CTC linear riêng. Không tạo decoder rồi bỏ loss trong khi vẫn giữ parameters không dùng trong DDP.

Adapter nội bộ chuẩn hóa `encode(feats, lengths) -> H, output_lengths, valid_mask`; trả bool mask `True=frame thật`. Đối chiếu mask upstream `[B,1,T]`, không đoán `True=padding`.

### 4.3 Cảnh báo quan trọng: gradient checkpointing

Trong bản `encoder.py` lưu ở lần audit trước, `gradient_checkpointing` được khai báo/gán, nhưng đường `forward_layers` đang gọi trực tiếp `layer(...)`; chưa thấy `torch.utils.checkpoint.checkpoint` trên đường này. **Chỉ đặt flag=true chưa chứng minh giảm activation memory.** Agent phải xác minh lại trên SHA đã clone.

Nếu vẫn chưa hoạt động, tạo patch bật checkpoint cho từng encoder block trong training:

- Dùng `torch.utils.checkpoint.checkpoint` với `use_reentrant=False`, bảo toàn RNG dropout.
- Giữ nguyên input/masks/context và tensor outputs của layer; evaluation gọi như cũ.
- Bind layer/context theo từng vòng lặp bằng hàm/partial riêng. Không để closure Python tham chiếu biến `layer` cuối vòng khi backward recompute.
- Không checkpoint toàn bộ model/loss trong bản đầu; không biến tensor thành CPU hoặc detach.
- So sánh output và gradient bật/tắt với dropout tắt; đo actual peak memory và thời gian.
- Profiling phải chứng minh recomputation. Nếu chưa làm patch/test, cấu hình ghi `activation_checkpointing: false`; không tuyên bố đã bật.

API và các lựa chọn checkpoint: [PyTorch checkpoint](https://docs.pytorch.org/docs/2.9/checkpoint.html). Đây là patch vận hành, không coi là đóng góp thuật toán.

### 4.4 LayerDrop và import

Đặt `encoder_layerdrop=0.0` cho core DDP để tránh mỗi rank bỏ các layer khác nhau, dẫn đến unused parameters hoặc graph khác. Giữ các dropout khác đồng nhất giữa ablation.

`chunkformer/__init__.py` import inference wrapper có thể kéo dependency ngoài encoder. Nếu cần lazy import, ghi patch chỉ thay import, kiểm tra không đổi encoder weights/forward. Không cài DeepSpeed chỉ để vượt một import chưa cần; cũng không xóa dependency ngẫu nhiên mà chưa kiểm tra.

## 5. Checkpoint, revisions và provenance

### 5.1 Snapshot đã dùng làm điểm xuất phát

| Artifact | Repo ID | Revision đã đối chiếu trước đây |
|---|---|---|
| Source | `khanld/chunkformer` | `24b12583947d2338bb39f005637ee43a64c364e7` |
| Encoder | `khanhld/vip-vl-base-vie` | `4b259337d96450d57403acdee82572c587151e22` |
| Teacher | `vinai/phobert-base` | `01daacda68afe13d83023d16ec647239e344a1e6` |
| MT | `vinai/vinai-translate-vi2en-v2` | `ae7baa85da07dbe8e23ac26a9f5ef560c17e2138` |
| Dataset | `nguyendv02/ViMD_Dataset` | `3a5b30157034e7eadd5c75fae1a820c6f9383398` |

Những revision này chưa phải hash file tải trong một training run. Agent resolve/check lại, lưu snapshot metadata và SHA256 các file thực dùng. Whisper-small, VnCoreNLP, VAD, Python packages cần chốt version/assets hash trong preparation; không để `latest/main` chưa resolve trong thí nghiệm chính.

### 5.2 Tải weights

- ViP-VL: `pytorch_model.pt`, `config.yaml`, `global_cmvn`, model card/license; encoder-only, task head khởi tạo mới.
- PhoBERT: tokenizer/config và **một** bộ weights tương thích loader; giữ revision giống cache metadata.
- VinAI: tokenizer/config/generation settings và một bộ weights; tải ở phiên evaluation hoặc lưu artifact riêng.
- Whisper: official backend + multilingual `small`, ghi checksum/model identity, không dùng `small.en`.
- VnCoreNLP: wrapper version, Java runtime, segmenter assets hashes.

Không tải cả `.bin` và `.safetensors` nếu không dùng cả hai. Không sửa weights trong HF cache. API chuẩn: [HF download guide](https://huggingface.co/docs/huggingface_hub/guides/download).

```python
from huggingface_hub import hf_hub_download

for name in ("pytorch_model.pt", "config.yaml", "global_cmvn"):
    path = hf_hub_download(
        repo_id="khanhld/vip-vl-base-vie",
        revision="4b259337d96450d57403acdee82572c587151e22",
        filename=name,
        cache_dir="/kaggle/working/hf_cache",
    )
    print(name, path)
```

### 5.3 Kiểm tra nạp encoder

1. Đọc checkpoint keys và config gốc, không đoán cấu trúc nested dict.
2. Dựng encoder đúng architecture; load CMVN đúng một lần.
3. Chọn/strip prefix `encoder.` phù hợp wrapper.
4. Assert mọi encoder parameter cần dùng có key/shape tương ứng; xuất load report.
5. Missing CTC/projection mới là bình thường; missing encoder không được bỏ qua.
6. So sánh một số tensor sau load, đếm số parameters load được, ghi weights SHA256.

`strict=False` không tự bỏ qua shape mismatch và không chứng minh model đã dùng pretrained weights. Không copy `optim.lr=0.001`, 800 epochs, quantizer hoặc crop/mask SSL thành cấu hình fine-tune ASR.

## 6. Cấu trúc source và sản phẩm bàn giao

```text
vi_asr_project/
  README.md
  IMPLEMENTATION_STATUS.md
  THIRD_PARTY.md
  pyproject.toml
  requirements-kaggle.in
  requirements-kaggle.lock.txt          # chỉ sau runtime test
  requirements-extra-kaggle.lock.txt    # phần cài thêm; bảo vệ torch/torchaudio runtime
  configs/
    base.yaml
    variants/{b1,p1,p2,d1,p2d}.yaml
    profiles/{cpu_smoke,t4_single,t4x2}.yaml
    paths/{local,kaggle}.yaml
    evaluation.yaml
  src/vi_asr/
    __main__.py
    cli.py
    config.py
    data/{prepare,audit,manifest,normalize,frontend,augment,tokenizer}.py
    data/{dataset,collate,distributed_sampler,teacher_cache}.py
    models/{encoder_adapter,ctc_asr,alignment,teacher,dialect,translation}.py
    training/{trainer,losses,distributed,checkpoint,seed,schedule}.py
    evaluation/{asr,cascade,metrics,bootstrap,benchmark}.py
    inference/{segmenter,transcribe,pipeline}.py
    utils/{hashing,environment,logging}.py
  scripts/{package_project,verify_bundle}.py
  notebooks/{01_prepare,02_train_t4x2,03_evaluate,04_demo}.ipynb
  tests/
    test_frontend_parity.py
    test_ctc_contract.py
    test_alignment_masks.py
    test_checkpoint_patch.py
    test_ddp_gradient_equivalence.py
    test_ddp_sampler.py
    test_resume.py
    test_metrics.py
    test_audio_only_inference.py
  metadata/{source_lock,normalization_policy,region_mapping}.json
  docs/{source_audit,protocol,kaggle_runbook,limitations}.md
  third_party/{chunkformer,patches}/
  licenses/
  artifacts/                          # không đưa weights/cache vào code ZIP
  runs/                               # không đưa kết quả vào code ZIP
```

Các module này là tên/API yêu cầu, không phải khẳng định đã tồn tại. Root package dùng `src` layout và `python -m vi_asr`. Notebook chỉ điều phối, không copy model/trainer thành phiên bản thứ hai.

Bốn gói khác nhau: **code ZIP**, **prepared data/cache**, **ASR inference bundle**, **results archive**. Source ZIP không chứa raw data, HF cache, `.git`, credentials, môi trường Python, checkpoint hoặc outputs notebook lớn. Giữ vendor license/source đủ tái lập.

## 7. Công việc local và Kaggle

| Giai đoạn | Local/agent làm trước | Kaggle thực hiện |
|---|---|---|
| Source | Clone/audit/patch/wrapper/head/CLI/tests/config | Import và kiểm tra runtime |
| Dữ liệu | Viết adapter, schema, audit rules | Tải/stream ViMD, materialize audio, đếm/audit |
| Tokenizer | Viết train-only builder và roundtrip tests | Học tokenizer từ transcript train |
| Teacher | Viết segmenter/token mapping/cache builder | Tạo token cache trong phiên riêng |
| Training | Hoàn thành trainer/DDP/sampler/resume | 1-GPU smoke → 2-GPU pilot → main runs |
| Evaluation | Viết metrics/translation/cascade | Decode valid/test, dịch và đo thực tế |
| Demo | Xây queue/segment UI | Kiểm tra hoặc chạy ở môi trường triển khai |

Không bắt máy local tải dataset hàng chục GB hoặc load mọi model. Sau sửa trực tiếp trên Kaggle phải đồng bộ lại source/config, tạo code revision mới. Không lấy notebook sửa tay chưa lưu vào source làm kết quả có thể tái lập.

## 8. Môi trường, dependencies và runtime lock

### 8.1 Chính sách

Ưu tiên Linux/WSL2 để tương đồng Kaggle. Source snapshot đã audit yêu cầu Python >=3.11; agent đọc lại `pyproject.toml` thực tế. Giữ cặp torch/torchaudio CUDA đã hoạt động trong image nếu phù hợp; **không nâng cấp toàn bộ thư viện mù quáng**.

Dependencies: torch, torchaudio, transformers, huggingface_hub, datasets/pyarrow, sentencepiece, numpy/scipy/soundfile, PyYAML, jiwer, sacrebleu, tqdm. Preparation thêm py_vncorenlp/Java và decoder audio phù hợp; demo thêm Gradio/VAD/FFmpeg theo backend. Dependencies vendor cần import audit riêng.

Không ghi bộ version “đã test trên Kaggle” khi chưa có runtime. Tạo `.in` trước, resolve theo image thực tế, load thử checkpoint và chạy smoke rồi mới lưu lock. Loader có thể yêu cầu torch mới hơn minimum trong upstream; không tắt kiểm tra của loader để ép load định dạng không được hỗ trợ.

Ghi `environment.json`: Python, Linux/image, torch/torchaudio/transformers/datasets, CUDA build/driver/cuDNN/NCCL, GPUs/VRAM/compute capability, CPU/RAM/disk, FFmpeg/Java. `pip freeze` là snapshot bổ sung, không thay cho lock và smoke tests.

### 8.2 T4 và precision

Chọn **FP16 autocast + GradScaler** cho core T4. Không mặc định BF16 hoặc FlashAttention-2; chỉ bật backend nếu phần cứng/thư viện thật sự hỗ trợ và có parity test. CTC log-softmax/loss và các phép normalize/softmax nhạy số học dùng FP32. Không gọi `model.half()` cho toàn bộ model để thay AMP.

API cụ thể phải khớp torch đã pin, ví dụ `torch.autocast("cuda", dtype=torch.float16)` và `torch.amp.GradScaler("cuda")` trên phiên bản hỗ trợ. [AMP examples](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html).

## 9. Chuẩn bị ViMD và kiểm soát leakage

### 9.1 Dataset đúng

ID là **`nguyendv02/ViMD_Dataset`**, không phải `nguyendv02/ViMD`. Giữ official splits; truy vấn tên split thực tế rồi map vào `train/valid/test`, lưu mapping. Không trộn tất cả rồi chia ngẫu nhiên theo dòng.

Nguồn: [ViMD HF](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset), [paper EMNLP 2024](https://aclanthology.org/2024.emnlp-main.426/). Số mẫu/giờ paper và HF snapshot có thể khác; báo cáo số **thực đã đọc** và exclusions, không ép parser tạo đúng con số paper.

### 9.2 Đường dữ liệu ưu tiên

Preparation stream/download pinned snapshot → decode từng mẫu → mono/resample 16 kHz → lưu local WAV PCM16 hoặc FLAC lossless → JSONL manifests. Main runs đọc local files, không phụ thuộc HTTP streaming/shuffle iterator trong mỗi epoch.

HF audio backend tùy version có thể trả dict array/sampling_rate hoặc decoder object; viết adapter và test một mẫu thật trước khi duyệt hết. Nếu cần đọc encoded bytes, chọn `Audio(decode=False)` theo API của version đã lock, decode bằng SoundFile/FFmpeg phù hợp. Không tải toàn bộ dataset vào RAM. [HF audio loading](https://huggingface.co/docs/datasets/audio_load).

Ví dụ dung lượng **lý thuyết**: mono PCM16 16 kHz = 32.000 bytes/giây. Nếu có 102,56 giờ thì WAV khoảng 11,81 GB thập phân trước overhead. Dataset download gốc có thể lớn hơn nhiều; đây không phải dung lượng HF hay disk bắt buộc. Tính từ duration thực và kiểm tra temp/cache/output song song.

### 9.3 Manifest

Mỗi dòng có `sample_id`, split, `audio_relpath`, source revision/filename, speaker ID, region/province/gender nếu có, sample rate, num_samples/duration, `text_raw`, `text_asr`, `text_teacher`, audio/text hashes. Hash file bytes và canonical decoded waveform là hai khái niệm khác; lưu rõ thuật toán.

Đường dẫn audio tương đối so với data root, không hard-code `/kaggle/working` vào manifest. `sample_id` ổn định, không chỉ dùng row index sau shuffle. Region map nếu bật: North=0, Central=1, South=2; unknown phải audit, không tự gán nhãn.

### 9.4 Audit và test lock

- Đếm samples/hours/speakers theo split, region, duration, token lengths; kiểm tra decode lỗi, silence/clipping/NaN.
- Kiểm tra exact audio duplicates và speaker overlap giữa splits; ghi report, không tuyên bố speaker-disjoint trước khi kiểm tra.
- Transcript trùng không tự là leakage; nhiều người có thể đọc cùng câu. Ghi riêng exact/near-duplicate và nguồn.
- Không dùng test để học tokenizer, tính CMVN mới, chọn λ/LR/epoch/decoder hoặc sửa rule theo lỗi test.
- Test có thể được kiểm tra integrity tự động theo rule định trước; không xem WER rồi sửa pipeline.
- Nếu official split có overlap, giữ protocol official và báo cáo hạn chế; muốn tạo protocol speaker-disjoint thì version riêng, định nghĩa trước khi main runs, không âm thầm thay test.
- Không cắt ngẫu nhiên audio rồi giữ transcript toàn câu. Nếu phải chia audio train cần nhãn segment đúng; bản đầu giữ nguyên utterance, dùng batching/checkpointing để xử lý độ dài.

Mọi exclusion có sample ID, rule, split và lý do. Lưu raw manifests, processed manifests, `audit.json`, `exclusions.jsonl`, `split_overlap.json` và hash.

### 9.5 Điều khoản dữ liệu

Card ViMD hiện ghi CC BY-NC-ND 4.0. Giữ attribution/điều khoản nguồn; dùng bản xử lý nội bộ cho nghiên cứu và không tự công khai bản audio biến đổi/cache có thể suy ra nội dung nếu chưa xác nhận quyền chia sẻ. Source ZIP không chứa dữ liệu người nói. Dữ liệu nhóm tự thu cần consent và xử lý thông tin nhạy cảm. Phân biệt code license, model license và dataset license, không gán một license chung cho cả project.

## 10. Audio frontend, augmentation và CTC tokenizer

### 10.1 Frontend phải khớp pretrained encoder

1. Decode waveform float về khoảng [-1,1]; xác nhận loader thực tế.
2. Chuyển mono theo policy cố định, resample về 16 kHz.
3. Trước `torchaudio.compliance.kaldi.fbank`, upstream đã audit nhân waveform với `1 << 15` (32.768). Không bỏ bước này hoặc nhân hai lần.
4. Fbank 80 mel, 25 ms, shift 10 ms, window `povey`, `energy_floor=0.0` theo source được pin; kiểm tra các defaults còn lại.
5. Dither train theo checkpoint recipe (điểm xuất phát 1.0), eval 0.0 để deterministic; khai báo rõ nếu thay đổi.
6. Dùng `global_cmvn` của checkpoint. Nếu encoder tự CMVN thì không CMVN lần nữa ở dataset.

So sánh frontend của nhóm với upstream cùng waveform, dither=0; assert shape và sai số. Sai scale/CMVN có thể làm fine-tune thất bại dù loss vẫn chạy. Không lấy config frontend từ Whisper rồi đưa trực tiếp vào ViP-VL.

### 10.2 Augmentation

Core khởi đầu: SpecAugment chỉ trên train, 2 frequency masks tối đa 8 bins và 2 time masks tối đa 30 input frames; cap theo tỷ lệ độ dài mẫu để không che toàn bộ câu ngắn. Giữ policy giống B1/P1/P2. Vị trí trước/sau CMVN phải được chốt và test theo wrapper; đề xuất sau CMVN, trước subsampling.

Nếu CMVN nằm trong encoder, cần hook/wrapper rõ ràng để đặt SpecAugment đúng vị trí, không vừa preprocess CMVN vừa để encoder áp lại. Có thể đặt SpecAugment trước CMVN như một lựa chọn khác nhưng phải ghi và giữ nhất quán; không gọi hai policy tương đương.

Speed perturb/noise là mở rộng, không cần bật ngay. Với speed perturb phải cập nhật lengths; transcript và token teacher giữ nếu nội dung không đổi. Với noise giữ transcript, log SNR/policy. Không text augmentation bằng đổi từ/paraphrase trong cặp audio–transcript rồi giữ audio cũ. Không random crop âm thanh có transcript toàn câu.

### 10.3 Ba luồng văn bản

- `text_raw`: giữ nguyên nguồn.
- `text_asr`: Unicode NFC, giữ dấu tiếng Việt; policy lowercase/punctuation/numbers định nghĩa bằng code/version, dùng cho tokenizer và metric.
- `text_teacher`: text phù hợp PhoBERT, word segmentation bằng VnCoreNLP; không dùng dấu gạch dưới phân đoạn từ làm input VinAI.

Không tự sửa lời nói địa phương thành câu chuẩn nếu transcript ASR cần phản ánh lời nói. Việc chuẩn hóa dialect là nhiệm vụ khác. Chuẩn hóa số phải nhất quán giữa reference/hypothesis và không xem test để viết exception.

### 10.4 CTC tokenizer độc lập PhoBERT

Điểm xuất phát: SentencePiece BPE target vocab 1.024, `character_coverage=1.0`, học **chỉ từ train.text_asr**. Tắt normalization ngầm ngoài policy đã chốt nếu API hỗ trợ; lưu full trainer args và corpus hash. Actual V lấy từ artifact, không hard-code 1.024 vào head.

Quy ước đơn giản: `pad_id=0`, `unk_id=1`, `bos_id=-1`, `eos_id=-1`; ID 0 đồng thời dùng làm CTC blank. Tokenizer không được phát padding/blank trong target hợp lệ. Nếu dùng blank riêng, cần mapping/token offset rõ ràng và tests; không trộn hai quy ước.

CTC targets nối phẳng + `target_lengths`; không đưa special tokens PhoBERT vào CTC. Kiểm tra roundtrip và tỷ lệ UNK theo split, không học lại vocabulary từ valid/test.

### 10.5 Điều kiện hợp lệ CTC

Với U target tokens và R cặp token giống nhau liền kề, cần `T_out >= U + R`. Tính lengths thật sau frontend/subsampling, kể cả khi bật augmentation. `zero_infinity=False` trong smoke để lộ lỗi; main không dùng zero-infinity để âm thầm bỏ mẫu.

Nếu có mẫu bất khả thi, audit nguyên nhân/tokenizer/lengths trước; không crop transcript, không lấy test để chọn vocab. Rule xử lý dữ liệu chốt trước main và báo coverage. [CTCLoss documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CTCLoss.html).

## 11. PhoBERT token cache

### 11.1 Tạo cache một lần

`transcript → NFC/policy → VnCoreNLP wseg → PhoBERT tokenizer → frozen PhoBERT eval → content-token hidden states`.

Chạy `eval()` và `no_grad/inference_mode`, không đưa PhoBERT vào ASR optimizer/DDP. Cache float16, chuyển FP32 khi tính loss cần độ chính xác. Bỏ `<s>`, `</s>`, `<pad>` và các special không thuộc nội dung. Giữ đúng thứ tự token và occurrence; hai lần xuất hiện cùng từ là hai vị trí riêng.

**Không mean-pool thành một vector câu.** Cache phải giữ `[N_i,768]` cho từng utterance. PhoBERT cần text đã word-segmented; tham khảo [hướng dẫn chính thức](https://github.com/VinAIResearch/PhoBERT).

### 11.2 Định dạng cache

Ưu tiên shard NumPy/memmap/safetensors phù hợp random access thay vì hàng vạn file nhỏ:

```text
teacher_cache/
  meta.json                # model/tokenizer/segmenter revision, hashes, layer, dtype
  index.jsonl              # sample_id → shard, offset, N, text/token hash
  tokens-000.npy           # [sum N,768], float16
  token_ids-000.npy        # [sum N], integer
  offsets-000.npy          # [num_samples+1], int64
```

Metadata có normalization/word segmentation versions, content-special policy, model layer, max length/window policy và manifest hash. Cache miss hoặc mismatch phải báo lỗi, không âm thầm trả vector zero.

Ví dụ lý thuyết: 15.000 mẫu, trung bình 60 token, 768 chiều, FP16 → `15000×60×768×2 = 1.382.400.000 bytes`, khoảng 1,38 GB trước index/metadata. Số thật tính từ tổng token cache, không dùng ước lượng làm yêu cầu disk tuyệt đối.

### 11.3 Chuỗi dài vượt giới hạn teacher

PhoBERT-base có max input length 256, phải trừ special tokens. Không `truncation=True` rồi giả vờ cache đủ toàn transcript. Chia content token IDs thành windows hợp lệ, encode với special tokens và ghép lại theo mapping; mỗi token content xuất hiện đúng một lần trong cache cuối, hoặc có policy overlap/average được ghi rõ.

Giữ một sequence N của toàn utterance sau ghép windows. Windowing giới hạn context teacher, cần báo limitation. Đây là xử lý text teacher, không tự chia audio theo token count. Token count PhoBERT và target CTC khác nhau là bình thường; không bắt N=U.

Chỉ cần cache train để tính loss. Cache valid chỉ dùng diagnostics định trước, không dùng reference ở đường decode WER; không cần cache test cho inference hoặc lựa chọn phương pháp.

## 12. Token–frame pooling và temporal prior

### 12.1 Scorer và supervision target tách biệt

Với H `[B,T,512]`, frozen E `[B,N,768]`:

```text
K = normalize(P_k(H))                       # [B,T,256]
Q = normalize(P_q(stopgrad(E)))              # [B,N,256]
S = Q @ transpose(K, -1, -2)                 # [B,N,T], cosine
A = softmax(frame_mask(S / tau + prior))    # normalize trên T
C = A @ H                                  # [B,N,512], values chỉ audio
E_hat = P_v(C)                              # [B,N,768]
```

`P_k:512→256`, `P_q:768→256`, `P_v:512→768` học được. Target loss là **E nguyên 768 chiều, frozen**, không phải E qua P_q trainable. Điều này tránh một nguồn target drift, chưa loại mọi shortcut do text-conditioned weights.

Không cộng E trực tiếp vào C/E_hat; không nối E vào CTC logits. Không khẳng định cosine giữa encoder gốc và teacher gốc đã có ý nghĩa; projections cần học.

### 12.2 Prior mềm theo chuỗi

Theo từng sample thật có N_i tokens, T_i frames:

```text
x_t   = (t + 0.5) / T_i
mu_n  = (n + 0.5) / N_i
prior[n,t] = -beta * (x_t - mu_n)^2 / (2*sigma^2)
logits[n,t] = S[n,t]/tau + prior[n,t]
```

Điểm xuất phát: `tau=0.1`, `sigma=0.2` trong trục thời gian [0,1], `beta=0.1`. P1 đặt beta=0; P2 có prior. Đây là **vị trí dự đoán thô đều theo số token**, không phải nhãn timestamp hay giả định mọi token có duration bằng nhau.

Tính prior bằng T_i/N_i thật, không dùng padded Tmax/Nmax. Prior cộng trong logit như trên, không chia thêm prior cho tau một lần nữa. Kiểm tra các đoạn silence đầu/cuối, từ lặp, câu dài và tốc độ nói khác nhau. Có thể dùng sigma rộng/beta nhỏ hơn theo validation; không tự áp hard diagonal window.

### 12.3 Ví dụ số

Một token có cosine `[0.80,0.90,0.85,0.95]` với 4 frame. Nếu chỉ softmax/tau=0.1, frame cuối được ưu tiên dù có thể nằm sai vùng thời gian. Dùng prior multiplicative tương đương `[0.8,1.0,0.8,0.01]`:

```python
import math

scores = [0.80, 0.90, 0.85, 0.95]
temporal_weights = [0.8, 1.0, 0.8, 0.01]
logits = [s / 0.1 + math.log(p) for s, p in zip(scores, temporal_weights)]
m = max(logits)
unnormalized = [math.exp(v - m) for v in logits]
attention = [v / sum(unnormalized) for v in unnormalized]
print(attention)  # khoảng [0.164, 0.557, 0.270, 0.009]
```

Prior trên chỉ minh họa cách hoạt động, không phải vector Gaussian được tính từ một sample thật. Weighted sum ưu tiên frame 2/3 gần nhau, nhưng bản thân softmax+prior không chứng minh contiguous/monotonic alignment.

### 12.4 Các phương án mở rộng

- Additive attention là ablation scorer: `v^T tanh(W_h H_t + W_e E_n + b)` với hidden vector rõ ràng; không trộn cùng nhiều thay đổi khác.
- Sigmoid gates cần normalize trước weighted sum nếu muốn giữ scale không phụ thuộc duration.
- Threshold/top-k chỉ sau khi soft version chạy tốt; cần fallback không có frame và phân tích gradient/support discontinuity.
- kNN theo similarity không tự có thứ tự thời gian; top-k có thể chọn các frame rời nhau.
- Monotonic path/forward-sum/CIF là mở rộng thuật toán riêng. Không gọi Gaussian prior là strict monotonic alignment.

### 12.5 Áp dụng MFGCN đúng phạm vi

MFGCN là speech emotion recognition; có text-conditioned weighted acoustic pooling. Cosine threshold 0,8 của paper dùng cho **graph edges giữa word nodes sau pooling**, không phải ngưỡng chọn audio frame. Temporal graph edges nối nodes liền kề, không bảo đảm frame attention liên tục. Chỉ lấy nguyên tắc pooling; không đưa cả GCN/classifier emotion vào ASR.

Nguồn: [MFGCN, Neurocomputing](https://doi.org/10.1016/j.neucom.2024.128646). Kiến trúc và prior ở đây là adaptation của nhóm, không tuyên bố tái hiện nguyên xi source paper.

## 13. Loss, masks và API model

### 13.1 Định nghĩa per-example loss

Cho từng sample i:

```text
l_ctc_i = CTCLoss(reduction='none')[i] / U_i
l_token_i = mean_content_tokens(1 - cosine(E_hat_i,n, E_i,n))
l_region_i = CE(region_logits_i, region_label_i)        # khi bật
l_i = l_ctc_i + lambda_token * l_token_i + lambda_region * l_region_i
```

Định nghĩa này tương ứng CTC mean chuẩn hóa target length rồi mean theo utterance; khác upstream có thể sum chia batch. Đảm bảo U_i>0. Chọn **mean token trong từng utterance, rồi mean utterance**, không để câu dài có trọng số lớn chỉ vì nhiều tokens. Công bố reduction; λ không tự chuyển được giữa các reduction.

Core train chỉ nhận samples có nhãn cần thiết. Nếu optional head có missing labels phải định nghĩa normalization riêng cho số sample có nhãn; không dùng công thức chung với denominator sai. Cách đơn giản là giữ đủ nhãn hoặc tách protocol, ghi coverage.

### 13.2 Masks và ổn định số

- Frame mask sau subsampling; token mask chỉ content thật, padding=false.
- Không softmax hàng mà mọi frame đều -inf. Assert mỗi sample có ít nhất 1 frame; padded token rows dùng logits hữu hạn rồi zero weights/output sau softmax.
- Softmax, cosine norm và loss tính FP32 khi cần; normalize có epsilon.
- Không detach H/C/A ngoài ý muốn. Có gradient về encoder, P_k/P_q/P_v; không có gradient teacher.
- Không lưu attention graph mỗi step để vẽ, chỉ detach diagnostics subset định trước.
- Raw loss, weighted loss, gradient norm và attention diagnostics phải log riêng.

### 13.3 Batch/API contract

```text
batch:
  sample_ids: list[str]
  feats: float32 [B,Fmax,80]
  feat_lengths: int64 [B]
  ctc_targets: int64 [sum U_i]
  target_lengths: int64 [B]
  teacher_states: float16/float32 [B,Nmax,768]   # chỉ khi train text branch
  teacher_lengths: int64 [B]
  region_ids: optional int64 [B]

model.forward(batch) -> dict(per_example_loss[B], components, detached_diagnostics)
model.encode(feats, lengths) -> H, output_lengths, valid_mask
model.transcribe(feats, lengths) -> token_hypotheses
```

Khi DDP phải gọi `ddp_model(batch)` qua forward đã wrap; không gọi `ddp_model.module.forward_train(...)` để bypass gradient reducer. Hàm inference không nhận transcript chuẩn/teacher/region labels.

## 14. Fine-tuning schedule và khởi tạo công bằng

### 14.1 Cấu hình khởi đầu, chưa tối ưu

| Thông số | Giá trị ban đầu |
|---|---|
| Optimizer | AdamW |
| Encoder LR | 1e-5 |
| CTC/projections/region LR | 1e-4 |
| Weight decay | 0.01; bias/norm no-decay nếu áp dụng chung |
| LR schedule | Linear warmup 5% tổng optimizer updates rồi linear decay |
| Clip norm | 5.0, sau unscale và accumulation normalization |
| Precision | FP16 AMP + GradScaler, loss nhạy số học FP32 |
| Budget ban đầu | Tối đa 20 epochs; valid WER mỗi epoch |
| Main protocol | Cố định số epochs/updates sau pilot, không đổi giữa variants |
| λ token | 0.1 khởi đầu; search nhỏ [0.01,0.1,0.3] nếu có budget |
| λ region optional | 0.1 khởi đầu, chọn riêng bằng valid |
| Encoder LayerDrop | 0.0 |
| Seeds | 42,123,2026 nếu chạy lặp |

Không bảo đảm 20 epochs là hội tụ hoặc vừa quota. Sau pilot tính số updates/giờ, chọn budget chung và ghi vào protocol trước main. Chọn best checkpoint bằng **validation WER**, tie-break CER rồi epoch sớm hơn. Có thể dùng early stopping patience 5 như một protocol khác, nhưng phải dùng cùng rule và báo actual exposure; core so sánh theo budget cố định dễ diễn giải hơn.

### 14.2 Fine-tune gì?

Core: pretrained encoder cập nhật từ đầu với LR nhỏ, task heads mới LR lớn hơn. PhoBERT frozen/cached, VinAI ngoài ASR trainer. Head-only warmup hoặc freeze lower blocks chỉ thêm nếu pilot có vấn đề; áp dụng cùng policy mọi variant và ghi phase transition.

Nếu thử ramp λ trong 500 successful optimizer updates đầu, công bố trước và giữ mọi text variant cùng schedule. Không vừa đổi LR/freeze/data/augmentation vừa bật alignment rồi quy toàn bộ cải thiện cho alignment.

### 14.3 Shared initialization và stochasticity

Mỗi seed tạo/lưu shared initialization của encoder+CTC; mọi B1/P1/P2 seed đó load cùng tensors. Khởi tạo thêm projections không được làm RNG tạo CTC khác. Lưu hashes encoder/head init.

Sampler order chung giữa variants; nếu cần so sánh chặt, augmentation seed theo `(seed,epoch,sample_id,occurrence)` thay vì phụ thuộc worker/rank/RNG bị forward nhánh phụ tiêu thụ. DDP có thể không bitwise deterministic; lưu deterministic flags và giới hạn tái lập. CTC/cuDNN backend tùy version có thể ảnh hưởng tính xác định.

Phân biệt seed khởi tạo model (chung giữa ranks) với RNG training theo rank và seed sampler toàn cục. Sau khi load/init chung có thể đặt training RNG theo `seed + rank` để dropout không bị đồng nhất ngoài ý muốn; lưu RNG từng rank để resume. Không dùng rank seed khác để dựng hai model init khác nhau rồi bỏ qua kiểm tra DDP sync.

## 15. Hiểu đúng T4 ×2, batch và bộ nhớ

### 15.1 Sơ đồ thiết bị

```text
rank 0 → cuda:0 → encoder + CTC + active auxiliary heads + optimizer
rank 1 → cuda:1 → encoder + CTC + active auxiliary heads + optimizer
                      ↕ đồng bộ gradient
```

Mỗi T4 phải chứa được **một bản model và microbatch của rank đó**. VRAM không tự gộp. Kiểm tra `torch.cuda.get_device_properties` và nvidia-smi thực tế; không ghi cố định tổng 32 GB usable.

Main dùng DDP, không `nn.DataParallel`, không `device_map='auto'` trong ASR training. Không đặt encoder GPU0 và PhoBERT GPU1 để gọi đó là DDP; PhoBERT đã cache. FSDP/ZeRO/model parallel không cần cho bản đầu nếu model vừa một T4. DDP có communication overhead, không hứa nhanh gấp đôi. [DDP overview](https://docs.pytorch.org/docs/2.9/notes/ddp.html).

### 15.2 Batch đề xuất

Khởi đầu **microbatch 2 utterances/GPU, accumulation 8, world size 2**:

`effective_batch = 2 × 8 × 2 = 32 utterances/update` nếu đủ batch và mọi microbatch có đúng 2 mẫu.

Một GPU đối chứng có microbatch 2, accumulation 16 để cùng effective batch 32. Nếu microbatch1 trên mỗi T4 thì accumulation16. Không tự tăng LR chỉ vì dùng hai GPU khi effective batch không đổi.

Bucket duration; khởi đầu cap `B_rank × F_max <= 6.400 input frames` và max 2 samples/rank. Đây là **pilot limit**, không bảo đảm vừa VRAM. Attention còn phụ thuộc độ dài bình phương, context và implementation; phải profile câu dài nhất/đuôi phân phối.

Đây là giới hạn batch chứ không phải cắt audio. Sample vượt cap phải đi qua singleton batch được kiểm tra hoặc rule dữ liệu khai báo; không âm thầm drop câu dài. Hai rank có thể có số mẫu khác trong một step; mục 16 chuẩn hóa loss đúng cho trường hợp này.

### 15.3 Chi phí alignment

Ví dụ B_rank=2,N=100,T=400: A có 80.000 phần tử FP32, khoảng 0,32 MB riêng tensor. Đây không phải toàn bộ memory: còn Q/K/values, gradient, encoder activations và optimizer. Additive scorer nếu materialize `[B,N,T,256]` cần khoảng 81,92 MB chỉ intermediate FP32, nên cosine là core tiết kiệm hơn ở cách hiện thực này.

Không suy toàn model chỉ tốn vài MB từ kích thước A. Báo parameter count thực từ code, peak allocated/reserved per rank và measured throughput.

## 16. DDP sampler và gradient accumulation

### 16.1 Sampler duration-aware toàn cục

Agent phải xây một lịch batch **chung** từ seed+epoch+manifest, rồi shard cho ranks:

1. Shuffle/bucket bằng độ dài, không dùng transcript/test metric làm criterion.
2. Tạo local microbatches phù hợp caps; ghép thành groups mỗi group có đúng R local batches, rank r lấy batch r.
3. Mọi rank có cùng số microsteps và cùng accumulation boundaries, mỗi local batch không rỗng.
4. Nhóm cuối nếu thiếu rank dùng policy được công bố: drop rất ít mẫu hoặc repeat/pad với IDs/weights rõ ràng. Bản đầu có thể drop nhóm cuối train, shuffle qua epochs và log số mẫu/giờ bị bỏ; không áp policy đó cho valid/test.
5. `set_epoch(epoch)`/rebuild plan theo seed; lưu plan hash, sample counts, duplicates/drop và `__len__` nhất quán.

Nếu dùng `DistributedSampler` chuẩn, hiểu rằng padding có thể lặp sample hoặc drop theo flag; tránh để duration batching riêng từng rank tạo số step khác nhau. **Không làm hai worker cùng đọc toàn bộ stream**. Validation dùng sampler không padding và không bỏ mẫu, xem mục 18. [PyTorch DataLoader/samplers](https://docs.pytorch.org/docs/2.9/data.html).

Core fine-tune dùng map-style local dataset, không HTTP streaming. DataLoader bắt đầu 1 worker/rank, `pin_memory=True`, prefetch thấp; CPU/RAM Kaggle có giới hạn. Có thể tăng sau đo throughput, không đặt 8 workers mỗi rank theo thói quen.

### 16.2 Công thức chuẩn hóa đúng khi batch không đều

Mục tiêu mỗi optimizer update là mean các **per-example total losses** trong accumulation window toàn cục:

`L_global = sum_{rank,microbatch,sample}(l_i) / W`, với W là tổng sample thực.

Một implementation tổng quát, do nhóm đề xuất:

- Chọn hằng số scale D>0 (khởi đầu D=32, không phải world size).
- Mỗi microbatch backward `sum(l_i)/D`.
- Dùng `no_sync()` cho các microsteps chưa cuối window; forward cũng phải nằm trong context đó.
- Microstep cuối đồng bộ DDP, nên gradient đã được average theo world size R.
- All-reduce local sample counts để có W; sau AMP unscale, nhân mọi gradient với **R×D/W**.
- Clip norm rồi optimizer step. W phải tính đúng window cuối, kể cả không đủ 8 microsteps.

Chứng minh: gradient sau DDP là `sum_global(grad l_i)/(R×D)`; nhân `R×D/W` cho mean cần có. Không all-reduce gradients thủ công lần nữa. Loss đang log không dùng normalization tạm D để so sánh; log sums/counts thật.

Ví dụ R=2, mỗi rank 8 microsteps×2 samples → W=32; với D=32 thì correction=2. Nếu window cuối chỉ 3 microsteps/rank thì W=12, correction=64/12. Nếu rank batch sizes khác nhau, dùng count thật, không mean-of-means.

Đây là lựa chọn trainer của nhóm; phải có test tương đương full-batch FP32 trước khi dùng AMP. Cách chuẩn đơn giản `loss_mean/accumulation_steps` chỉ đúng khi microbatch sizes đều và xử lý partial window đúng.

### 16.3 Các invariant bắt buộc

- Tất cả ranks cùng số forward/backward và cùng `no_sync` boundaries.
- Không skip riêng một rank vì audio lỗi/OOM/invalid label. Pre-audit; runtime lỗi bất ngờ thì dừng job đồng bộ, lưu nguyên nhân và khởi chạy lại cấu hình đã sửa.
- Một rank không có teacher cache của sample cần alignment là lỗi dữ liệu, không âm thầm tắt nhánh.
- Không thay requires_grad/optimizer groups giữa chừng sau wrap DDP mà chưa dựng lại các thành phần đúng cách.
- Chỉ instantiate active heads; không dùng `find_unused_parameters=True` để che module thừa/LayerDrop chưa xử lý. Nếu thuật toán thật có unused params, phải mô tả và test riêng.

## 17. DDP training loop, AMP và lỗi phân tán

### 17.1 Khởi tạo

Launch bằng `torchrun --standalone --nnodes=1 --nproc_per_node=2 -m vi_asr train ...`. Chương trình đọc `LOCAL_RANK/RANK/WORLD_SIZE`, gọi `torch.cuda.set_device(local_rank)`, init process group NCCL với timeout đủ cho validation/checkpoint, rồi dựng model trên đúng device.

DDP wrap `device_ids=[local_rank]`, core `find_unused_parameters=False`, `gradient_as_bucket_view=True` nếu version hỗ trợ. Không bật static_graph chỉ vì muốn nhanh; kiểm tra graph trước. Model không có trainable nhánh bị bỏ qua và LayerDrop=0. Constructor tất cả ranks giống nhau. [torchrun](https://docs.pytorch.org/docs/2.9/elastic/run.html), [DDP API](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html).

### 17.2 Thứ tự trainer cần hiện thực

```text
seed + environment + source/config/hash checks
→ init NCCL/device → load shared encoder/CTC init + active heads
→ optimizer/scheduler/scaler → wrap DDP → resume states nếu có
→ tạo epoch batch plan chung
→ từng accumulation window:
    zero_grad(set_to_none=True)
    với từng microbatch của window:
      forward trong autocast và sync/no_sync context phù hợp
      CTC/token loss nhạy số học tính FP32
      kiểm tra finite theo policy đồng bộ
      scaler.scale(sum_per_example_loss / D).backward()
    all-reduce sample count W
    scaler.unscale_(optimizer)
    nhân gradients R*D/W
    clip_grad_norm
    scaler.step(optimizer)
    scaler.update()
    scheduler.step() chỉ nếu optimizer update thành công
    log sums/counts/time/memory
→ validation audio-only → best/last checkpoints
→ process_group cleanup trong finally
```

Đây là pseudocode, không phải trainer đầy đủ có thể copy chạy ngay. Agent phải xử lý overflow/nonfinite, actual sampler windows và API version trước khi gọi `train` hoàn tất.

### 17.3 AMP overflow và scheduler

Mọi rank dùng cùng scale và success/skip decision. Với gradients đã sync và loss finite, overflow phải được kiểm tra/đồng bộ nhất quán; thêm assertions scale/step giữa ranks trong debug. Không để rank0 update trong khi rank1 skip.

`scaler.step` có thể skip optimizer update; scheduler không được bước khi skip. Agent có thể dùng scale trước/sau theo hành vi GradScaler đã test (scale giảm báo overflow), hoặc tín hiệu supported phù hợp phiên bản. Không xem return của `optimizer.step()` thường là None để suy skip/success. Log `successful_updates`, `consumed_microsteps`, overflow count và LR.

Nếu loss nonfinite trước backward, tất cả rank cùng abort window hoặc dừng run theo policy đã chốt; không một rank `continue`. Debug finite-check có thể all-reduce flag cùng các vị trí trên mọi rank. Không chỉnh loss thành zero để che lỗi CTC rồi báo run ổn.

### 17.4 NCCL và DataLoader

Smoke all-reduce tensor nhỏ trước full model. Nếu NCCL lỗi, ghi topology/driver/version/log; thử cấu hình được tài liệu hỗ trợ theo lỗi thực tế. Không mặc định đặt `NCCL_P2P_DISABLE=1` hay `NCCL_IB_DISABLE=1` để coi mọi lỗi đã giải quyết.

DataLoader multiprocessing dùng phương thức tương thích NCCL (ưu tiên spawn nếu worker được dùng), top-level collate/worker functions có thể pickle, main có guard thích hợp. Khi debug giảm `num_workers=0` để thấy lỗi thật. Main vẫn phải profile CPU starvation trước khi tăng workers.

## 18. Validation phân tán và checkpoint/resume

### 18.1 Validation không lặp hoặc mất mẫu

Đề xuất: shard IDs theo `indices[rank::world_size]`, không padding. Mỗi rank decode bằng **unwrapped eval model** (`ddp_model.module`) không có DDP forward collectives; số batches có thể khác, chỉ all-reduce counters cuối cùng. Không gọi DDP wrapper forward với số lượt khác nhau nếu còn buffer sync.

Mỗi rank xuất `predictions.valid.rank{r}.jsonl`; gather/merge theo sample_id, assert mỗi reference đúng một hypothesis và không thiếu. All-reduce tổng S/D/I/N; WER toàn corpus là `(ΣS+ΣD+ΣI)/ΣN`, không mean WER của hai ranks hoặc mean WER từng câu. CER tương tự.

Nếu chọn rank0-only eval, dùng unwrapped model, ranks khác chờ có timeout đủ lớn và broadcast metric/best/stop decision. Không dùng hai cách lẫn lộn. Eval `inference_mode`, dropout tắt, augmentation/dither eval deterministic; reference chỉ phục vụ metric ngoài model.

### 18.2 Nội dung checkpoint

Checkpoint training gồm model active branches, optimizer, scheduler, scaler, epoch, successful updates, consumed microsteps, best metric, config/source/manifest/tokenizer/cache hashes, sampler plan/version, RNG từng rank (Python/NumPy/torch CPU/CUDA), world size và precision policy.

Chỉ rank0 ghi shared file, atomic temp→rename; collect per-rank RNG đúng đồng bộ, không bắt rank0 giả RNG rank1. Lưu `model.module.state_dict()` để inference không vướng prefix `module.`. Checkpoints `best.pt`, `last.pt`, periodic retention có giới hạn để tránh hết disk.

### 18.3 Resume trung thực

Core bảo đảm **resume ở epoch boundary**. Trước hết hoàn thành epoch, validation và last checkpoint; khởi chạy lại từ epoch tiếp theo với optimizer/scheduler/scaler/RNG/sampler đúng. Job chỉ save model weights không phải full resume.

Nếu cần periodic checkpoint mid-epoch để tránh mất cả epoch, agent phải lưu cursor **batch đã tiêu thụ**, accumulation boundary, sampled plan và augmentation state. DataLoader prefetch cursor không phải consumed cursor. Có thể dùng stateless sample augmentation và deterministic plan replay, nhưng phải test liên tục-vs-resume. Nếu chưa hoàn thành, ghi checkpoint mid-epoch là restart gần đúng, không tuyên bố bitwise/exact resume.

Không đổi world size 2→1 hoặc tokenizer/manifest/context giữa run rồi coi là tiếp tục tương đương; fail hash validation hoặc ghi thành run mới. Trước phiên Kaggle kết thúc, lưu output/version và xác minh checkpoint có thể đọc lại; không xem `/kaggle/working` là ổ lưu bền vĩnh viễn.

## 19. Cấu hình YAML chuẩn

### 19.1 Quy tắc config

Typed schema; reject unknown keys và invalid combinations. Merge `base → variant → hardware profile → paths → CLI overrides`, xuất `resolved_config.yaml` và hash trước khi load model. Hardware profile không được đổi backbone/loss/data; paths chỉ override đường dẫn. Relative paths trong base resolve từ `paths.project_root`, data paths từ `paths.data_root`; không phụ thuộc cwd tình cờ.

Các YAML dưới đây là **schema agent cần implement**, không tương thích sẵn với run.sh upstream. Các giá trị đòi hỏi pilot/revision runtime phải được xác nhận trước main.

### 19.2 `configs/base.yaml`

```yaml
schema_version: 2
experiment:
  name: vipvl_ctc
  seed: 42
  phase: development
data:
  hf_id: nguyendv02/ViMD_Dataset
  revision: 3a5b30157034e7eadd5c75fae1a820c6f9383398
  split_mapping: audited
  local_audio: true
  allow_random_audio_crop: false
  require_hash_validation: true
frontend:
  sample_rate: 16000
  num_mel_bins: 80
  frame_length_ms: 25.0
  frame_shift_ms: 10.0
  window_type: povey
  fbank_waveform_multiplier: 32768.0
  energy_floor: 0.0
  train_dither: 1.0
  eval_dither: 0.0
  cmvn_source: checkpoint
  cmvn_apply_count: 1
augmentation:
  specaugment: true
  placement: after_cmvn_before_subsampling
  freq_masks: 2
  freq_max_bins: 8
  time_masks: 2
  time_max_frames: 30
  max_time_fraction_per_mask: 0.1
  speed_perturb: false
tokenizer:
  type: sentencepiece_bpe
  vocab_size_target: 1024
  train_split_only: true
  character_coverage: 1.0
  pad_id: 0
  unk_id: 1
  bos_id: -1
  eos_id: -1
  ctc_blank_id: 0
model:
  encoder_id: khanhld/vip-vl-base-vie
  encoder_revision: 4b259337d96450d57403acdee82572c587151e22
  encoder_config_from_checkpoint: true
  encoder_layerdrop_override: 0.0
  activation_checkpointing: true
  require_checkpoint_patch_test: true
  context_policy: full_utterance
  ctc_head_dropout: 0.0
  ctc_vocab_from_tokenizer: true
teacher:
  id: vinai/phobert-base
  revision: 01daacda68afe13d83023d16ec647239e344a1e6
  frozen: true
  mode: cached_content_tokens
  hidden_size: 768
  output_layer: last_hidden_state
  window_overflow: stitch_content_tokens
alignment:
  enabled: false
  scorer: cosine
  score_dim: 256
  temperature: 0.1
  values: audio_hidden
  prediction_dim: 768
  target: frozen_original_teacher_states
  weighting: softmax_over_valid_frames
  temporal:
    enabled: false
    center: uniform_content_token_position
    sigma_normalized: 0.2
    strength: 0.1
dialect:
  enabled: false
  classes: [North, Central, South]
  dropout: 0.1
loss:
  ctc: per_example_divide_target_length
  zero_infinity: false
  token: cosine_mean_tokens_then_utterances
  lambda_token: 0.0
  lambda_region: 0.0
  lambda_ramp_updates: 0
training:
  optimizer: adamw
  encoder_lr: 0.00001
  heads_lr: 0.0001
  weight_decay: 0.01
  warmup_fraction: 0.05
  decay: linear
  epochs: 20
  fixed_budget: true
  grad_clip_norm: 5.0
  loss_backward_divisor: 32.0
  validation_every_epochs: 1
  select_by: valid_wer
  tie_break: [valid_cer, earliest_epoch]
  resume_contract: epoch_boundary
decoding:
  type: ctc_greedy
  external_lm: false
  reference_conditioning: false
mt:
  id: vinai/vinai-translate-vi2en-v2
  revision: ae7baa85da07dbe8e23ac26a9f5ef560c17e2138
  frozen: true
  source_language: vi_VN
  target_language: en_XX
  beams: 5
  do_sample: false
  max_new_tokens: 256
paths:
  project_root: .
  data_root: artifacts/prepared
  teacher_root: artifacts/teacher_cache
  models_root: artifacts/models
  output_root: runs
```

`activation_checkpointing:true` là yêu cầu fail-fast nếu patch chưa được kiểm tra; không phải khẳng định source gốc đã hỗ trợ. Nếu pilot chưa cần checkpoint có thể đặt false và đo; config main phải phản ánh code thực tế. Context `full_utterance` phải được adapter ánh xạ sang arguments đúng source bằng test, không đoán từ giá trị chunk_size mặc định.

### 19.3 Variants

`configs/variants/b1.yaml`:

```yaml
experiment: {name: b1_ctc}
alignment: {enabled: false}
dialect: {enabled: false}
loss: {lambda_token: 0.0, lambda_region: 0.0}
```

`configs/variants/p1.yaml`:

```yaml
experiment: {name: p1_token_pooling}
alignment:
  enabled: true
  temporal: {enabled: false}
dialect: {enabled: false}
loss: {lambda_token: 0.1, lambda_region: 0.0}
```

`configs/variants/p2.yaml`:

```yaml
experiment: {name: p2_token_temporal}
alignment:
  enabled: true
  temporal: {enabled: true}
dialect: {enabled: false}
loss: {lambda_token: 0.1, lambda_region: 0.0}
```

Optional `d1` = B1+region only, λ_region>0, λ_token=0; `p2d` = P2+region. Định nghĩa đủ YAML và không tự gộp chung với kết quả P2. Additive scorer là variant riêng nếu có budget.

### 19.4 `configs/profiles/t4x2.yaml`

```yaml
runtime:
  device: cuda
  expected_world_size: 2
  distributed_backend: nccl
  precision: fp16_amp
  grad_scaler: true
  num_workers_per_rank: 1
  dataloader_start_method: spawn
  pin_memory: true
  persistent_workers: false
  prefetch_factor: 2
  ddp_find_unused_parameters: false
  ddp_gradient_as_bucket_view: true
  ddp_static_graph: false
  process_group_timeout_minutes: 45
batching:
  max_samples_per_rank: 2
  max_padded_fbank_frames_per_rank: 6400
  accumulation_steps: 8
  duration_bucketing: true
  training_tail_policy: drop_incomplete_rank_group_logged
  accumulation_normalization: global_sample_count
  validation_shard: no_padding_no_drop
```

Single-T4 profile giữ effective batch tương đương: world_size1, max_samples2, accumulation16; tương tự mọi thứ khác nếu đo scaling. `cpu_smoke` tắt AMP/DDP/checkpoint patch demand khi không dùng path đó. Nếu workers=0 thì không truyền prefetch/persistent_worker arguments không phù hợp.

### 19.5 Kaggle paths

```yaml
paths:
  project_root: /kaggle/working/vi_asr_project
  data_root: /kaggle/input/vi-asr-prepared/prepared
  teacher_root: /kaggle/input/vi-asr-teacher/teacher_cache
  models_root: /kaggle/input/vi-asr-models/models
  output_root: /kaggle/working/runs
```

Các slug trên là ví dụ artifact do nhóm tự chuẩn bị và attach; agent/user thay bằng tên thật. Không tuyên bố các Kaggle Dataset đó đang tồn tại. Model loaders phải dùng local verified snapshots khi offline; nếu online download thì ghi đường cache/hash rồi chuyển sang local paths.

## 20. Baseline, ablation và ngân sách

### 20.1 Bảng bắt buộc

| Run | Encoder/ASR | Token pooling | Temporal prior | Region loss | Vai trò |
|---|---|---|---|---|---|
| B_ref | Whisper-small zero-shot | Không | Không | Không | Baseline theo đề bài |
| B1 | ViP-VL+CTC fine-tune | Không | Không | Không | Baseline kiến trúc trực tiếp |
| P1 | Cùng B1 | Có | Không | Không | Đóng góp text-guided token pooling |
| P2 | Cùng B1 | Có | Có | Không | Đóng góp prior mềm |
| D1, optional | Cùng B1 | Không | Không | Có | Region-only |
| P2D, optional | Cùng B1 | Có | Có | Có | Kết hợp region sau core |

Whisper dùng `task=transcribe`, language Vietnamese; không dùng `task=translate` thay cascade vì sẽ không tạo đúng baseline transcript vi→VinAI. Ghi backend/version/model checksum, decoding params, segmentation và preprocessing. [Whisper official](https://github.com/openai/whisper).

B_ref/P1/P2/B1 dùng cùng MT và normalization/segmentation eval. So sánh Whisper với model fine-tune phải ghi rõ khác nguồn pretraining/adaptation; causal claim về alignment chỉ dựa B1/P1/P2 cùng protocol.

### 20.2 Fair comparison

Giữ train/valid/test, init encoder/CTC, tokenizer, augmentation, batch policy, exposure, decoder, frontend/context giống nhau. Nếu P2 OOM buộc giảm batch thì điều chỉnh accumulation giữ effective batch và áp dụng cùng profile cho các main variants; không âm thầm drop dài hơn riêng P2.

Chọn λ/tau/prior bằng valid trong search nhỏ có ngân sách giống nhau; ghi đầy đủ trials kể cả thất bại. Không chọn seed tốt nhất để báo. Không chọn phương pháp dựa test.

### 20.3 Số runs thực tế

- Khởi đầu: B1/P1/P2 mỗi loại 1 seed + B_ref inference. Đây là 3 training runs, chưa đủ ước lượng biến động seed mạnh.
- Củng cố với ngân sách vừa: chọn final variant bằng valid; chạy B1 và final trên 3 seeds. Nếu 3 runs đầu đã có seed42, thêm 4 runs → tổng **7 main training runs** (variant còn lại một seed). Ghi rõ số lần lặp khác nhau.
- Đầy đủ core: B1/P1/P2 ×3 seeds = **9 training runs** + B_ref inference.
- Tuning/pilot/controls và region variants tính thêm, không giấu vào “một lần train”.

Nếu ngân sách không đủ, giảm phạm vi trước, giữ báo cáo trung thực. Không gọi ba checkpoint từ một trajectory là ba seeds độc lập.

### 20.4 Controls cho nhánh text

Giữ reference CTC đúng, chỉ shuffle teacher targets giữa các sample để kiểm tra lợi ích thông tin text thật; không shuffle cả CTC labels. Thêm prior-only scorer/constant values diagnostic khi cần. Text-conditioned attention có thể đi tắt qua cách chọn weights; frozen teacher không tự loại shortcut.

Vẽ attention trước/sau prior trên dev IDs cố định, gồm từ lặp/giọng vùng miền/câu dài; không cherry-pick heatmap đẹp. Attention entropy, center/coverage/collision là diagnostics, chưa chứng minh alignment đúng. Muốn claim boundary accuracy phải có tập gán thời gian độc lập và metric tương ứng.

## 21. Các cổng kiểm tra trước full training

| Gate | Kiểm tra bắt buộc | Bằng chứng |
|---|---|---|
| G0 | Source/revisions/license/import/config schema | source audit, import log |
| G1 | Data integrity/split/manifest/tokenizer/cache | audit JSON, hashes, coverage |
| G2 | Encoder load/frontend parity/CTC shapes | load report, numeric assertions |
| G3 | Alignment mask/loss/gradients/teacher freeze | tests chạy được |
| G4 | Tiny overfit B1/P1/P2 trên 16–32 train samples | loss/WER curves; không dùng test |
| G5 | DDP sampler + FP32 gradient equivalence | 1 process vs 2 ranks test |
| G6 | FP16/checkpoint patch/OOM edge cases/resume | GPU pilot logs |
| G7 | Full-corpus validation aggregation/inference export | exact ID coverage/parity |
| G8 | Freeze protocol rồi main runs | signed-off protocol hash/date |

### 21.1 Tests có ý nghĩa

1. Fbank parity waveform và scale; CMVN đúng một lần.
2. CTC collapse: `[a,a,blank,a] → aa`; check repeated-label minimum lengths và blank targets.
3. Alignment A rows sum=1 cho content; padding frames weight0; padded tokens không ảnh hưởng loss.
4. Padding invariance: cùng input thêm padding phải cho loss/output tương đương trong tolerance.
5. Constant-values test: mọi H_t giống nhau thì C không được mang thông tin token chỉ từ weighted sum, dù weights khác.
6. Gradient: teacher không grad; encoder/scorer/projection có grad khi bật; no teacher residual vào values/CTC.
7. Checkpoint patch parity output/gradient, dropout0; đo memory reduction thực.
8. DDP gradient equivalence dùng toy deterministic model, batch không đều giữa ranks và partial accumulation; so mean full batch FP32 trước clipping/optimizer.
9. Sampler coverage/no-unexpected-duplicates/equal microsteps; valid/test có mỗi ID đúng một lần.
10. Resume test cùng epoch schedule/RNG/optimizer; không chỉ so model load được.
11. Audio-only contract: thay/xóa reference metadata không đổi hypothesis.
12. Export→reload CTC output parity; model bundle không cần teacher để chạy.

Tiny overfit chỉ xác nhận đường học hoạt động, không chứng minh generalization. GPU tensor giả không thay checkpoint/data thật. Nếu gate thất bại, sửa và chạy lại gate liên quan; không full-train để tìm lỗi shape/mask cơ bản.

## 22. Chuẩn bị ZIP và dữ liệu trên Kaggle

### 22.1 Code ZIP

Agent tạo `vi_asr_project.zip` chứa duy nhất root `vi_asr_project/`, có README/CLI/config/vendor/tests/licenses. Kèm SHA256, file inventory và source commit; notebook không chứa secrets/token. Chuẩn bị runtime lock đã test hoặc ghi rõ bootstrap chưa kiểm chứng.

Không bundle `.git`, env, downloaded weights/audio, cache, runs hoặc user-specific absolute paths. Package script kiểm tra required files/import và config trước khi ZIP. ZIP là source vận chuyển, không thay Git history phục vụ chấm đóng góp.

### 22.2 Data/model artifacts riêng

Preparation notebook có thể chạy phiên CPU để materialize audio/tokenizer/audit. Teacher cache chạy phiên GPU riêng rồi unload teacher. Hai T4 của main train dành cho DDP ASR, không giữ VinAI/PhoBERT resident.

Prepared artifact chứa manifests + relative audio paths + tokenizer/audit/hashes; teacher artifact chứa token cache; model artifact chứa pinned snapshots. Chỉ lưu/chia sẻ theo quyền dữ liệu, mặc định giữ private. `/kaggle/input` read-only, output viết `/kaggle/working`.

Nếu mạng bật, có thể tải từ HF ngay preparation, lưu xong kiểm tra hashes. Nếu mạng tắt, attach artifacts và wheelhouse Linux tương thích image; không mang wheel Windows lên cài Kaggle. Không clone/download lớn ở mỗi epoch hoặc mỗi DDP rank.

## 23. Notebook Kaggle theo từng cell

**Trạng thái:** đây là nội dung cell mẫu và CLI contract. Agent cần tạo notebooks thực, triển khai subcommands, chạy smoke và sửa theo môi trường. Không coi việc copy các cell này là đã có project runnable.

### Cell 0 — Chọn môi trường

Chọn Accelerator T4 ×2 nếu tài khoản có; kiểm tra UI quota/session/output storage hiện tại. Không hard-code số giờ miễn phí hoặc giới hạn phiên. Attach code ZIP/prepared/cache/models đúng slug. Bật Internet chỉ khi cần tải assets đã pin.

### Cell 1 — Preflight phần cứng

```python
import os
import sys
import shutil
import subprocess
import torch
import torchaudio

print(sys.version)
print("torch", torch.__version__, "torchaudio", torchaudio.__version__)
print("CUDA build", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "Notebook chưa có CUDA"
assert torch.cuda.device_count() == 2, "Cần kiểm tra lại T4 x2 hoặc chọn profile khác"
for idx in range(2):
    prop = torch.cuda.get_device_properties(idx)
    print(idx, prop.name, prop.total_memory / 2**30, "GiB", prop.major, prop.minor)
print("disk", shutil.disk_usage("/kaggle/working"))
subprocess.run(["nvidia-smi"], check=True)
```

Preflight chỉ in thông tin, không giữ model trên notebook GPU. Trainer chạy subprocess torchrun riêng; tránh để kernel cha chiếm VRAM trước launch.

### Cell 2 — Giải nén và xác định root

```python
from pathlib import Path
import zipfile

# Thay slug bằng artifact code thật đã attach.
zip_path = Path("/kaggle/input/vi-asr-code/vi_asr_project.zip")
workspace = Path("/kaggle/working").resolve()
with zipfile.ZipFile(zip_path) as bundle:
    for member in bundle.infolist():
        destination = (workspace / member.filename).resolve()
        if not destination.is_relative_to(workspace):
            raise ValueError("ZIP member nằm ngoài workspace")
    bundle.extractall(workspace)
project_root = workspace / "vi_asr_project"
assert (project_root / "pyproject.toml").exists()
print(project_root)
```

Chạy lại cell chỉ khi biết ZIP version; verify bundle hash và không âm thầm overwrite source đã sửa trong phiên. Package script phải tạo archive thường, không symlink lạ.

### Cell 3 — Cài project và kiểm tra dependencies

```python
import subprocess
import sys

# Extra lock do agent resolve/test trên image; không cài lại torch CUDA mù quáng.
subprocess.run([
    sys.executable, "-m", "pip", "install", "-r",
    str(project_root / "requirements-extra-kaggle.lock.txt"),
], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-deps", "-e",
    str(project_root / "third_party" / "chunkformer"),
], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(project_root),
], check=True)
subprocess.run([sys.executable, "-m", "vi_asr", "--help"], check=True)
```

`requirements-extra-kaggle.lock.txt` cần được tạo bên cạnh full environment lock; loại torch/torchaudio khỏi phần cài lại và có constraints/verification để dependencies khác không kéo torch sai. `--no-deps` chỉ dùng vì dependencies đã được audit/cài ở bước trước, không dùng để bỏ qua package thiếu. Nếu runtime thay đổi buộc upgrade torch/torchaudio thì làm riêng, restart kernel và chạy lại preflight/smoke.

### Cell 4 — Paths/common arguments và audit artifacts

```python
common = [
    "--config", str(project_root / "configs/base.yaml"),
    "--paths", str(project_root / "configs/paths/kaggle.yaml"),
]
subprocess.run([
    sys.executable, "-m", "vi_asr", "audit", *common,
    "--report-dir", "/kaggle/working/preflight_audit",
], check=True, cwd=project_root)
```

Audit kiểm tra data/cache/tokenizer/model hashes và không decode test để chọn cấu hình. Nếu chưa chuẩn bị artifacts, chạy **notebook preparation riêng** theo thứ tự CLI sau, paths trỏ output working thay vì input read-only:

```bash
python -m vi_asr prepare-data --config configs/base.yaml --paths configs/paths/kaggle_prepare.yaml
python -m vi_asr audit --config configs/base.yaml --paths configs/paths/kaggle_prepare.yaml --report-dir artifacts/prepared/audit
python -m vi_asr build-tokenizer --config configs/base.yaml --paths configs/paths/kaggle_prepare.yaml
python -m vi_asr cache-teacher --config configs/base.yaml --paths configs/paths/kaggle_prepare.yaml --split train --device cuda:0
```

Agent phải tạo `kaggle_prepare.yaml` với output writable và triển khai đầy đủ các lệnh. Không chạy cả preprocessing lẫn main training trên cùng GPU session nếu có thể tách để tiết kiệm budget.

### Cell 5 — Một GPU smoke trước

```python
smoke_env = os.environ.copy()
smoke_env["CUDA_VISIBLE_DEVICES"] = "0"
subprocess.run([
    sys.executable, "-m", "vi_asr", "smoke", *common,
    "--variant", str(project_root / "configs/variants/p2.yaml"),
    "--profile", str(project_root / "configs/profiles/t4_single.yaml"),
    "--run-dir", "/kaggle/working/runs/smoke_p2",
], check=True, cwd=project_root, env=smoke_env)
```

Smoke dùng vài train samples + masks/gradients/encoder load, không main budget. Sau khi subprocess kết thúc mới launch hai GPU. Không giữ tensor CUDA của smoke trong kernel notebook.

### Cell 6 — Hai GPU pilot

```python
pilot_cmd = [
    "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=2",
    "-m", "vi_asr", "pilot", *common,
    "--variant", str(project_root / "configs/variants/p2.yaml"),
    "--profile", str(project_root / "configs/profiles/t4x2.yaml"),
    "--run-dir", "/kaggle/working/runs/pilot_p2",
    "--optimizer-updates", "100",
]
subprocess.run(pilot_cmd, check=True, cwd=project_root)
```

Pilot kiểm tra all-reduce, batch dài/teacher dài, FP16, normalization, sampler, validation, memory và save/reload. Warm-up rồi đo; 100 updates không thay full-corpus edge tests và tiny-overfit. Xong pilot chốt batch/budget/protocol; không lấy pilot checkpoint P2 làm init riêng main.

### Cell 7 — Main run

```python
train_cmd = [
    "torchrun", "--standalone", "--nnodes=1", "--nproc_per_node=2",
    "-m", "vi_asr", "train", *common,
    "--variant", str(project_root / "configs/variants/p2.yaml"),
    "--profile", str(project_root / "configs/profiles/t4x2.yaml"),
    "--seed", "42",
    "--run-dir", "/kaggle/working/runs/p2_seed42",
]
subprocess.run(train_cmd, check=True, cwd=project_root)
```

Đổi variant/run-dir cho B1/P1 và seed theo protocol. Chạy từng run để không tranh GPU. Dùng stdout trực tiếp hoặc ghi log song song có progress; không `capture_output=True` toàn bộ training dài để giữ log trong RAM. Main chỉ bắt đầu khi gates pass.

### Cell 8 — Resume phiên khác

Attach checkpoint archive từ phiên trước, verify hash; command main thêm:

```text
--resume /kaggle/input/vi-asr-run-checkpoints/p2_seed42/last.pt
```

Giữ run identity/config/world size, output directory mới có provenance nối với run trước. Loader phân biệt `--resume` full state và `--init-from` weights-only; không đặt hai cờ cùng nhau. Checkpoint periodic chưa có exact mid-epoch contract cần báo mode thực.

### Cell 9 — Evaluation, export và đóng gói kết quả

Đây là CLI cần agent implement; lệnh Bash có thể chạy trong notebook shell tại project root:

```bash
python -m vi_asr evaluate-asr --config configs/base.yaml --paths configs/paths/kaggle.yaml --checkpoint /kaggle/working/runs/p2_seed42/best.pt --split valid --run-dir /kaggle/working/eval/p2_valid
python -m vi_asr export-asr --checkpoint /kaggle/working/runs/p2_seed42/best.pt --output /kaggle/working/model_bundle
python -m vi_asr collect-results --runs-root /kaggle/working/runs --output /kaggle/working/results_archive
```

Test/cascade chỉ chạy sau protocol freeze theo mục 24–25. `evaluate-asr` mặc định single GPU nếu không được launch torchrun, ghi device và benchmark context; không tự suy inference benchmark hai GPU từ training profile. `export-asr` lấy resolved config trong checkpoint nên không bắt buộc `--config`; `collect-results` không cần model config.

Agent phải có `--help` cho mọi lệnh, lỗi đường dẫn rõ ràng, fail nếu artifact thiếu; không tạo file empty rồi báo xong. Các commands bổ sung `baseline-whisper`, `evaluate-cascade`, `benchmark`, `demo`, `verify-bundle` cần cùng quy ước paths/output và tài liệu sử dụng.

## 24. Metric, thống kê và lỗi ASR

### 24.1 Chọn model và khóa test

1. Trong development chỉ train và valid; dùng valid chọn λ/LR/checkpoint/context/normalization/decoder.
2. Chốt danh sách main variants, seeds, checkpoint selection, metrics và source/config hashes trong `docs/protocol.md`.
3. Khi đã freeze mới decode test cho bảng kết quả cuối. Không xem test mỗi epoch.
4. Báo mọi main variants đã định trước, không chỉ model thắng. Sửa lỗi code sau khi đã xem test cần version mới, ghi rõ lần đánh giá và hạn chế; không xóa lịch sử để giả test chưa được dùng.

Test integrity audit theo rule định trước khác với dùng test score để chọn phương pháp. Test references không đi vào forward nhận dạng, dù evaluation script cần đọc để tính metric.

### 24.2 ASR metrics

- Corpus WER = `(S+D+I)/N_reference_units`, báo riêng substitution/deletion/insertion.
- CER với policy Unicode/case/punctuation/space được công bố.
- Với tiếng Việt, whitespace tokens thường gần đơn vị âm tiết hơn từ đa âm tiết. Ghi rõ metric tokenization; nếu dùng word segmentation để đo word-level WER thì áp dụng cùng segmenter cho refs/hyps và báo riêng, không trộn hai định nghĩa vào một cột.
- Bảng theo North/Central/South, duration buckets, noise subset nếu có nhãn hợp lệ; có số mẫu/giờ/denominator mỗi nhóm.
- Overall corpus WER không phải average ba vùng. Macro-region WER là metric phụ, ghi riêng nếu dùng.
- Empty hypothesis/reference có rule rõ; không loại mẫu có WER cao sau decode. Prediction coverage phải 100% protocol set hoặc báo failure IDs.

Lưu `predictions.jsonl`: sample_id, raw/normalized reference, hypothesis, region, duration, S/D/I/N, checkpoint/config hash. Không chỉ lưu một số WER tổng.

### 24.3 Thống kê

Với nhiều seeds, báo mean±std của **scores từng independent run**, không bootstrap utterances rồi gọi đó là seed std. Chênh lệch WER báo cả absolute percentage points và relative reduction có denominator rõ.

Paired bootstrap cùng sample IDs giữa hai models để ước lượng khoảng tin cậy chênh lệch; nếu nhiều utterances cùng speaker và có speaker ID, cân nhắc bootstrap theo speaker để phản ánh phụ thuộc. Đây là thí nghiệm bổ sung, không thay việc chạy seeds. Lưu bootstrap seed/số lần resample/đơn vị sampling và code.

Ví dụ minh họa: WER 20% xuống 18% là giảm 2 điểm phần trăm hoặc giảm tương đối 10%; **không phải kết quả của dự án**.

### 24.4 Alignment diagnostics

Theo dõi masked token loss, cosine đúng/shuffled teacher, variance audio/token predictions, normalized attention entropy, expected center, spread, token collisions và before/after-prior heatmaps. Không gọi center tăng dần là bằng chứng toàn attention map strictly monotonic.

Nếu không có ground-truth boundaries, kết luận chính dựa WER/CER và controls, không tự gán “alignment accuracy”. Timestamp đoạn demo có thể từ VAD, không đồng nhất token timestamp.

## 25. VinAI, đánh giá cascade và dữ liệu tham chiếu

### 25.1 Module dịch cố định

Tải model/tokenizer cùng revision. Dùng raw Vietnamese text phù hợp MT; `src_lang="vi_VN"`, target `en_XX`, theo cách `generate` mà official example hỗ trợ. Baseline và variants cùng beam=5, `do_sample=False`, cùng segmentation/max length/normalization. [VinAI official usage](https://github.com/VinAIResearch/VinAI_Translate).

`max_new_tokens=256` là giới hạn khởi đầu, không có nghĩa mọi câu dịch đều ngắn hơn 256 token. Chia đoạn theo câu/độ dài tokenizer và theo dõi output chạm giới hạn; không silent truncation. Giữ mapping segment_id và thứ tự. Không đưa `học_sinh` từ preprocessing PhoBERT vào thay văn bản MT thông thường.

Transcript ASR rỗng phải có policy rõ: giữ English output rỗng hoặc trạng thái no-speech theo ứng dụng, vẫn giữ sample trong evaluation theo rule chung. Không gửi chuỗi rỗng để model sinh nội dung rồi coi là lời nói thật, không bỏ sample khỏi bảng metric.

Evaluation load VinAI sau khi trainer đã kết thúc/unload. Có thể dùng GPU0 chạy MT, GPU1 để trống hoặc chạy independent inference job được kiểm soát; không giữ hai bản full ASR training chiếm VRAM rồi thêm model dịch. Benchmark deployment có thể preload cả ASR/MT trên một device nếu profile thực tế cho phép; báo topology và memory thật.

### 25.2 Ba đường đánh giá

| Đường | Input MT | Mục đích |
|---|---|---|
| MT trên transcript chuẩn | Reference Vietnamese text | Mốc kiểm tra MT khi không có lỗi ASR |
| Cascade baseline | Transcript từ Whisper/B1 | Mốc hệ thống |
| Cascade proposed | Transcript từ P1/P2 | Tác động thay ASR dưới cùng MT |

Transcript chuẩn → MT là mốc chẩn đoán, không phải “upper bound” toán học bảo đảm cao nhất. Đôi khi lỗi ASR có thể tình cờ hợp reference English hơn; không suy đơn điệu từ WER sang BLEU.

### 25.3 Dữ liệu dịch và test contamination

ViMD ASR không được mặc định có English references. Muốn BLEU/chrF++ cascade cần tập `(audio, Vietnamese reference, English reference)` thật. Tạo tập nhỏ bằng human translation/review từ subset test đã chốt trước, hoặc dữ liệu nhóm tự thu có consent; lưu selection rule, IDs, regions/durations và annotation protocol.

Không lấy chính output VinAI làm reference rồi đánh giá VinAI; không gắn điểm BLEU cho dữ liệu chỉ có transcript Việt. Nếu dùng AI hỗ trợ dịch reference, ghi rõ provenance và có human review; báo giới hạn của reference.

VinAI official repo nêu vấn đề khoảng 10K training pairs mTet-v2 trùng PhoMT test. Vì chọn v2, **không dùng PhoMT test nguyên trạng để khẳng định đánh giá MT độc lập chưa thấy**. Dùng test riêng phù hợp cascade hoặc protocol đã kiểm soát contamination, ghi rõ nguồn pretrained có thể chưa biết đầy đủ. [Lưu ý evaluation của VinAI](https://github.com/VinAIResearch/VinAI_Translate#evaluation).

### 25.4 Metric dịch

Dùng corpus sacreBLEU và chrF++ (`word_order=2` theo API version), lưu signature/version/settings, cùng reference order và normalization. Không mean sentence BLEU để báo corpus BLEU. COMET tùy budget, ghi model identity; không cần để đáp ứng core. [sacreBLEU](https://github.com/mjpost/sacrebleu).

Luôn báo số samples/sentences của tập cascade, phân biệt full ViMD ASR test và subset có English reference. Có cùng subset cho mọi ASR variants; không chọn lại subset theo model thắng.

## 26. Demo upload/microphone và audio dài

### 26.1 Cấu trúc ứng dụng

Gradio/Streamlit: upload file hoặc record → segmenter → ASR service → MT service → bảng song song `segment_id/start/end/vi/en/status`. Load models một lần theo process; bounded queue để tránh GPU OOM khi nhiều yêu cầu.

File dài: VAD hoặc cửa sổ có overlap; giữ thời gian tuyệt đối. Nếu ghép overlap cần dedup có rule kiểm tra, không loại mọi từ trùng vì người nói có thể lặp thật. Không cắt audio tại một vị trí rồi dùng nhãn toàn câu để train.

Microphone phiên bản đầu: thu chunks → VAD endpoint → decode đoạn hoàn chỉnh → dịch đoạn/câu. Đây là **near-real-time theo đoạn**, không tự là true token streaming. Đo latency từ cuối lời nói đến transcript/bản dịch, tính cả VAD hangover và hàng đợi.

### 26.2 Context và streaming

Core training full utterance; demo VAD segments cũng dùng full context trong mỗi đoạn. Muốn streaming cache với chunk-size/right-context khác, phải kiểm tra train–inference context mismatch hoặc xây training context policy tương ứng và benchmark riêng.

Không đọc các con số streaming của upstream rồi áp vào model+UI của nhóm. Timestamps VAD là timestamps đoạn; muốn token/word timestamps cần thuật toán và đánh giá riêng. Nhánh PhoBERT train-only không cung cấp timestamps lúc inference.

### 26.3 Demo tests

- WAV/MP3/FLAC, mono/stereo, sample rate khác 16 kHz.
- Audio im lặng, quá ngắn, tiếng nói+nhiễu, Bắc/Trung/Nam, file dài.
- Ngắt/mở lại microphone, queue liên tục, không ghép nhầm thứ tự vi/en.
- Input lỗi có thông báo, output không mất segment, GPU memory ổn qua nhiều lượt.
- Demo suy luận không đòi reference transcript/cache teacher.

FastAPI/Docker/deploy là điểm cộng sau core; pin runtime và không nhét dataset/private audio vào image hoặc repo.

## 27. Profiling, OOM và ước lượng thời gian

### 27.1 Đo gì?

Mỗi rank: peak allocated/reserved, GPU utilization, step time sau warm-up, data loading time, audio seconds/update, pad ratio, T/N maxima, CPU/RAM/disk. Đồng bộ CUDA trước/sau các phép đo timing GPU cần chính xác.

Tách chuẩn bị data/cache, training, validation, decode, translation, packaging. Không báo runtime chỉ lấy forward một microbatch trong khi bỏ backward/all-reduce/validation.

Ước lượng time theo pilot:

```text
updates_per_epoch = số accumulation windows thật từ epoch plan
train_time ≈ epochs × updates_per_epoch × median_seconds_per_update
total_time ≈ train_time + validation + cache/data + evaluation + save/resume overhead
```

Ví dụ **giả định**: 15.000 train samples, effective batch32 → khoảng469 updates/epoch nếu counts đều; 20 epochs khoảng9.380 updates. Nếu đo được 1,5 giây/update thì phần train khoảng3,91 giờ; nếu5 giây/update thì13,03 giờ. Đây không phải tốc độ ViP-VL trên Kaggle và chưa gồm overhead; dùng measured plan/counts thật.

DDP scaling so sánh1 GPU với2 GPU cùng model/effective batch/data, warm-up và lượng work tương đương. Không so1 GPU batch16 với2 GPU batch32 rồi gọi mọi chênh lệch là tăng tốc phần cứng.

### 27.2 Ưu tiên khi OOM

1. Kiểm tra notebook parent/other processes và model PhoBERT/VinAI còn resident không.
2. Giảm microbatch2→1, tăng accumulation8→16 giữ effective batch khoảng32.
3. Kiểm tra activation checkpoint patch thực sự chạy; tăng bucketing giảm padding.
4. Profile câu dài, teacher window/N; hạn chế diagnostics materialization, không giữ loss tensors/graphs trong list.
5. Nếu full attention dài vẫn không vừa, thiết kế context/chunk policy mới dùng chung cho variants hoặc protocol dữ liệu có lý do; không cắt transcript tùy tiện.
6. LoRA/partial freeze/FSDP chỉ xét khi có nhu cầu đo được, không thêm trước khi sửa memory leak hoặc padding.

OOM một rank làm DDP job cần dừng đồng bộ; không bắt exception rồi bỏ batch riêng rank đó. Đổi profile sau OOM phải ghi revision, rerun phù hợp để giữ fair comparison. `empty_cache()` không giải phóng tensor còn sống hoặc thay memory design.

### 27.3 RTF và latency

- ASR RTF = ASR wall time/audio duration, khai báo có/không gồm decode I/O/frontend/VAD.
- Cascade RTF tính toàn pipeline theo cùng phạm vi công bố.
- Mic latency báo median/p95 sau endpoint, có queue/VAD/MT.
- Báo model/device/precision/batch/context, cold vs warm start. RTF<1 trên file không tự chứng minh microphone low-latency.
- Thời gian2 GPU training không phải thời gian inference deploy1 GPU; benchmark topology riêng.

### 27.4 Lỗi thường gặp

| Hiện tượng | Kiểm tra trước |
|---|---|
| CTC inf/blank-only | Scale fbank, CMVN, loaded tensors, targets/blank, T>=U+R, LR |
| NaN alignment | All-masked rows, normalize eps, padded token loss, FP16 softmax |
| Loss text đẹp nhưng WER không tăng | Shortcut, target drift, λ lớn, prior áp quá mạnh, teacher mismatch |
| GPU0 chạy, GPU1 idle | Launch torchrun, rank/device, sampler, process logs, CPU starvation |
| DDP treo cuối epoch | Ranks khác step count, exception riêng rank, validation collectives, checkpoint barrier |
| Có flag checkpoint nhưng memory không giảm | Source chưa gọi checkpoint, path không dùng flag, retained graphs |
| WER phân tán khác single eval | Duplicate/drop IDs, mean-of-means, refs/hyps ordering, model.eval/masks |
| Resume LR sai | weights-only load, scheduler counted microsteps, world size/budget đổi |
| Dịch sai chiều | src_lang/decoder start ID/tokenizer revision; không dùng word-segmented text |
| Hết disk giữa phiên | Giữ raw+Arrow+WAV duplicate, checkpoints quá nhiều, download weights2formats |

## 28. Tái lập, báo cáo và tiêu chí bàn giao

### 28.1 Mỗi run phải có

```text
runs/<run_id>/
  resolved_config.yaml
  protocol_hash.txt
  environment.json
  source_lock.json
  data_tokenizer_cache_hashes.json
  load_report.json
  init_hashes.json
  sampler_summary.json
  train_metrics.jsonl
  valid_metrics.jsonl
  profiler_summary.json
  best.pt
  last.pt
  predictions.valid.jsonl
  status.json
```

Test/cascade predictions/metrics có thể ở evaluation directory riêng, liên kết checkpoint hash. Failure logs và run status giữ lại. Status: `planned → code_ready → smoke_passed → pilot_passed → running → completed/failed`; chỉ completed khi actual outputs/config/checkpoint hoặc evaluation artifacts đầy đủ.

Mỗi seed có Git/source revision và dependency/model/data revisions. Không chứa secret/API token trong logs/notebook/ZIP. File dưới `sources/` ở project đồng bộ là read-only references; viết outputs/source mới ở thư mục riêng.

### 28.2 Bảng kết quả báo cáo

ASR bảng: run, seed count, params train/infer, WER/CER overall, WER3vùng, S/D/I, training exposure, peakVRAM/rank, thời gian. MT/cascade bảng: data subset N, MT settings, BLEU/chrF++, ASR input source, RTF/latency.

Ô chưa chạy ghi `not_run`, không điền dấu số có vẻ là kết quả. Các ví dụ trong tài liệu này phải ở mục minh họa, không chép vào experimental results.

Báo cáo theo đề bài: giới thiệu → related work → data/audit → baseline → proposed architecture/loss → setup/results/ablation/errors → demo/deployment → limitations/conclusion → references và phân công. Trích paper, source code, datasets và pretrained checkpoints; ghi vai trò AI hỗ trợ và phần nhóm thực sự hiểu/thực hiện.

### 28.3 Tiêu chí bàn giao source runnable

- [ ] Source ZIP mới có root đúng, vendor/patch/licenses, SHA256 và README.
- [ ] CLI/subcommands/schema/notebooks được triển khai, không chỉ stub hoặc file rỗng.
- [ ] Data adapter/splits/audit/exclusions/tokenizer/cache có provenance và coverage.
- [ ] Encoder load/frontend/CTC đúng; activation checkpoint patch đã test hoặc tắt và ghi rõ.
- [ ] Alignment giữ token sequence, frozen teacher target, masks đúng, không teacher injection vào CTC.
- [ ] DDP2 ranks chạy, sampler/normalization/AMP/validation/checkpoint tests pass.
- [ ] One-GPU smoke, two-GPU pilot và tiny overfit có actual logs.
- [ ] Main runs đúng budget/ablation và valid-only selection; test lock được thực hiện.
- [ ] MT giữ cố định, cascade dùng references hợp lệ, không báo benchmark contaminated là clean.
- [ ] Export bundle audio-only, UI upload/mic có hai ngôn ngữ, dài/audio lỗi được kiểm tra.
- [ ] Model bundle, results archive, notebook outputs và instructions resume được xác minh đọc lại.

Nếu đang chỉ được giao viết source mà chưa có GPU/Kaggle access, agent hoàn thành mọi phần độc lập và ghi chính xác phần chưa đo. Không yêu cầu user tin kết quả chỉ từ source inspection; không tuyên bố đã train từ việc tạo notebook.

## 29. Tài liệu tham khảo

Các link primary/official dưới đây dùng cho kiến trúc và cách vận hành, không phải bằng chứng proposal của nhóm đã đạt kết quả.

| Nguồn | Vai trò |
|---|---|
| [ChunkFormer, ICASSP2025 paper](https://arxiv.org/abs/2502.14673) | Backbone và long-form speech transcription |
| [ChunkFormer official source](https://github.com/khanld/chunkformer) | Encoder/frontend/recipe; chốt SHA trước tái sử dụng |
| [ViP-VL paper](https://arxiv.org/abs/2606.10360) và [checkpoint](https://huggingface.co/khanhld/vip-vl-base-vie) | Pretrained Vietnamese encoder và encoder config |
| [PhoBERT, Findings EMNLP2020](https://aclanthology.org/2020.findings-emnlp.92/) | Vietnamese text representations |
| [PhoBERT official repo](https://github.com/VinAIResearch/PhoBERT) | Word segmentation/max length/usage/license |
| [VnCoreNLP official](https://github.com/vncorenlp/VnCoreNLP) | Segmenter assets/preprocessing |
| [Deng et al., ICASSP2022](https://arxiv.org/abs/2203.03582) | Representation knowledge transfer từ language model vào CTC; không đồng nhất module đề xuất |
| [MFGCN, Neurocomputing](https://doi.org/10.1016/j.neucom.2024.128646) | Text-conditioned acoustic pooling trong SER; chỉ lấy cảm hứng phù hợp |
| [ViMD, EMNLP2024](https://aclanthology.org/2024.emnlp-main.426/) và [dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset) | Dữ liệu/nhãn/phân tích đa vùng miền |
| [VinAI Translate, INTERSPEECH2022](https://www.isca-archive.org/interspeech_2022/nguyen22e_interspeech.html) | Hệ thống dịch Việt–Anh |
| [VinAI official repo](https://github.com/VinAIResearch/VinAI_Translate) và [v2 checkpoint](https://huggingface.co/vinai/vinai-translate-vi2en-v2) | Load/generate/settings và contamination note |
| [Whisper official](https://github.com/openai/whisper) | External ASR baseline |
| [PyTorch DDP design](https://docs.pytorch.org/docs/2.9/notes/ddp.html) | Data parallel gradient synchronization |
| [DDP API](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html) | Wrapper/no_sync/parameter constraints |
| [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) | Process setup/checkpoint patterns |
| [torchrun](https://docs.pytorch.org/docs/2.9/elastic/run.html) | Multi-process launch |
| [DataLoader/samplers](https://docs.pytorch.org/docs/2.9/data.html) | Sharding/worker/loading |
| [AMP examples](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html) | FP16/scaler/accumulation |
| [Activation checkpoint API](https://docs.pytorch.org/docs/2.9/checkpoint.html) | Recompute và non-reentrant checkpoint |
| [CTCLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CTCLoss.html) | Loss/reduction/target constraints |
| [HF downloads](https://huggingface.co/docs/huggingface_hub/guides/download) và [audio loading](https://huggingface.co/docs/datasets/audio_load) | Download/cache/revision/data backend |
| [sacreBLEU official](https://github.com/mjpost/sacrebleu) | BLEU/chrF++ và metric signature |
| [Kaggle notebooks docs](https://www.kaggle.com/docs/notebooks) | Runtime/output/platform constraints cần kiểm tra tại phiên |

Link PyTorch2.9 làm mốc API có version, **không yêu cầu Kaggle cài đúng2.9**; agent đối chiếu docs theo version torch thực tế. Không gọi preprint là bài đã peer-reviewed nếu chưa xác minh venue; phân biệt paper, model card và code implementation. Những hyperparameters và công thức normalization DDP cụ thể của nhóm phải ghi là quyết định implementation.

## 30. Prompt hoàn chỉnh giao agent

> Hãy triển khai một project hoàn chỉnh theo toàn bộ đặc tả này. Đọc source ChunkFormer ở SHA cố định, audit encoder/frontend/masks/CMVN/factory và vendor package cùng license. Viết wrapper ViP-VL+CTC riêng, load encoder có report và shared initialization. Thêm nhánh text-guided token–frame pooling dùng frozen PhoBERT token cache, cosine scoring, frame softmax, audio-only values, projection lên teacher dimension và optional weak temporal prior. CTC luôn nhận chuỗi audio gốc; inference không nhận reference/teacher/region labels. VinAI Translate vi→en giữ cố định và chạy sau ASR.
>
> Chuẩn bị ViMD ở revision đã pin, giữ official splits, audit overlap/coverage, xây manifests và tokenizer train-only, cache PhoBERT theo từng content token với long-text windows không silent truncation. Không copy dữ liệu/private audio vào source ZIP. Giữ normalization và frontend khớp checkpoint, đặc biệt waveform scale và CMVN đúng một lần.
>
> Xây training trên Kaggle2T4 bằng DDP2 processes, FP16 AMP, duration-aware sampler có cùng step boundaries, accumulation chuẩn hóa theo global sample count, validation không duplicate/drop mẫu, rank0 checkpoint và per-rank state để resume. Kiểm tra gradient_checkpointing có thật trên encoder; nếu chưa, patch non-reentrant checkpoint từng block và test output/gradient/memory. Tắt LayerDrop core, tránh unused heads. Chạy one-GPU smoke, tiny overfit, DDP gradient equivalence, two-GPU pilot, resume/export tests trước main.
>
> Triển khai B_ref Whisper-small zero-shot, B1 CTC, P1 token pooling, P2 thêm prior theo cùng protocol. Nếu đủ budget mới thêm region/additive/controls, không trộn thay đổi rồi quy kết. Chọn bằng validation, freeze protocol trước test; báo WER/CER overall/theo vùng và cascade BLEU/chrF++ trên tập có English references hợp lệ. Giữ MT settings cố định, ghi rõ PhoMT/mTet-v2 contamination note khi liên quan. Không tự tạo điểm hoặc hứa T4 memory/runtime khi chưa đo.
>
> Bàn giao source ZIP+SHA256, notebooks Kaggle, configs/lock đã xác minh, README, source audit/patch/license/provenance, data/cache preparation scripts, actual test/pilot logs, checkpoint/resume instructions, inference bundle và demo upload/microphone song ngữ. CLI trong tài liệu là contract cần xây; không dừng ở skeleton. Đánh dấu rõ code_ready/smoke_passed/pilot_passed/completed và công việc chưa chạy nếu không có runtime. Không sửa tài liệu synced trong sources/. Mỗi quyết định chưa chốt dùng lựa chọn đơn giản đã mô tả, ghi assumption và tiếp tục các phần độc lập.

---

**Quyết định cốt lõi:** giữ cascade đáp ứng đề bài; nghiên cứu token-level text guidance trong training; bổ sung temporal information qua ablation; dùng DDP để khai thác2T4; mọi lợi ích về WER, dịch và hiệu năng đều phải được đo bằng protocol tái lập.

**Kiểm tra tài liệu tại thời điểm biên soạn:** đã kiểm tra thứ tự 30 mục, cân bằng code fences, parse cú pháp 9 Python code blocks và tính lại ví dụ attention/bộ nhớ/thời gian. Công thức correction accumulation được kiểm tra bằng tính toán gradient giải tích cho số mẫu đều, không đều và window cuối ngắn. Đây không phải test PyTorch/DDP trên GPU; YAML là schema mẫu, chưa được xác nhận bằng loader của project. Chưa triển khai/clone/train dự án chỉ từ việc tạo tài liệu này.
