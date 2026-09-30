# research_v2 — §2.2 Nhánh thị giác chất lượng cao: `[Z ‖ T]` vs `[Z ‖ Q ‖ T]`

Câu hỏi: lấy **đúng** candidate `qlora_r32_decoder` của $1 (Gemma 3 4B NF4 +
QLoRA r32 trên mọi Linear của decoder, 1 epoch = 200 bước, cùng 6400 mẫu train,
cùng loader, cùng 3000 ảnh test), rồi **nối thêm 8 vector** từ một **nhánh thị
giác RIÊNG** — `Q = Aligner(Compressor(Extractor(ảnh 448²)))` — ngay sau 256
token ảnh, thì chất lượng báo cáo X-quang có đổi không?

| candidate | nguồn | chuỗi vào decoder |
|---|---|---|
| **A** `qlora_r32_decoder` (tham chiếu) | $1 — weight, predictions, điểm từng ảnh lấy lại (tem khớp), **không** train lại | `[Z ‖ text]` |
| **B** `qlora_r32_decoder_vextractor` | §2.2 — Stage 0 → Stage 1 → main stage y hệt A + nhánh | `[Z ‖ Q ‖ text]` |

**Biến duy nhất là chuỗi vào decoder.** Z (256 token ảnh) **không bị đụng vào**;
B chỉ có thêm 8 vị trí nằm ngay sau khối ảnh. 8 vị trí ấy có nhãn `-100`,
`attention_mask = 1`, `token_type_ids = 0` — nên **số token vào CE không đổi** và
CE của A với B vẫn là cùng một đại lượng. Mọi khoá quyết định kết quả của A
(`seed`, `data`, `image_policy`, `base`, `prompt`, `generate`, `train`,
`metrics`) được chép nguyên văn từ `configs/lora.yaml` của $1. Vân tay fixture
của `configs/vextractor.yaml` phải **trùng** $1 (selftest kiểm, `train` dừng nếu
lệch).

---

# 1. Kiến trúc — BA KHỐI

```
                  ┌─► SigLIP (đóng băng) ─► projector ─► Z (256, 2560)   [đường của A]
ảnh ──────────────┤
                  └─► resize 448×448, chuẩn hoá theo preprocessor_config  [đường RIÊNG]
                         │
   ┌─────────────────────▼───────────────────────────────────┐
   │ 1. EXTRACTOR — FlashInternImage-L (DCNv4), CẮT stage 3     │  Stage 0 dạy
   │    (B,3,448,448) ─► (B,28,28,640) ─► V = (784, 640)      │
   └─────────────────────┬───────────────────────────────────┘
                         │ V
   ┌─────────────────────▼───────────────────────────────────┐
   │ 2. COMPRESSOR — QFormer, 8 latent @768, N = 4 nhóm       │  Stage 1 dạy
   │                                                          │
   │    kv = RMSNorm(V)                          640          │
   │    q  = 8 latent @768                                    │
   │    N × [ khối 1:  q += SelfAttn(N(q))                    │
   │                   q += CrossAttn(N(q), kv)   ◄─ đọc V    │
   │                   q += SwiGLU(N(q))                      │
   │          khối 2:  q += SelfAttn(N(q))                    │
   │                   q += SwiGLU(N(q))         ]            │
   │    ─► V_c = (8, 768)                                     │
   └─────────────────────┬───────────────────────────────────┘
                         │ V_c
   ┌─────────────────────▼───────────────────────────────────┐
   │ 3. ALIGNER — MHAttn(768) ‖ SwiGLU 768→2048→2560          │  Main stage dạy
   │    ─► Q (8, 2560)                                        │
   └─────────────────────┬───────────────────────────────────┘
                         │
   [Z ‖ Q ‖ text] ─► Gemma decoder (QLoRA r32) ─► CE
```

**BA khối, BA giai đoạn, và mỗi giai đoạn dạy ĐÚNG MỘT khối.**

| | dạy | đứng yên | hàm mục tiêu |
|---|---|---|---|
| **Stage 0** | Extractor (**115,64 M**) | — | DINO tự giám sát (teacher = EMA, multi-crop) từ init ImageNet-22k |
| **Stage 1** | Compressor (**36,98 M**) | Extractor (teacher Stage 0) | CE với phân phối prototype của teacher — teacher **đóng băng hoàn toàn** |
| **main** | Aligner (**10,75 M**) + LoRA từ bước 0; Compressor sau 20% bước | Extractor | CE trên báo cáo thật |

---

## 1.1. Extractor — `flash_intern_image_l_22k_384`, `n_stages: 3`

**FlashInternImage là một CNN, không phải một transformer.** Lõi của nó là
**DCNv4** — deformable convolution v4: mỗi vị trí học `group × 9` *offset* và
*mask* (gộp vào MỘT `Linear`, và **bỏ softmax** trên mask của DCNv3), rồi lấy
mẫu lưới trong một kernel FUSED. Nên nó có trường thu nhận thích nghi như
attention nhưng chi phí tuyến tính theo số ô, và nó **không có pos-embed để nội
suy, không có cửa sổ để đệm, không có cls token**.

**Thân của nó chạy trên CHUỖI, không trên lưới.** DCNv4 nhận `(N, L, C)` cộng
`shape = (H, W)` RỜI, nên `FlashInternImageBody` làm phẳng lưới ngay sau
`patch_embed` rồi luồn `shape` qua từng level; mỗi `downsample` trả về `shape`
mới. Đó là khác biệt bố cục lớn nhất so với đường DCNv3 — và là lý do lát cắt
phải chép đúng vòng lặp của upstream thay vì gọi `forward_features` (bản
upstream chạy hết bốn level rồi qua `conv_head` + `avgpool` và trả về MỘT
vector, không phải một lưới).

> **⚠ Backbone này KHÔNG nạp qua timm, cũng KHÔNG qua `trust_remote_code`.**
> Code model không nằm trong package pip (PyPI chỉ có *toán tử* DCNv4) mà ở
> thư mục `classification/` của repo GitHub, vốn không phải một package cài
> được — nên nó được **VENDOR** nguyên văn vào `src/vendor/flash_intern_image.py`
> (commit `4b848f7`, sha256 ghim trong `src/vendor/NOTICE.md`). Trọng số là một
> `.pth` **trần** tải bằng `hf_hub_download` từ `OpenGVLab/DCNv4`.
>
> `backbone_body` vì thế có **BA** đường, và `arch` chọn đúng một:
> `flash_intern_image_*` -> vendor + `.pth`; `*internimage*` ->
> `trust_remote_code` (DCNv3); còn lại -> timm. Ba họ, ba cách cắt: `.levels`
> (chuỗi) · `.model.levels` (lưới NHWC) · `.stages`.

> **⚠⚠ KHÔNG có đường lùi PyTorch — và đây là rủi ro vận hành duy nhất của
> lần đổi này.** `flash_intern_image.py` `import DCNv4` **trần**, và
> `DCNv4/functions/dcnv4_func.py` lại `from DCNv4 import ext` ở tầng module:
> thiếu extension CUDA đã biên dịch thì nó chết ngay lúc **import**, không phải
> lúc forward. PyPI chỉ có **sdist** (1.0.0.post2, 01/2024 — trước Blackwell
> ~1,5 năm), `setup.py` raise nếu thiếu `CUDA_HOME`, và sm_120 cần nvcc ≥ 12.8:
>
> ```bash
> bash scripts/setup_dcnv4.sh      # `scripts/setup.sh` cũng TỰ gọi nó
> ```
>
> Script ấy làm sáu việc mà một dòng `pip install` không làm được: kiểm chính
> **torch** có cubin cho card này chưa (không thì dịch DCNv4 là vô nghĩa), dựng
> một `CUDA_HOME` tối thiểu từ wheel `nvidia-cuda-nvcc-cu12` khi box không có
> toolkit (~150 MB, không cần `sudo`), chặn trước ca `nvcc < 12.8` với sm_120,
> đặt `TORCH_CUDA_ARCH_LIST` theo compute cap ĐO ĐƯỢC, thêm
> `--no-build-isolation` (`setup.py` của DCNv4 `import torch` ngay lúc build),
> và cuối cùng **chạy thật kernel ở fp32/bf16/fp16** — vì "pip install xong"
> không bằng "kernel chạy trên card này". Vì sao nó không thể là một dòng trong
> `requirements.txt`: xem đầu `requirements-dcnv4.txt`.
>
> Không build được thì trả `vextractor.extractor.arch` về
> `OpenGVLab/internimage_l_22k_384` — **đường DCNv3 vẫn còn nguyên trong code**
> và chạy ở mọi máy; kiến trúc y hệt, `V` vẫn `(784, 640)`, chênh đúng 24 930
> tham số (0,02%).
>
> **Cái mua được**: DCNv4 là kernel **fused** — nó không materialize tensor
> `(N·group, H·W, group_channels, 9)` mà bản PyTorch của DCNv3 giữ cho backward
> ở **mỗi** block. Đó đúng là thứ đã làm Stage 0 hết 16 GB VRAM.
>
> Để so sánh, đường DCNv3 thì ngược lại: **kernel CUDA của nó KHÔNG bao giờ
> được dùng.** Trong `dcnv3_func.py` của OpenGVLab, `pkg_resources` được `import`
> **sau** khối `try` đã dùng nó, nên nhánh "có kernel" luôn ném `NameError` và
> bị `except:` bắt: `has_cuda_kernel` **luôn** là `False` và model luôn chạy
> `DCNv3_pytorch`. Tin tốt: không phải biên dịch extension nào (đúng thứ hay
> chết trên Windows). Tin xấu: bản PyTorch trải 9 điểm lấy mẫu bằng
> `grid_sample`, nên nó chậm hơn và ngốn activation hơn — `stage0.micro_batch`
> và `vextractor.extractor.with_cp` là hai chỗ để trả giá đó.

> **⚠ `_22k_384` là trọng số CÓ GIÁM SÁT trên ImageNet-22k**, không phải tự
> giám sát. Nên Stage 0 ở đây **không** phải "chạy tiếp đúng hàm mục tiêu" như
> với một checkpoint DINOv3: nó là DINO self-distillation **khởi đi từ một init
> có giám sát**. Được một backbone mạnh, mất tính liên tục của hàm mục tiêu —
> và đừng đọc đường loss của Stage 0 như thể nó nối tiếp một lượt pretrain tự
> giám sát nào. (HF không phát hành bản `l_1k_224`: họ L chỉ có `22k_384` và
> `22kto1k_384`.)

**`n_stages: 3` — cắt SAU stage 3, dừng ở stride 16 / 640 kênh.** FlashInternImage-L
có `channels: 160`, `depths: [5, 5, 22, 5]`, nên bốn stage chạy ở 160 · 320 ·
**640** · 1280 kênh. Ở 448² (stem /4 rồi hai lần /2 trước khi tới stage 3) lưới
là **28×28 = 784 ô × 640** — đây là `V`. Lát cắt ấy là một lựa chọn có hai mặt:
bỏ tầng ngữ nghĩa cao nhất, đổi lấy một lưới **mịn gấp 4 lần** so với 14×14 của
đủ bốn stage, và Cross-attn của Compressor cùng bản đồ saliency của Lens đều
đọc trực tiếp lưới ấy. Nó cũng là chỗ **22 trong 37 block** của backbone nằm,
nên nó không phải một lát cắt "cho rẻ".

**Cắt stage là HAI việc, và cả hai đều cần.** `levels[i]` chạy `depths[i]` block
ở `160·2^i` kênh **rồi mới** downsample; nên (1) `downsample` của level cuối còn
lại bị **bỏ** — giữ nó là trả về lưới của stage SAU, nửa độ phân giải và gấp đôi
kênh; (2) `levels[3]` bị **xoá khỏi `ModuleList`**, không chỉ bỏ không gọi. Tham
số của một level chết vẫn nằm trong `state_dict`, vẫn đi vào optimizer của
Stage 0, vẫn được ghi vào checkpoint — và với FlashInternImage-L đó là
**95,9 M** tham số, nhiều hơn Compressor và Aligner cộng lại.

**Và `conv_head` cũng phải bị bỏ** — chỗ này FlashInternImage khác
`InternImageModel` của HF: nó dựng `conv_head` **dù `num_classes=0`** (chỉ
`head` mới rơi về `nn.Identity`), nên bỏ không gọi là để lại thêm **2,5 M**
tham số chết (Conv 1280→1920 + BN) đi vào optimizer và vào checkpoint.

**`AutoModel` dựng thân với `num_classes=0`**, nên `conv_head` và `head` không
tồn tại: 43 M trọng số phân loại 21 841 lớp của checkpoint 22k là khoá **thừa**
lúc nạp (transformers cảnh báo rồi bỏ), không phải tham số chết đi vào
optimizer.

> `n_patch` (784) và `d_feat` (640) đến từ **một lượt forward THẬT** với tensor
> 0 (trong `eval()` + `no_grad()` + `fork_rng`, nên không tiêu RNG), không từ
> một công thức stride gõ tay — chúng không thể lệch với model.

**MỘT `state_dict` chạy cả 448² lẫn 224².** DCNv3 lấy mẫu bằng `grid_sample` nên
không tham số nào gắn với cỡ ảnh: 448² -> 28×28, 224² -> 14×14, cùng một bộ
trọng số. Điều kiện bắt buộc của multi-crop Stage 0 — và nó được **probe THẬT**
ở nửa độ phân giải ngay lúc dựng (`multi_res`), không được tin vì nó đúng về
nguyên tắc. `n_local > 0` với một backbone chỉ nhận một cỡ ảnh bị **dừng ở
đó**, thay vì chết ở lượt forward local crop đầu tiên sau khi đã nạp xong dữ
liệu lẫn trọng số.

**Không prefix token.** InternImage trả một **LƯỚI**, không có cls token nào, nên
`stage0.dino.pool` bắt buộc `mean` (trung bình **784 ô**) và `pool: cls` bị **từ
chối ngay lúc dựng backbone** — trước khi nạp dữ liệu, trước khi dựng optimizer
— chứ không lặng lẽ rơi về `mean`.

**Bố cục trục là `NHWC`, và nó được HỎI chứ không đoán.** DCNv3 làm việc trên
`(N, H, W, C)`, nên `InternImageBody.forward_features` trả đúng bố cục ấy —
không permute thêm lần nào — và `output_fmt = "NHWC"` nói ra điều đó.
`patches()` gộp bằng `flatten(1, 2)`. Dùng nhầm nhánh NCHW cho nó **không ném
lỗi**: nó trả `(B, W·C, H)`, `d_feat` đọc ra **28** thay vì 640, `Linear(28, …)`
dựng được, và cả đường ống chạy hết trên rác. Đó là lý do trục kênh đến từ
`output_fmt`, không từ `f.dim()`: NCHW và NHWC **cùng 4 chiều**.

**Không có BatchNorm nào để phải xử lý.** InternImage chuẩn hoá bằng LayerNorm
(`norm_layer: LN` ở cả stem, downsample và mọi block), tính theo từng vị trí của
từng ảnh, không mang theo thống kê chạy nào — nên Stage 0 không cần epoch nào
để kéo thống kê về ảnh CXR.

**Chuẩn hoá ảnh: đây là chỗ DUY NHẤT của cả nhánh phải GÕ TAY một cặp số, và
lý do thì đáng nói ra.** Checkpoint là một `.pth` **trần** — nó không mang
`preprocessor_config.json` nào (đường DCNv3 có), không mang `pretrained_cfg`
nào (đường timm có). Nên `FLASH_II_NORMALIZE` ghi thẳng cặp số, kèm nguồn chính
xác: `classification/config.py` @ `4b848f7` dòng 187-188 (`_C.AUG.MEAN` /
`_C.AUG.STD` = mean/std ImageNet), và `flash_intern_image_l_22kto1k_384.yaml`
**không** ghi đè hai khoá ấy (`dataset/build.py` gọi
`transforms.Normalize(AUG.MEAN, AUG.STD)`). `body.pretrained_cfg` vẫn được gắn
từ chính hằng ấy, để `_check_normalize` so với MỘT nguồn chứ không rơi về
`IMAGENET_*` — hai cặp số tình cờ bằng nhau là đúng ca mà một cổng kiểm im
lặng.
`vextractor.vext_tensor()` là chỗ **duy nhất** dựng ảnh của nhánh và nó **bắt
buộc** phải nhận mean/std (không có mặc định); `assert_same_pixels` kiểm lại
trên một mẫu thật mỗi lần chạy.

**`with_cp: true` là MẶC ĐỊNH, và đó là con số đo được.** Nó bật
`torch.utils.checkpoint` trong từng `InternImageLayer`. `dcnv3_core_pytorch`
giữ một tensor `(N·group, H·W, group_channels, 9)` **mỗi block** để backward —
ở 448² với InternImage-L là ~72 MB cho mỗi block của level 0 (112×112,
group 10), nhân **32 block**, rồi nhân **8 view** mà `dino.loss` giữ sống cùng
lúc. Tắt khoá này thì Stage 0 OOM ngay ở mảnh vật lý 4 trên một card 16 GB. Vì
kernel CUDA của DCNv3 **không bao giờ** được dùng, bản PyTorch là bản duy nhất
từng chạy — nên đây là ca THƯỜNG, không phải ngoại lệ của một máy. Giá: ~30%
thời gian Stage 0.

Nó cũng là khoá DUY NHẤT của khối `extractor` **không** vào vân tay Stage 0:
nó đổi **chi phí** của lượt forward, không đổi một tensor nào — nên bật/tắt
không bắt chạy lại một lượt DINO nhiều giờ. Khai `with_cp: true` với một `arch` của timm bị **từ
chối ồn ào**: model của timm không đọc khoá ấy, và một khoá không ai đọc là im
lặng tuyệt đối.

## 1.2. Compressor — QFormer, 8 latent @768, N = 4

**8 latent là khoá KHAI tường minh** (`vextractor.compressor.n_query`). Lưới
của Extractor (**784 ô** ở 448²) là **nguồn để hỏi**, không phải khối Q: 8
vector đi hỏi nó qua **Cross-attn** và nén lại.

**640 → 768 xảy ra NGAY TRONG Cross-attn.** `k_proj`/`v_proj` nhận 640 và trả
768; `q_proj`/`o_proj` ở 768. **Không có `Linear 640→768` đứng trước** — một
lớp chiếu riêng sẽ thêm 0,49 M tham số mà không thêm sức biểu diễn nào (nó chỉ
gộp được vào `k`/`v` phía sau), và nó buộc cả bốn nhóm dùng chung MỘT bản 768
chiều của lưới thay vì để mỗi nhóm tự chiếu lấy thứ nó cần. Thứ **duy nhất**
đứng trước Cross-attn là `RMSNorm(640)` trên V (640 tham số): nó đưa key/value
về RMS ≈ 1, thang mà mọi phép attention bên trên được thiết kế cho.

**Khối 2 KHÔNG có Cross-attn, và đó là chủ ý.** Nó để 8 vector *thống nhất lại
với nhau* sau khi vừa đọc xong ảnh, trước khi nhóm sau lại đi đọc tiếp. Một
Cross-attn thứ hai liền kề chỉ là đọc lại cùng một lưới bằng cùng một bộ câu hỏi
vừa được cập nhật nửa vời. Với `N = 4` thì cả cụm đọc ảnh đúng **bốn** lần, xen
giữa là ba lượt tự-trộn.

**Pre-norm + residual quanh TỪNG khối con** — năm dấu `+` và năm RMSNorm mỗi
nhóm, đúng quy ước Gemma 3. Selftest so **thẳng** năm công thức ấy, nên trả nhầm
một residual sẽ bị bắt ngay.

**Mọi attention chạy ở `head_dim` = 128, và đó là khoá KHAI.** Số head được
**suy ra**: `n_heads = d_latent // head_dim` = **6** ở 768. Chiều ngược lại —
khai `n_heads` rồi để `head_dim` rơi ra — là cách head_dim đổi **lặng lẽ** mỗi
lần ai đó sửa `d_latent`: 8 head cho đúng 128 ở 1 024, nhưng cùng 8 head ở 768
ra 96. Khai cái **bất biến**, suy ra cái phụ thuộc; `d_latent` không chia hết
`head_dim` thì **dừng**, không làm tròn. Cùng `head_dim` ấy được **dùng lại**
cho MHAttn của Aligner, không khai lại — một khoá thứ hai cho cùng con số là
một khoá có thể lệch.

**HAI khối SwiGLU, hai khoá khai riêng.** `vextractor.compressor.mlp_hidden`
(512) là bề rộng FFN **bên trong Compressor** — một **nút THẮT** (`SwiGLU
768→512→768`, hẹp hơn `d_latent`): mỗi lượt FFN buộc 8 vector bỏ bớt thứ gì đó
thay vì nở ra. `vextractor.aligner.mlp_hidden` (2048) là bề rộng FFN của
**Aligner**, và khối ấy đi hướng ngược lại.

| khối con (mỗi nhóm) | tham số |
|---|---|
| Self-attn @768 (×2) | 2 359 296 mỗi cái |
| Cross-attn (q/o @768, k/v 640→768) | 2 162 688 |
| `SwiGLU 768→512→768` (×2) | 1 179 648 mỗi cái |
| 5 × RMSNorm(768) | 3 840 |
| **khối 1** (self + cross + ffn) | **5 703 936** |
| **khối 2** (self + ffn) | **3 540 480** |
| **MỘT nhóm khối** (khối 1 → khối 2) | **9,24 M** |

> Đặc tả ghi `~9.24M (xN)` — số ĐO được cho một nhóm là **9 244 416**, khớp.
> Tổng Compressor ở `N = 4` thì đặc tả ghi `~36.96M` (= 4 × 9,24) còn số đo là
> **36,98 M**: chênh lệch là `8 × 768` của `latent` cộng `640` của
> `RMSNorm(V)` — hai thứ nằm NGOÀI mọi nhóm khối.

**Chi phí nằm ở `d_latent`, không ở `n_query`.** 8 latent chỉ là **6 144**
tham số, trong khi mỗi Self-attn @768 là 2,36 M. Đổi `n_query` gần như không
đụng tới bảng trên (nó đụng tới *thời gian* và tới độ dài chuỗi decoder); đổi
`d_latent` thì đụng tới tất cả, theo d².

## 1.3. Aligner — `MHAttn(768)` + `SwiGLU 768→2048→2560`

**HAI khối con, và chỉ khối sau đổi chiều:**

    x = V_c + MHAttn(RMSNorm(V_c))
    Q = down( SiLU(gate(y)) ⊙ up(y) ),   y = RMSNorm(x)

**MHAttn ở đây KHÔNG đọc ảnh.** Nó là một lượt **tự**-chú ý trên đúng 8 vector
của `V_c`; Cross-attn của Compressor là chỗ duy nhất chạm lưới. Nên nó không
thêm một bit thông tin thị giác nào — việc của nó là trộn 8 vector MỘT lần nữa
ngay tại ranh giới ngữ nghĩa, sau khi Compressor đã nén xong và trước khi `down`
đưa chúng sang không gian embedding của decoder. Đó cũng là khác biệt duy nhất
về *bố cục* so với bản trước, nơi Aligner là một FFN thuần.

**Residual quanh MHAttn, KHÔNG quanh SwiGLU.** Khối attention giữ nguyên chiều
nên dấu `+` có nghĩa; khối SwiGLU đổi 768 → 2560 nên không có gì để cộng vào —
và đó là lý do Aligner không phải một block transformer đầy đủ: nửa sau của nó
là một phép **đổi không gian**, không phải một lượt tinh chỉnh.

**RMSNorm trước SwiGLU không phải trang trí, và đây là chỗ đáng đọc kỹ.** SwiGLU
được mượn từ FFN của decoder, nơi đầu vào **luôn** đã qua RMSNorm — RMS ≈ 1 là
một giả định ngầm của cả thiết kế. `V_c` thì không: nó tới Aligner với RMS nhỏ
(latent khởi tạo ở 0,02, và residual do latent chi phối). Một khối có **cổng
nhân**, ăn đầu vào nhỏ, ở init nhỏ, thì **bình phương** cái nhỏ ấy. Đo thật,
trên chính khối này với ba cách khởi tạo:

| Aligner | RMS(Q) ở bước 0 |
|---|---|
| MLP(GeLU) không cổng, std 0,02 | ~6e-3 |
| SwiGLU, cả ba ma trận std 0,02, **không norm** | **~1e-4** |
| **RMSNorm + SwiGLU, init `1/√fan_in`** | **~5,8e-1** |

Bản giữa là hỏng thật, không phải chuyện thẩm mỹ: trên model đồ chơi của
selftest nó **underflow hẳn về 0**, và bản đồ saliency của Lens thành một ma
trận không. Nhân init lên để bù thì chữa *triệu chứng* — con số bù suy từ RMS
đo được chứ không từ nguyên tắc nào, và nó trôi ngay khi Compressor đổi.
RMSNorm chữa *nguyên nhân*: nó đặt đầu vào của SwiGLU lên đúng thang mà SwiGLU
được thiết kế cho, nên `1/√fan_in` — init chuẩn — lại có nghĩa. Giá: **768
tham số** mỗi norm, và Aligner có hai.

`_init` (normal(0, 0.02) cho mọi `nn.Linear` mới) **bỏ qua** ba ma trận có cổng,
y như nó bỏ qua Extractor — đè 0,02 lên chúng là quay lại đúng hàng giữa của
bảng trên, và im lặng. Nhưng nó **vẫn chạm** bốn chiếu của MHAttn: attention
không có cổng, và nó đi theo đúng quy ước của Compressor.

> Chỗ `skip` cho Extractor **không** phải đề phòng: mỗi `InternImageLayer` cài
> `offset`, `mask`, `input_proj`, `output_proj` và hai lớp MLP bằng `nn.Linear`
> trên tensor NHWC, nên InternImage-L cắt còn 3 stage vẫn có **192**
> `nn.Linear` (32 block × 6). Thiếu `skip` là xoá sạch Extractor một cách im
> lặng.

`aligner.mlp_hidden` 2048 > 768 — một chỗ **nở ra**, cùng hướng với FFN của
decoder (Gemma 3: 2560 → 10240) và ngược hướng với nút thắt 512 bên trong
Compressor. Hai khối SwiGLU, hai khoá, hai hướng — và đó là chủ ý: Compressor
*nén*, Aligner *dịch*.

Một khác biệt đáng biết nếu muốn khớp decoder cho bằng: FFN của Gemma 3 thực ra
là **GeGLU** (`gelu_pytorch_tanh`), không phải SwiGLU; ở đây dùng SiLU đúng như
đặc tả.

## 1.4. Bảng tham số

Mọi con số dưới đây là **số ĐO ĐƯỢC** trên model thật (selftest dựng lại nhánh
từ `configs/vextractor.yaml` rồi đòi đúng những chuỗi này có trong README —
xem bài 22).

| khối | tham số |
|---|---|
| **Extractor** FlashInternImage-L @448 → `(784, 640)`, `n_stages: 3` | **115,64 M** — *Stage 0 dạy; sau đó teacher inference* |
| RMSNorm(640) trên V | 640 |
| 8 latent @768 | 6 144 |
| **1 nhóm khối** (Self ×2 + Cross + SwiGLU ×2 + 5 norm) | **9,24 M** |
| **Compressor** (N = 4) → `(8, 768)` | **36,98 M** — *Stage 1 dạy* |
| **Aligner** `MHAttn(768)` + `SwiGLU 768→2048→2560` → Q (8, 2560) | **10,75 M** — *main stage dạy* |
| **nhánh ở main stage** | **163,37 M**, trong đó **học được 47,73 M** |
| DINO head (CHỈ Stage 0 / Stage 1, bị bỏ sau đó) | **7,09 M** |
| **Stage 0**: student (Extractor + head) | **122,72 M** *(+ teacher EMA 122,72 M nữa trong VRAM)* |

> ⚠ Đặc tả ghi Extractor `~115.57M`; số đo được là **115 637 280**. Lệch
> **0,07 M** (0,06%) — đúng phần biases của LayerNorm/conv mà một phép nhẩm hay
> bỏ. (Đường DCNv3 là **115 662 210**: hai họ chênh nhau đúng **24 930** tham
> số, vì DCNv4 gộp `offset`+`mask` vào một `Linear` và bỏ `LayerNorm` + `GELU`
> trong `dw_conv`.) Aligner thì khớp tuyệt đối: MHAttn **2 359 296** (đặc tả ~2.36M) +
> SwiGLU **8 388 608** (~8.39M) + hai RMSNorm 1 536 = **10 749 440** (~10.75M).

**Vì sao Q đứng SAU Z**: attention của decoder là nhân quả, nên mọi vị trí của Q
đọc được toàn bộ 256 token ảnh, còn phần văn bản phía sau đọc được cả Z lẫn Q.
Đặt Q trước Z thì 8 vector ấy chỉ nhìn thấy BOS + câu lệnh hệ thống. Chỗ nối
lấy từ `recombiner.last_image_index(ids) + 1` (khâu train, đệm phải) và
`recombiner.image_end_indices(ids) + 1` (khâu sinh, đệm trái — mỗi hàng một chỗ).

---

# 2. Lịch trình huấn luyện

## 2.0. Lịch CHUNG (`src/schedule.py`)

Stage 0 và Stage 1 dùng **cùng một hàm**. Hai chặng, cộng một hệ số nhân:

1. **warm-up** tuyến tính `lr_min` → `lr_max` — trong `warmup_epochs` (**1**) epoch
2. **giữ** `lr_max` — **mọi epoch còn lại**
5. **mỗi epoch val không cải thiện** → chia lr hiện tại cho `plateau_gamma`
   (**1,5**), sàn `lr_min`; val cải thiện thì **giữ nguyên** lr

Với Stage 0 (6400 mẫu, batch 32 → **200 bước/epoch**): dốc 200 bước từ 1e-8 lên
2e-5, rồi giữ 2e-5 cho tới lúc dừng.

> **`lr_max` là MỨC CHẠY, không phải một đỉnh tạm rồi hạ.** Không còn `lr_main`,
> không còn chặng cosine nào trong lịch chung. Thứ **duy nhất** kéo lr xuống là
> chặng 5 — tức chính **đường val**, chứ không phải một lịch định trước. Một
> đường lr do val điều khiển thì không cần đoán trước lúc nào nên hạ.

**Bước 0 có đúng `lr_min`**, và `lr_min` là **ba thứ cùng lúc**: đầu dốc
warm-up, **sàn** của chặng 5, và **đáy** của epoch đuôi. Nên đừng đọc nó chỉ
như "lr nhỏ nhất" — đổi nó là đổi cả ba.

**Chặng 5 là một hệ số NHÂN, không phải một chặng thứ ba trên trục thời gian.**
Nó chia đường lr trên cho `1,5^n_decays`, trong đó `n_decays` là **tổng số
epoch val không cải thiện** từ đầu giai đoạn. Ba tính chất, cả ba đều là lựa
chọn có hệ quả:

* **cộng dồn, không reset.** "decay lr *hiện tại*" rồi "duy trì lr" nghĩa
  là một lần tụt là tụt hẳn: val khá lên trở lại thì lr đứng yên ở mức đã tụt
  chứ không leo lại. Đường lr vì thế **đơn điệu giảm** kể từ epoch 2.
* **`patience` và chặng 5 là HAI bộ đếm.** `bad` đếm chuỗi hỏng **liên tiếp**,
  reset mỗi lần val khá lên, và quyết định lúc nào **dừng**. `n_decays` đếm
  **tổng** số epoch hỏng, không bao giờ reset, và quyết định **lr**. Gộp chúng
  là để lr leo lại mỗi lần val khá lên — đúng thứ "duy trì lr" loại bỏ. Muốn
  tắt chặng 5 thì đặt `plateau_gamma: 1.0`, **không phải** `patience: 0`.
* **sàn ở `lr_min`.** Sau đủ nhiều lần chia (2e-5 → 1e-8 là **19** lần), lr sẽ
  chui xuống dưới `lr_min`; lúc ấy epoch đuôi — vốn cosine *về* `lr_min` —
  thành một cú **tăng** lr, ngược hẳn với ý của nó.

**Vì sao 1,5 chứ không phải 3.** Hệ số nhỏ chia quãng đường tới sàn thành
**nhiều nấc**: 2e-5 → 1e-8 là **19** lần chia thay vì 7, nên lr hạ dần theo
đúng nhịp val xấu đi thay vì rơi hẳn sau vài epoch hỏng. Đổi lại, một chuỗi
hỏng ngắn gần như không đụng tới lr — với `patience` = 4 (đồng bộ cho cả hai
giai đoạn) thì tới lúc **dừng** lr mới chỉ còn `1/1,5⁴ ≈ 20%` mức chạy, so với
`1/3⁴ ≈ 1,2%` ở hệ số 3. Chặng 5 vì thế là một cái **phanh mềm**, không phải
một cái công tắc.

Ví dụ với Stage 0 (`lr_max = 2e-5`, `lr_min = 1e-8`):

| số epoch val hỏng | 0 | 1 | 2 | 3 | 4 | 6 | 9 | 13 | ≥ 19 |
|---|---|---|---|---|---|---|---|---|---|
| lr | 2,0e-5 | 1,3e-5 | 8,9e-6 | 5,9e-6 | 4,0e-6 | 1,8e-6 | 5,2e-7 | 1,0e-7 | 1,0e-8 (sàn) |

Stage 1 dùng **một bộ đếm chung cho cả hai nhóm** (attn và ffn), vì val là
**một** con số — không có cách nào nói nhóm nào làm nó xấu đi, nên tách hai bộ
đếm là bịa ra một tín hiệu không tồn tại. Sàn thì riêng: mỗi nhóm kẹp ở `lr_min`
của chính nó.

Ba siêu tham số **khác** có đơn vị epoch và **không** đi theo `warmup_epochs`:
momentum EMA của teacher (cosine trên **toàn** giai đoạn, `max_epochs`),
`teacher_temp_warmup_epochs` và `freeze_last_layer_epochs` — chúng tả hàm mục
tiêu DINO chứ không tả lịch lr.

**Cái ĐUÔI sau patience.** Khi `patience` epoch liên tiếp val không cải thiện,
giai đoạn **không dừng ngay**: nó chạy thêm `tail_epochs` epoch với **cosine từ
lr đang chạy (đã gồm mọi lần chia của chặng 5) về `lr_min`**, rồi mới lấy
checkpoint tốt nhất theo val. Epoch ấy là một lần "hạ cánh": nó cho phép trọng
số rơi vào đáy gần nhất của vùng nó đang ở, và điểm val của nó vẫn được đem so
với kỷ lục cũ — nên nếu nó tệ hơn thì `best` vẫn là epoch cũ, không mất gì.

`lr_from` của đuôi là lr **thật** của bước cuối đã chạy — tức đã gồm mọi lần
chia của chặng 5. Lấy `lr_max` gõ tay là để lr nhảy **lên** mấy nấc rồi mới đi
xuống, đúng ngược với ý "hạ cánh".

## 2.1. Stage 0 — DINOv3 finetuning (`src/stage0_vextractor.py`, `src/dino.py`)

| Mô tả | Chỉ huấn luyện **Extractor** với dữ liệu MIMIC-CXR hiện tại |
|---|---|
| Mục tiêu | Biến Extractor thành module học đặc trưng chất lượng cao trên miền CXR |
| lr | `lr_min=1e-8` · `lr_max=2e-5` (warm-up 1 epoch rồi **giữ** `lr_max`) |
| Epoch | `50` (trần) |
| Chặng 5 | mỗi epoch val không cải thiện → lr ÷ `1,5` (sàn `lr_min`), val khá lên thì giữ nguyên |
| Patience | `4` — đạt patience thì cosine decay từ lr **hiện tại** về `lr_min` thêm 1 epoch rồi mới lấy best val |
| Điểm xuất phát | Nạp lại trọng số pretrain rồi finetune tiếp |

```
ảnh CXR ─► multi-crop: 2 global 448² (→ 28×28) + 6 local 224² (→ 14×14)
        ─► student = FlashInternImage-L (stage 3) + DINO head ─┐
        ─► teacher = EMA(student), CHỈ thấy 2 global            ┴─► CE(t ‖ s) = loss
```

**"Load lại training state của pretrain" nghĩa là gì ở đây.** Checkpoint mà
OpenGVLab phát hành chỉ có **trọng số** — không optimizer, không center, không
teacher EMA riêng. Nên Stage 0 nạp trọng số ấy vào **cả student lẫn teacher**,
dựng DINO head / optimizer / center mới, rồi chạy **hàm mục tiêu DINO** trên
miền ảnh CXR. `pretrained: false` bị **từ chối** — finetune một backbone ngẫu
nhiên trả lời một câu hỏi khác hẳn.

**Không nạp Gemma.** Nhánh có encoder riêng, nên cả lượt finetune chỉ là
Extractor (**115,64 M**) + một DINO head (**7,09 M**) = **122,72 M**, và
**teacher EMA là một bản sao đầy đủ nữa trong VRAM**. Thứ duy nhất còn cần từ
model là `d_model` (2560), đọc từ `AutoConfig` (một file json, không tải trọng
số).

> **Đây là giai đoạn ĐẮT NHẤT của cả thí nghiệm, và nó đắt hơn bản trước.**
> Multi-crop nhân số lượt forward lên `2 + n_local` = **8** cho student cộng 2
> lượt `no_grad` cho teacher, trên một backbone **2,3× lớn hơn** FocalNet-Small
> và chạy DCNv3 bằng **PyTorch thuần** (9 điểm `grid_sample` mỗi vị trí). Hai
> khoá để trả giá: `vextractor.extractor.with_cp` (MẶC ĐỊNH `true` — gradient
> checkpointing trong từng `InternImageLayer`, đổi ~30% thời gian lấy phần lớn
> activation; xem §1.1) và `configs/gpu/<card>.yaml` → `stage0.micro_batch`
> (khởi điểm **4**; autobatch dò lại từ đó).
>
> **`probe_sizes` của Stage 0 phải chứa mốc 1.** Nếu mốc NHỎ NHẤT của nó OOM,
> `autobatch` dò **xuống** dưới mốc ấy rồi chọn mảnh lớn nhất còn vừa; OOM ở
> mọi mảnh thì nó **DỪNG** và chỉ ra `with_cp`. Bản trước giữ lại
> `micro_batch` — tức một số LỚN HƠN mốc vừa OOM — và chết ở bước đầu của
> epoch 1 sau khi đã nạp xong dữ liệu lẫn trọng số.

**`pool: mean`.** InternImage không có cls token, nên DINO head đọc **trung bình
784 ô** của lưới. `pool: cls` bị **từ chối ngay lúc dựng backbone**, không lặng
lẽ rơi về `mean`: config nói một đằng chạy một nẻo là thứ không bao giờ tự lộ
ra. `pool_feature` cũng **nhận `fmt`** (`NHWC`) thay vì đoán từ `f.dim()`.

**Bốn thứ giữ DINO khỏi sụp về một điểm**, và cả bốn đều phải có: (1) teacher là
**EMA** của student, momentum cosine 0.996 → 1, không nhận gradient; (2)
**centering** — trừ một vector trung bình trượt của logits teacher trước softmax;
(3) **sharpening** — τ_t (0.04) **nhỏ hơn** τ_s (0.1), và `dino_cfg` **từ chối**
cấu hình đảo dấu; (4) **multi-crop** — student thấy tất cả các view, teacher chỉ
thấy global, cặp trùng view bị bỏ (2 × 8 − 2 = 14 số hạng với `n_local: 6`).

**Lật ngang mặc định TẮT** (`aug.hflip: 0.0`). Ảnh CXR có chiều: tim nằm bên
trái, và *dextrocardia* là một chẩn đoán chứ không phải một augmentation. Tăng
cường còn lại viết bằng **PIL** (crop tỉ lệ, sáng/tương phản, blur) — cùng đường
với `vext_tensor`, nên ảnh Stage 0 và ảnh main stage không thể lệch nhau vì hai
thư viện resize khác nhau.

**Epoch được chọn theo LOSS TỰ GIÁM SÁT trên val CỐ ĐỊNH.** Không còn nhãn để
chấm, nên `val_score = −val_loss`: dấu trừ giữ quy ước "cao hơn = tốt hơn" của
phần best/patience, số thật vẫn in ra dưới tên `val_loss`. Lượt val **không**
cập nhật center và **không** cập nhật EMA — cập nhật center ở đó là để tập val
lặng lẽ dạy model một thứ đi thẳng vào loss của bước train kế tiếp, đúng nghĩa
rò rỉ dù không có nhãn nào. τ_t ở val lấy giá trị **cuối**, không theo lịch
warm-up, vì τ đổi thì chính thang đo của loss đổi.

**Checkpoint Stage 0 chứa ĐÚNG trọng số Extractor, và là bản TEACHER (EMA)** —
`base/` và `final/` đều lưu `teacher_body`, kèm `weights: teacher_ema` trong meta
để các giai đoạn sau kiểm lại. Teacher là trung bình trượt của student nên nó êm
hơn, ổn định hơn, và **chính nó** là thứ sinh target suốt Stage 0 — dùng student
ở sau là dùng bản nhiễu hơn của cùng một encoder. DINO head thì đi ra **file
riêng** (`dino_head.safetensors`): Stage 0 không cần nó nữa, nhưng **Stage 1
thì cần**.

`base/` là checkpoint tốt nhất của phần theo lịch (đẩy Hub ngay khi phần ấy
xong); `final/` là checkpoint tốt nhất sau cả epoch đuôi — Stage 1 và main stage
dùng bản này. Không có đuôi thì `final/` trùng `base/` từng bit.

## 2.2. Stage 1 — Compressor transfer learning (`src/stage1_vextractor.py`)

| Mô tả | Chỉ huấn luyện **Compressor** với dữ liệu MIMIC-CXR hiện tại |
|---|---|
| Mục tiêu | Làm giàu không gian Compressor trước khi đưa vào Aligner → Concat |
| lr (Attn) | `lr_min=1e-8` · `lr_max=3e-5` |
| lr (FFN) | `lr_min=1e-7` · `lr_max=3e-4` |
| Epoch | `30` (trần); warm-up 1 epoch rồi **giữ** `lr_max` của từng nhóm |
| Chặng 5 | mỗi epoch val không cải thiện → **cả hai nhóm** ÷ `1,5` (sàn `lr_min` của từng nhóm) |
| Patience | `4` — đạt patience thì cosine decay từ lr **hiện tại** về `lr_min` thêm 1 epoch rồi mới lấy best val |
| Nhãn | DINOv3 Teacher (Extractor Stage 0) + Head Infer (Stage 0) sinh nhãn chuẩn DINOv3 |

```
view_t ─► Extractor(0) ─► pool ─► DINO head(0) ───────────► t_raw   [ĐỨNG YÊN]
view_s ─► Extractor(0) ─► Compressor ─► mean(V_c) ─► head MỚI ─► s_raw
loss = CE( softmax((t_raw − center)/τ_t) ‖ log_softmax(s_raw/τ_s) )
```

**KHÔNG có EMA ở Stage 1, và đó là khác biệt đáng đọc kỹ.** Ở Stage 0 teacher là
trung bình trượt của student, nên cặp EMA + centering là thứ giữ DINO khỏi sụp
về một điểm. Ở Stage 1 teacher là một **checkpoint**: với một ảnh, đích không đổi
suốt cả giai đoạn, và student không kéo được teacher theo mình — sụp không xảy
ra được. Centering và sharpening **vẫn giữ**, nhưng lý do đã hẹp lại: chúng định
*hình dạng* của đích chứ không còn chống sụp. Bỏ centering là để vài prototype
thống trị mọi ảnh; bỏ sharpening là học một phân phối gần đều, và 8 vector không
phải nén gì cả.

**HAI nhóm lr, và đó là điểm riêng của Stage 1.** Ma trận attention quyết định
8 vector **đọc ở đâu** trên lưới 784 ô, và đó là thứ dễ hỏng nhất khi bị đẩy
nhanh: một bước lớn làm phân phối attention sụp về vài ô rồi không quay lại. FFN
thì chỉ biến đổi thứ đã đọc được, nên nó chịu được một mức cao hơn — ở đây cao
hơn **10 lần** (3e-4 so với 3e-5). `latent` + mọi RMSNorm + DINO head mới đi
theo **nhóm FFN**: đặc tả chỉ khai hai nhóm, nên chỗ để chúng phải nói ra chứ
không để ngầm. Ở `N = 4` hai nhóm lệch nhau khá xa về khối lượng —
**27,53 M** attn so với **9,46 M** ffn — nên "hai nhóm" ở đây không phải hai
nửa bằng nhau.

**HAI view, và đúng hai**: teacher một global 448², student một global 448²
**khác** (augment khác). Multi-crop của Stage 0 vô nghĩa ở đây — teacher đứng
yên nên thêm view chỉ là tính lại cùng một đích; và `patches()` đòi đúng 784 ô
nên local crop 224² sẽ bị **từ chối ồn ào** chứ không lặng lẽ chạy trên lưới
thô hơn.

**`mean(V_c)` — trung bình 8 vector.** DINO head cần một vector mỗi ảnh, và 8
latent là một tập không có thứ tự ưu tiên nào (không có cls token trong đó). Lấy
vector đầu là chọn bừa một trong 8; lấy trung bình là hỏi "cả cụm 8 vector,
gộp lại, có mang đúng thứ teacher mang không".

**Mọi siêu tham số DINO** (τ_s/τ_t, centering, prototype, augment) mượn **nguyên**
của `stage0.dino` — khai lại ở đây là để hai giai đoạn chạy hai công thức khác
nhau mà không ai thấy. Khối `stage1` chỉ khai **lịch học**.

**Cái giá của Stage 1, và nó có thật.** Nó chỉ dạy Compressor bắt chước một phép
pool của teacher, nên nó **không** biết gì về báo cáo và không trả lời được "8
vector ấy có mang thông tin bệnh lý chưa". Thứ nó mua là một điểm khởi đầu
**không ngẫu nhiên** cho Compressor, để 200 bước của main stage không phải vừa
học nén ảnh vừa học viết báo cáo cùng lúc.

## 2.3. Main stage — Decoder finetuning (`src/train_vextractor.py`)

Dùng **lịch r32 của $1**: cùng fixture, thứ tự `randperm(seed)`, batch mỗi lượt
đã khoá, 1 epoch = 200 bước, LoRA one-cycle đỉnh 1e-4, wd 0.01, và cùng cách đo
val (CE trên 256 mẫu bốc lại mỗi 25 bước). **Riêng LoRA giữ nguyên cấu hình hiện
tại.** Phần thêm là **BỐN nhóm lr** trên hai khối:

| nhóm | lr | mốc |
|---|---|---|
| Aligner · FFN (SwiGLU + 2 RMSNorm) | `1e-6` → `1e-4` | warm-up 10% tổng bước rồi PHẲNG |
| Aligner · Attn (4 chiếu MHAttn) | `1e-7` → `1e-5` | warm-up 10% tổng bước rồi PHẲNG |
| Compressor · FFN | `1e-7` → `1e-5` | freeze 20% → warm-up 10% → PHẲNG |
| Compressor · Attn | `1e-8` → `1e-6` | freeze 20% → warm-up 10% → PHẲNG |

**Attn thấp hơn FFN đúng 10 lần ở CẢ HAI khối**, và lý do là một: ma trận
attention quyết định 8 vector đọc/trộn **ở đâu**, còn FFN chỉ biến đổi thứ đã
đọc xong. Đó chính là lý do đã tách hai nhóm ở Stage 1, dịch nguyên sang main
stage — `zcfg` **từ chối** cấu hình đảo dấu (`attn_lr_main > ffn_lr_main`), vì
đảo lại là đảo đúng lý do tách nhóm.

**Bốn nhóm ấy PHỦ KÍN hai khối.** `attn_parameters()` + `ffn_parameters()` của
mỗi khối là một **phân hoạch** (selftest so bằng `id()`), và `train` **dừng**
nếu tổng số tensor của bốn nhóm không bằng số tensor của Compressor + Aligner:
một tham số học được nằm ngoài optimizer là một tensor đứng yên suốt 200 bước
mà không có gì báo. Optimizer cũng đặt lr **theo TÊN nhóm**, không theo chỉ số
— chèn một nhóm vào giữa thì chỉ số đổi im lặng và hai khối đổi lịch cho nhau.

### a. Aligner

| Mục tiêu | Học cách biến tín hiệu đặc trưng thị giác sang không gian ngữ nghĩa |
|---|---|
| Lịch trình | Training **đồng thời với LoRA từ đầu** |

Aligner là ma trận ngẫu nhiên kẹp giữa một encoder đã dạy xong (Stage 0 + Stage
1) và một decoder đã dạy xong ($1). Bước Adam đầu tiên ở lr đầy đủ trên một ma
trận ngẫu nhiên là cách chắc chắn nhất để Q lệch hẳn khỏi phân phối mà LoRA đang
học cùng lúc — nên nó cần dốc, dù chỉ 20 bước. Cả hai nhóm của nó dùng **cùng
một dốc** (`align_warmup_frac`): chúng nằm trong cùng một khối ngẫu nhiên và
phải đi vào cùng một lúc.

### b. Compressor

| Mục tiêu | Học cách xử lý đặc trưng đã có theo tín hiệu từ LoRA → Aligner |
|---|---|
| Lịch trình | 1. Freeze hoàn toàn trong **20%** step đầu. 2. Unfreeze và warm-up trong **10%** step rồi fix cứng `lr_main` |

Tín hiệu mà Compressor phải học *theo* chưa tồn tại ở bước 0 — Aligner còn ngẫu
nhiên và LoRA chưa đi được đâu — nên 20% bước đầu là thời gian để nó xuất hiện.
`requires_grad=False` chứ không phải lr = 0: Compressor là khối học được **đầu
tiên** sau Extractor, nên tắt grad ở đó **cắt hẳn** phần backward bên dưới
Aligner — tiết kiệm thật, không chỉ là không cập nhật. Vì `requires_grad` tắt cả
khối, hai nhóm của nó **không thể** mở ở hai thời điểm khác nhau dù có muốn.

**Mốc khai theo TỈ LỆ, không theo số bước tuyệt đối.** Main stage khoá 1 epoch =
200 bước theo $1, nhưng `--dry-run` và mọi lần đổi `train_batch` đổi con số ấy.
Khai "40 bước" thì với một lần chạy 4 bước, Compressor không bao giờ mở — và
không có gì báo. Vì thế `MAIN_CFG_KEYS` **không có khoá `*_steps` nào**.

### c. Extractor

**Không bao giờ mở.** Nó nạp bản **teacher (EMA)** từ Stage 0 (`final/`) và chạy
ở **chế độ TEACHER INFERENCE** suốt main stage — đúng chế độ nó sinh target ở
Stage 0: `requires_grad=False` + `eval()` + forward trong `no_grad` + **không
nằm trong optimizer**. 0 bước cập nhật, 0 gradient, 0 activation được giữ.

### d. Phần còn lại

* Autobatch dò VRAM ở **đúng** trạng thái của vòng lặp.
* Clip gradient **tách riêng** LoRA và nhánh, để nhánh mới không làm nhỏ bước
  cập nhật của LoRA so với A.
* B nặng hơn A: **8 token** thêm vào chuỗi **cộng** một lượt
  FlashInternImage-L (3 stage, DCNv4) ở 448² cho mỗi ảnh. Lượt ấy chạy trong `no_grad` nên
  phần nặng còn lại là Cross-attn của 8 latent trên **784 ô**. Nếu batch đã khoá
  không vừa VRAM, micro-batch được chạy thành từng mảnh và CE được ghép lại theo
  token; kết quả đúng bằng con số HF tính trên cả micro-batch, nên gradient
  không đổi.
* **Stage 0 thì ngược lại nặng hơn hẳn**: multi-crop nhân số lượt forward lên
  (2 + `n_local`) cho student cộng 2 lượt `no_grad` cho teacher, nên
  `stage0.micro_batch` ở config GPU chỉ là khởi điểm.
* Khởi tạo nhánh chạy trong `fork_rng`, nên LoRA được khởi tạo từ đúng trạng
  thái RNG như A.
* Theo dõi `RMS(Q)/RMS(Z)` mỗi bước — độ lớn một chiều của Q so với một chiều
  của token ảnh, ghi vào `train_stats.json` cùng đường cong val.

---

# 3. Kiểm định (`src/stats_ab.py`)

Ghép cặp theo ảnh: d_i = điểm(B, i) − điểm(A, i), trên BLEU-1..4, ROUGE-1/2/L
và BERTScore-F1 (metric chính).

* **Paired bootstrap**: 10 000 lần, lấy lại mẫu theo ảnh. Cho CI95 BCa của
  mean(d) và p_boot.
* **Permutation test** đảo dấu: 10 000 lần, cho p_perm.
* Holm riêng từng họ trên các metric.
* **Thắng** khi cả hai kiểm định bác H0 sau Holm **và** |Cohen dz| ≥ 0.10.

REPORT.md ghi thêm: đồ thị `figures/metrics_ab.png`, số ảnh B thắng/hoà/thua,
Stage 0 và Stage 1 từng epoch, và bảng main stage của A và B.

---

# 4. Lens — mổ khối Q trên MỘT ảnh (`src/lens_vextractor.py`, lệnh `lens`)

Phần minh hoạ, tách khỏi kiểm định. Chọn **một ảnh test nhiều `pos` +
`uncertain` nhất** (`lens.sample: auto`) và dùng đúng ảnh đó cho cả ba phần.

**Chỉ chạy candidate B.** Z của B giống A *từng bit* (nhánh không đụng vào 256
token ảnh) và chuỗi chỉ dài thêm khối Q — nên **Q chính là phần `B − A`**.

Mọi bản đồ là đạo hàm của **một số** `y = Σ_t log p(ref_t | ref_<t, ảnh)`
(`lens.target: ref_logprob`), nên chỉ tốn một lượt backward.

1. **[Language] Độ nhạy token × lớp** — `S[ℓ,t] = ‖∂y/∂h_(ℓ,t)‖₂`, **MỘT dải
   ngang ba ô** (`lens_layers.png`): **Z** (256 token ảnh) · **Q** (8 latent,
   đánh dấu **từng `Q[k]`**) · **T** (phần văn bản). Trục Y là vị trí token với
   chỉ số thật, trục X là lớp, trên **đủ mọi lớp** (34/34).

   **Tách THANG MÀU, không tách FILE.** Phần văn bản nhạy hơn phần thị giác hàng
   bậc nên nếu dùng chung một `vmax` thì dải Q ra **đen tuyền**, và kết luận tự
   nhiên là *"decoder không đọc Q"* — trong khi cái thật sự xảy ra chỉ là một
   phép chuẩn hoá màu. Cái giá: không còn so được **độ sáng** giữa ba ô, nên câu
   *"Q mạnh yếu thế nào so với Z"* trả lời bằng **số**, không bằng mắt.

   **`T` là phần BÙ, không phải đuôi**: trong chuỗi của Gemma có văn bản ở cả hai
   phía — BOS + system prompt **trước** 256 token ảnh, câu hỏi + báo cáo **sau**
   khối Q.
2. **[Language] Logit lens** — `ŷ_ℓ = W_U · RMSNorm(h_(ℓ,last))` → top-k token,
   kèm bảng 14 nhãn bệnh lý của ảnh đó. Đây là chỗ duy nhất còn dùng
   `lens.layer_frac` (1/4 lớp cuối).
3. **[Visual] Bốn ô trên MỘT dải ngang — ảnh gốc · Z · khối Q · đầu ra
   Extractor**: (1) ảnh gốc; (2) **Z**, `‖∂y/∂X‖` dừng ở đầu ra encoder, lưới
   64²; (3) **khối Q**, `‖∂‖Q‖²/∂I‖` — target là **chính các High-Q token**,
   truy ngược `Q → Aligner → Compressor → Extractor → I` về tận pixel của ảnh
   448²; (4) **đầu ra Extractor**, `‖feature‖₂` của lưới teacher (28²).

   **Ô 4 KHÔNG phải gradient** — nó là *độ lớn đặc trưng*. **Cặp đáng đọc nhất
   là ô 4 so ô 3**: Extractor mã hoá cả ảnh, còn Compressor chỉ hỏi một phần —
   chênh lệch giữa hai ô chính là *phần Extractor có mà 8 latent không lấy*.

   Đường truy ngược chạy ở đúng chế độ của main stage: `_q_saliency_map` mở cờ
   đóng băng **chỉ** để nối đồ thị gradient từ pixel lên Q — `requires_grad` của
   **mọi** tham số vẫn bị ép về False, `eval()` giữ nó không dropout, và cờ cũ
   được trả lại ngay trong `finally`.

Đầu ra `runs/<subdir>/Lens/`: `Lens_Explain.md`, `figures/lens_layers.png` ·
`figures/lens_visual.png`, `logit_lens.csv`, `lens_maps.npz`, `lens.json`.

---

# 5. Chạy

### 0. Setup (MỘT LẦN trên mỗi máy)

```bash
bash scripts/setup.sh                       # tự dò container / cục bộ
RGEXP_REMOTE=1 bash scripts/setup.sh        # ép chế độ container (không dựng venv)
RGEXP_UPGRADE_TORCH=1 bash scripts/setup.sh # nâng torch (~2.5 GB) thay vì ghim transformers
```

`setup.sh` tự lo cả toán tử DCNv4: khi `vextractor.extractor.arch` là
`flash_intern_image_*` **và** máy có GPU, nó gọi `scripts/setup_dcnv4.sh`
(dựng `CUDA_HOME` từ wheel nếu box không có toolkit, dịch cho đúng compute cap,
rồi chạy thật kernel ở fp32/bf16/fp16). Bỏ qua bước ấy: `RGEXP_SKIP_DCNV4=1`.
Chạy lại một mình:

```bash
bash scripts/setup_dcnv4.sh
```

Lần chạy đầu còn tải `flash_intern_image_l_22k_384.pth` (~1,25 GB) từ
`OpenGVLab/DCNv4`. **Toán tử DCNv4 là bắt buộc — không có đường lùi PyTorch**
(xem §1.1); không dịch được thì đổi `arch` về `OpenGVLab/internimage_l_22k_384`
và mọi thứ chạy tiếp trên đường DCNv3.

### 1–4. Thí nghiệm

```bash
export HF_TOKEN=hf_xxxxxxxx          # gemma gated + WRITE vào RGExp123/Research_results
export WANDB_API_KEY=xxxxxxxx        # thiếu -> W&B tự ghi offline vào ./wandb

python scripts/selftest_vextractor.py             # ~30 kiểm tra, vài giây, không cần GPU
bash scripts/run_vextractor.sh train --dry-run    # 8 ảnh, 16 mẫu, Stage 0/1 ≤2 bước, main 4 bước

bash scripts/run_vextractor.sh train --detach     # 1. kéo A -> Stage 0 -> Stage 1 -> main B -> sinh -> chấm
bash scripts/run_vextractor.sh train --gpu rtx5060ti --detach   # ... trên RTX 5060 Ti 16 GB (Blackwell)
bash scripts/run_vextractor.sh stats              # 2. bootstrap + permutation -> REPORT.md (GHI ĐÈ)
bash scripts/run_vextractor.sh lens               # 3. mổ khối Q trên 1 ảnh -> Lens/Lens_Explain.md
bash scripts/run_vextractor.sh push               # 4. -> current/$2.2_V-Extractor_Test/
```

Nguồn của A, theo thứ tự ưu tiên: (1) máy đã chạy $1 (`adapters_lora/_fixture`
và `runs/lora_rank_target/`), (2) Hub `current/$1_LoRA_Test/`. Chỉ dùng khi tem
khớp: vân tay config, sha256 weight trên Hub, batch sinh, đủ 3000 dòng. Weight
của A không được tải, vì A không sinh lại.

`train` chạy lại sau khi đứt thì **đi tiếp**: Stage 0 có checkpoint mỗi epoch
(`_stage0/last/state.pt`); B có weight thì bỏ qua train; sinh thiếu thì sinh
tiếp. Các cờ:

* `--retrain-all` (hoặc `vextractor_exp.retrain_all: true`): huấn luyện lại
  **toàn bộ B** — Stage 0, Stage 1, main stage, sinh, chấm. A ($1) không đổi.
* `--restage0`: chạy lại TOÀN BỘ Stage 0, kéo theo `--restage1` và `--retrain`.
* `--restage1`: chạy lại Stage 1 (chuyển giao Compressor), kéo theo `--retrain`.
* `--stages data,stage0`: chỉ chạy Stage 0.
* `--retrain`: chỉ main stage.
* `--stages infer,score` · `--no-pull`.

---

# 6. Lưu trữ (bố cục như $1)

```
adapters_vextractor/_fixture/                  = fixture của $1 (vân tay trùng) + sha weight của B
adapters_vextractor/_stage0/                   STAGE 0 — DINO tự giám sát
    base/vextractor_stage0.safetensors|.json TRỌNG SỐ EXTRACTOR — TEACHER (EMA), tốt nhất của phần theo lịch
    base/dino_head.safetensors               DINO head TEACHER (Stage 1 cần)
    final/…                                  tốt nhất sau epoch đuôi — Stage 1 + main stage dùng bản này
    loader.json                              fixture + sha key, `labels: null` (TỰ GIÁM SÁT), thứ tự,
                                             batch/micro, lịch (warm-up + GIỮ lr_max, kèm số bước thật
                                             + khối `plateau` của chặng 5), `augment` + `dino`,
                                             spec, vân tay
    stage0_history.json                      từng epoch: train/val loss DINO, val_score, phase, lr, n_decays
    last/state.pt                            chỉ để chạy tiếp (không đẩy Hub)
adapters_vextractor/_stage1/                   STAGE 1 — chuyển giao Compressor
    compressor/vextractor_compressor.safetensors|.json   CHỈ Compressor (không Extractor, không Aligner)
    stage1_history.json                      từng epoch: val loss, lr_attn, lr_ffn, phase, n_decays
adapters_vextractor/qlora_r32_decoder_vextractor/   MAIN STAGE
    adapter_model.safetensors, adapter_config.json   LoRA r32
    vextractor_branch.safetensors|.json      TRỌN nhánh: Extractor (đóng băng) + Compressor + Aligner
    vextractor_stage0.safetensors|.json      bản chép Stage 0 đã nạp (truy nguồn)
    train_stats.json                         batch/accum/mảnh, lr, val curve, RMS(Q)/RMS(Z), VRAM, phút
runs/vextractor/                               REPORT.md, manifest.json, predictions/, metrics/, arms/, stage0/, stage1/
```

Trên Hub (`RGExp123/Research_results`):

```
current/$2.2_V-Extractor_Test/Reports/    REPORT.md, manifest, metrics/, predictions/ (A + B), arms/, stage0/
current/$2.2_V-Extractor_Test/fixture/    loader.json + split (= $1, thêm sha weight B)
current/$2.2_V-Extractor_Test/stage0/     base/, final/, loader.json, stage0_history.json
current/$2.2_V-Extractor_Test/stage1/     compressor/, stage1_history.json
current/$2.2_V-Extractor_Test/weights/qlora_r32_decoder_vextractor/
```

Weight của A vẫn ở `current/$1_LoRA_Test/weights/qlora_r32_decoder/`.

**Hai vân tay, hai phạm vi.** Vân tay Stage 0 chỉ đọc `vextractor.extractor`
(`arch`, `pretrained`, `n_layers`, `n_stages` + phép chuẩn hoá suy từ `arch`;
`with_cp` và `image_size` KHÔNG vào — chúng đổi chi phí, không đổi tensor nào),
khối `stage0` và `seed` — **không** gồm `compressor` / `aligner` / `main`, vì
Stage 0 không dựng những thứ ấy. Vân tay Stage 1 thêm **hình dạng Compressor**
và **vân tay Stage 0** (đổi teacher là đổi nhãn). Và cả hai cổng so **nội dung**
đã ghi, không so hash: sửa phạm vi vân tay không hồi tố được con số đã nằm trong
history, nên so hash là vứt mọi checkpoint đẩy lên trước lần sửa.

---

# 7. Cấu trúc

```
configs/vextractor.yaml · configs/gpu/{rtx3060,rtx5060ti}.yaml (stage0/stage1 micro_batch + probe_sizes)
requirements.txt · requirements-dcnv4.txt (toán tử DCNv4 — vì sao ở file riêng: đọc đầu file)
scripts/run_vextractor.sh · selftest_vextractor.py · setup.sh · setup_dcnv4.sh · _env.sh
src/vendor/             flash_intern_image.py NGUYÊN VĂN @ commit 4b848f7 (sha256 trong NOTICE.md)
                        — code model không có trong package pip; ĐỪNG sửa file trong đó
src/run_vextractor.py    train (kéo A → Stage 0 → Stage 1 → main B → sinh → chấm) · stats · lens · push
src/schedule.py          LỊCH DÙNG CHUNG: warmup_flat (dốc 1 epoch rồi GIỮ lr_max) ·
                         plateau_lr (chặng 5: ÷1,5 mỗi epoch val hỏng, sàn lr_min) ·
                         phase_of/phase_span · tail_lr · frozen_warmup_flat
                         — một chỗ duy nhất viết công thức
src/vextractor.py        Extractor — HAI đường dựng: InternImage (HF, trust_remote_code,
                         cắt .model.levels, NHWC) hoặc timm (ladder có probe forward thật,
                         NCHW/NHWC/NLC, cắt .stages) + Compressor (QFormer: 8 latent @768,
                         N nhóm × [SelfAttn → CrossAttn → SwiGLU │ SelfAttn → SwiGLU],
                         kv 640→768 trong Cross-attn) + Aligner (MHAttn(768) +
                         SwiGLU 768→2048→2560, init 1/√fan_in) · vext_tensor · ghi/nạp checkpoint
src/dino.py              DINO: multi-crop (PIL), DINO head, teacher EMA, center,
                         lịch τ_t / momentum · CompressorTransferTrainer (Stage 1)
src/stage0_vextractor.py Stage 0 — DINO tự giám sát (KHÔNG nạp Gemma, KHÔNG nhãn) + loader.json
src/stage1_vextractor.py Stage 1 — chuyển giao Compressor, HAI nhóm lr (attn | ffn)
src/train_vextractor.py  main stage của B (vòng lặp chép từ train_recomb._train_once, nối Q)
src/lens_vextractor.py   Lens (CHỈ B): dải 3 ô heatmap token×lớp (Z/Q/T) · logit lens ·
                         dải 4 ô (gốc / Z / khối Q / đầu ra Extractor)
src/report_vextractor.py REPORT.md
src/stats_ab.py          bootstrap ghép cặp (BCa) + permutation, Holm
src/runners/gemma3_vextractor.py sinh: A = [Z ‖ text], B = [Z ‖ Q ‖ text] (cùng đường)
src/recombiner.py        RMSNorm, encode_vision, build_inputs_embeds, inject_tokens,
                         image_end_indices / last_image_index (chỗ nối Q)
src/fixture.py · hub_sync.py · push.py · train_recomb.py · backbone.py …
```

## `_archive/`

* `_archive/recovery/`: toàn bộ §2.1 Recovery (Q = Recovery(X), rút từ chính X
  của SigLIP). README ở `_archive/README_2.1_Recovery.md`.
* `_archive/zrecomb/`: toàn bộ §2.0 Recombiner Test (Z → Z').
* `_archive/v2/`: toàn bộ $1 LoRA Test (rank × chỗ gắn), kèm bản chụp mọi module
  dùng chung đúng như lúc $1 chạy.
* §2.0–§2.5: xem `_archive/README_*.md`.
