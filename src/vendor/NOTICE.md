# `src/vendor/` — code bên thứ ba, ghim theo commit

Mọi file ở đây là bản chép **nguyên văn**. Không sửa; muốn đổi hành vi thì bọc
ở `src/vextractor.py`.

## `flash_intern_image.py`

| | |
|---|---|
| Nguồn | <https://github.com/OpenGVLab/DCNv4> — `classification/models/flash_intern_image.py` |
| Commit | `4b848f7dd7da74ff03f7d278f902c6fd05b391b5` |
| sha256 | `2a588890e8201ec9580455ce5d0e042aab97b82bc08f5e780c3b84a410733d16` |
| Dòng | 795 |
| Giấy phép | MIT — "Copyright (c) 2024 OpenGVLab" (xem header của file) |

Kiểm lại bản chép này đúng nguyên văn upstream:

```bash
curl -sSL https://raw.githubusercontent.com/OpenGVLab/DCNv4/4b848f7dd7da74ff03f7d278f902c6fd05b391b5/classification/models/flash_intern_image.py \
  | sha256sum
# 2a588890e8201ec9580455ce5d0e042aab97b82bc08f5e780c3b84a410733d16
```

### Vì sao phải vendor, chứ không `pip install`

Package `DCNv4` trên PyPI **chỉ chứa toán tử** (`DCNv4/modules/dcnv4.py`,
`DCNv4/functions/`, và extension CUDA). Code **model** FlashInternImage không
nằm trong package — nó ở thư mục `classification/` của repo GitHub, vốn không
phải một package cài được. Ba lựa chọn, và vì sao chọn lựa chọn này:

* **vendor bản ghim (đang dùng)** — đi theo git của repo này, không phụ thuộc
  mạng lúc chạy, đọc được khi debug, và `state_dict` sinh ra từ đúng những dòng
  đã được kiểm;
* clone repo lúc setup — thêm một bước mạng và một nguồn sự thật nằm ngoài git
  của repo này;
* viết lại — rủi ro lệch một chi tiết so với checkpoint, và sự lệch ấy KHÔNG
  ném lỗi: `load_state_dict(strict=False)` chỉ trả về một danh sách khoá thiếu
  mà không ai đọc.

### Phụ thuộc lúc chạy

File này `import DCNv4` **trần**, không có `try/except`, và
`DCNv4/functions/dcnv4_func.py` lại `from DCNv4 import ext` ở tầng module. Nên:

> **Không có đường lùi PyTorch.** Thiếu extension CUDA đã biên dịch thì
> `import DCNv4` chết ngay lúc import, không phải lúc forward.

Khác hẳn DCNv3 của InternImage, nơi `has_cuda_kernel` luôn `False` và bản
PyTorch luôn chạy được. Xem `vextractor.flash_internimage_body` cho thông báo
lỗi và cách cài.
