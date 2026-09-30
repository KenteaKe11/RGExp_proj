"""Code của BÊN THỨ BA, ghim theo commit. KHÔNG sửa file trong thư mục này.

Mỗi file ở đây là một bản chép **NGUYÊN VĂN** từ một repo upstream, kèm sha256
trong `NOTICE.md`. Giữ nguyên văn là điều kiện để làm được hai việc:

  * `diff` được với upstream khi họ sửa (một dòng bị sửa tay ở đây là một dòng
    không ai biết đã lệch);
  * nạp được checkpoint upstream — TÊN KHOÁ của `state_dict` sinh ra từ chính
    những dòng này, nên sửa một `nn.Module` là làm `load_state_dict` lệch, và
    `strict=False` ở đâu đó sẽ nuốt sự lệch ấy không một tiếng.

Muốn đổi hành vi thì bọc ở `src/vextractor.py`, không sửa ở đây.
"""
