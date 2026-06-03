"""
compute_complexity.py
Tính FLOPs và Params cho UNetCOD (UNet3Plus_B3_BEM_CBAM)

Yêu cầu:
    pip install fvcore

Sử dụng:
    python compute_complexity.py
    python compute_complexity.py --model UNet3Plus_B3_BEM_CBAM --img_size 352
"""

import argparse
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count

from model_registry import create_model


# ===================== Configuration =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Tính FLOPs và Params cho UNetCOD")
    parser.add_argument("--model",    type=str, default="UNet3Plus_B3_BEM_CBAM",
                        help="Tên model trong model_registry")
    parser.add_argument("--img_size", type=int, default=352,
                        help="Kích thước ảnh đầu vào (mặc định: 352)")
    parser.add_argument("--device",   type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose",  action="store_true",
                        help="In chi tiết FLOPs theo từng module con")
    return parser.parse_args()


# ===================== Helper =====================

def fmt_m(n):   return f"{n/1e6:.4f} M"
def fmt_g(n):   return f"{n/1e9:.4f} G"
def get(d, k):  return d.get(k, 0)


# ===================== Main =====================

def main():
    args = parse_args()
    H = W = args.img_size

    print("=" * 65)
    print(f"  Model    : {args.model}")
    print(f"  Input    : 1 × 3 × {H} × {W}")
    print(f"  Device   : {args.device}")
    print("=" * 65)

    # ── Tạo model ở chế độ suy luận (không có nhánh boundary) ──────────
    # predict_boundary=False để FLOPs/Params phản ánh đúng inference
    model = create_model(args.model, args.device)
    model.eval()

    # Nếu model có thuộc tính predict_boundary, tắt đi để đo inference
    if hasattr(model, "predict_boundary"):
        model.predict_boundary = False
    if hasattr(model, "backbone") and hasattr(model.backbone, "predict_boundary"):
        model.backbone.predict_boundary = False

    dummy = torch.zeros(1, 3, H, W, device=args.device)

    # ── FLOPs ───────────────────────────────────────────────────────────
    flops = FlopCountAnalysis(model, dummy)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)

    by_mod = flops.by_module()

    # ── Params ──────────────────────────────────────────────────────────
    def params_of(module):
        return sum(p.numel() for p in module.parameters())

    # Truy cập encoder, cbam, decoder, bem, seg_head
    # Hỗ trợ cả cấu trúc trực tiếp và cấu trúc bọc qua .backbone
    root = model.backbone if hasattr(model, "backbone") else model

    def safe_get_module(names):
        """Thử lần lượt các tên, trả về module đầu tiên tìm thấy."""
        for name in names:
            obj = root
            found = True
            for part in name.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    found = False
                    break
            if found:
                return obj
        return None

    enc  = safe_get_module(["encoder"])
    bem  = safe_get_module(["bem"])
    seg  = safe_get_module(["seg_head", "segmentation_head"])

    cbam_mods = [safe_get_module([f"cbam{i}", f"cbam_stage{i}"]) for i in range(1, 6)]
    dec_mods  = [safe_get_module([f"decoder{i}"]) for i in range(1, 5)]

    # Params
    enc_p   = params_of(enc)  if enc  else 0
    cbam_p  = [params_of(m)   if m else 0 for m in cbam_mods]
    dec_p   = [params_of(m)   if m else 0 for m in dec_mods]
    bem_p   = params_of(bem)  if bem  else 0
    seg_p   = params_of(seg)  if seg  else 0
    total_p = params_of(model)

    # BEM với boundary head (training only)
    if bem is not None and hasattr(bem, "boundary_head") and bem.boundary_head is not None:
        bhead_p = params_of(bem.boundary_head)
    else:
        bhead_p = 0
    bem_train_p = bem_p + bhead_p  # nếu boundary_head đã tắt thì bhead_p=0

    # FLOPs — key theo tên module trong đồ thị
    prefix = "backbone." if hasattr(model, "backbone") else ""
    enc_f  = get(by_mod, f"{prefix}encoder")
    cbam_f = [get(by_mod, f"{prefix}cbam{i}") for i in range(1, 6)]
    dec_f  = [get(by_mod, f"{prefix}decoder{i}") for i in range(1, 5)]
    bem_f  = get(by_mod, f"{prefix}bem")
    seg_f  = get(by_mod, f"{prefix}seg_head") or get(by_mod, f"{prefix}segmentation_head")
    total_f = flops.total()

    # ── In bảng Params ─────────────────────────────────────────────────
    SEP = "-" * 65
    print(f"\n{'PARAMS':^65}")
    print(SEP)
    print(f"  {'Thành phần':<42} {'Params':>10}  {'%':>6}")
    print(SEP)
    print(f"  {'Bộ mã hoá (EfficientNet-B3)':<42} {fmt_m(enc_p):>10}  {enc_p/total_p*100:>5.1f}%")

    print(f"  {'Mô-đun CBAM':<42}")
    cbam_chs = [40, 32, 48, 136, 384]
    for i, (p, ch) in enumerate(zip(cbam_p, cbam_chs), 1):
        print(f"    {'e'+str(i)+f' ({ch} kênh)':<40} {fmt_m(p):>10}  {p/total_p*100:>5.2f}%")
    print(f"    {'Tổng CBAM':<40} {fmt_m(sum(cbam_p)):>10}  {sum(cbam_p)/total_p*100:>5.2f}%")

    print(f"  {'Bộ giải mã UNet3+':<42}")
    for i, p in enumerate(dec_p, 1):
        print(f"    {'Decoder '+str(i)+' (d'+str(i)+')':<40} {fmt_m(p):>10}  {p/total_p*100:>5.2f}%")
    print(f"    {'Tổng bộ giải mã':<40} {fmt_m(sum(dec_p)):>10}  {sum(dec_p)/total_p*100:>5.2f}%")

    print(f"  {'BEM (suy luận)':<42} {fmt_m(bem_p):>10}  {bem_p/total_p*100:>5.2f}%")
    if bhead_p > 0:
        print(f"  {'BEM (huấn luyện, +boundary head)':<42} {fmt_m(bem_train_p):>10}")
    print(f"  {'Đầu phân vùng (Conv 1×1)':<42} {fmt_m(seg_p):>10}  {seg_p/total_p*100:>5.2f}%")
    print(SEP)
    print(f"  {'TỔNG (suy luận)':<42} {fmt_m(total_p):>10}  100.0%")
    print(SEP)

    # ── In bảng FLOPs ──────────────────────────────────────────────────
    print(f"\n{'FLOPs (MACs)':^65}")
    print(SEP)
    print(f"  {'Thành phần':<42} {'GFLOPs':>10}  {'%':>6}")
    print(SEP)
    print(f"  {'Bộ mã hoá (EfficientNet-B3)':<42} {fmt_g(enc_f):>10}  {enc_f/total_f*100:>5.1f}%")

    print(f"  {'Mô-đun CBAM':<42}")
    sizes = [H//4, H//8, H//16, H//32, H//32]
    for i, (f, ch, s) in enumerate(zip(cbam_f, cbam_chs, sizes), 1):
        print(f"    {'e'+str(i)+f' ({ch}ch, {s}×{s})':<40} {fmt_g(f):>10}  {f/total_f*100:>5.2f}%")
    print(f"    {'Tổng CBAM':<40} {fmt_g(sum(cbam_f)):>10}  {sum(cbam_f)/total_f*100:>5.2f}%")

    print(f"  {'Bộ giải mã UNet3+':<42}")
    dec_sizes = [H//32, H//16, H//8, H//4]
    for i, (f, s) in enumerate(zip(dec_f, dec_sizes), 1):
        print(f"    {'Decoder '+str(i)+f' ({s}×{s})':<40} {fmt_g(f):>10}  {f/total_f*100:>5.2f}%")
    print(f"    {'Tổng bộ giải mã':<40} {fmt_g(sum(dec_f)):>10}  {sum(dec_f)/total_f*100:>5.2f}%")

    print(f"  {'BEM ('+str(H)+'×'+str(W)+')':<42} {fmt_g(bem_f):>10}  {bem_f/total_f*100:>5.2f}%")
    print(f"  {'Đầu phân vùng ('+str(H)+'×'+str(W)+')':<42} {fmt_g(seg_f):>10}  {seg_f/total_f*100:>5.2f}%")
    print(SEP)
    print(f"  {'TỔNG':<42} {fmt_g(total_f):>10}  100.0%")
    print(SEP)

    # ── Verbose: chi tiết theo module con ──────────────────────────────
    if args.verbose:
        print(f"\n{'CHI TIẾT PARAMS (fvcore)':^65}")
        print(parameter_count_table(model, max_depth=4))

        print(f"\n{'CHI TIẾT FLOPs THEO MODULE':^65}")
        print(SEP)
        for k, v in sorted(by_mod.items(), key=lambda x: -x[1]):
            if v > 0 and k != "":
                print(f"  {k:<50} {fmt_g(v):>10}")
        print(SEP)


if __name__ == "__main__":
    main()