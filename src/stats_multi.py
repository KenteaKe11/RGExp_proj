"""§2.5 — BA candidate, BA cặp, hai kiểm định ghép cặp trên từng cặp.

Module này KHÔNG cài lại phép tính nào. `paired_bootstrap`, `permutation_paired`,
`cohen_dz`, `holm`, `align_blocks` đều import nguyên văn từ `src/stats_ab.py` —
đường đã cho ra bảng §2.3 và đã có selftest đối chiếu với scipy. Thứ duy nhất
đổi ở đây là **họ hiệu chỉnh Holm**, và đó đúng là thứ k = 3 làm đổi.

── VÌ SAO KHÔNG GỌI THẲNG `stats_ab.compare_all` BA LẦN ────────────────────
`compare_all` chạy Holm TRONG MỘT CẶP, trên các metric. Gọi nó ba lần rồi
hiệu chỉnh thêm một lần nữa là hiệu chỉnh HAI LẦN: mỗi p đã bị phạt theo số
metric, rồi bị phạt lại theo số cặp, và bảng kết quả trở nên bảo thủ theo một
cách không ai khai báo.

Ở §2.5 câu hỏi là "trên metric NÀY, ba cặp cái nào khác nhau" — nên họ Holm
đúng là **ba cặp trong một metric**, chạy riêng cho từng metric và riêng cho
từng kiểm định (permutation và bootstrap là hai họ, vì hai p của cùng một cặp
trả lời cùng một câu hỏi trên cùng dữ liệu).

`stats.adjust_scope` khai điều đó ra thay vì để nó là một quy ước ngầm:
    pairs   — Holm trên 3 cặp, trong từng metric   (mặc định của §2.5)
    metrics — Holm trên các metric, trong từng cặp (hành vi của §2.3)
    all     — Holm trên toàn bộ 3 × n_metric p

── DẤU CỦA HIỆU SỐ ─────────────────────────────────────────────────────────
Mọi hiệu số đọc là `B − A` với (A, B) là cặp đã sắp. `stats.reference` đẩy
candidate tham chiếu về phía A. Ở §2.6 tham chiếu là baseline gốc, nên cột
`mean_diff` dương nghĩa là **candidate hơn baseline** (B − A, A = baseline). Đó là một quy ước, và nó
được in thẳng vào REPORT.md để không ai phải đoán chiều.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .stats_ab import (align_blocks, cohen_dz, holm, materiality_warning,
                       paired_bootstrap, permutation_paired,
                       resolution_warning)
from .metrics import chexbert_micro_f1

__all__ = ["align_blocks", "pair_list", "compare_pairs", "summary_lines",
           "resolution_warning"]


def _chexbert_bootstrap_scores(df: pd.DataFrame, sample_indices: np.ndarray) -> np.ndarray:
    return chexbert_micro_f1(
        df["chexbert_tp"].to_numpy(dtype=np.float64)[sample_indices],
        df["chexbert_fp"].to_numpy(dtype=np.float64)[sample_indices],
        df["chexbert_fn"].to_numpy(dtype=np.float64)[sample_indices], axis=1)


def _chexbert_permutation_p(a: pd.DataFrame, b: pd.DataFrame,
                            sample_indices: np.ndarray, observed: float,
                            n_perm: int, rng: np.random.Generator,
                            chunk: int = 64) -> float:
    """Paired randomization test on the mean of the shared subset micro-F1s."""
    cols = ("chexbert_tp", "chexbert_fp", "chexbert_fn")
    ac = np.stack([a[c].to_numpy(dtype=np.float64) for c in cols], axis=1)
    bc = np.stack([b[c].to_numpy(dtype=np.float64) for c in cols], axis=1)
    # (n_subsets, subset_size, 3), in (TP, FP, FN) order.
    av, bv = ac[sample_indices], bc[sample_indices]
    n = ac.shape[0]
    exceed = 0
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        swap = rng.integers(0, 2, size=(m, n), dtype=np.int8).astype(bool)
        chosen = swap[:, sample_indices]  # (m, n_subsets, subset_size)
        aa = np.where(chosen[..., None], bv[None, ...], av[None, ...]).sum(axis=2)
        bb = np.where(chosen[..., None], av[None, ...], bv[None, ...]).sum(axis=2)
        # Counts are already summed over the 1,000 sampled images. Preserve
        # the bootstrap axis here: (permutation chunk, 4 subsets), then average
        # the four subset-level micro-F1 values for each permutation.
        def micro_f1(counts):
            tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
            den = 2.0 * tp + fp + fn
            return np.divide(2.0 * tp, den, out=np.zeros_like(den), where=den > 0)

        fa = micro_f1(aa)
        fb = micro_f1(bb)
        permuted = (fb - fa).mean(axis=1)
        exceed += int(np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-15))
        done += m
    return float((exceed + 1) / (n_perm + 1))


def pair_list(arm_ids: list[str], mode: str = "all",
              reference: str | None = None,
              contrasts: list | None = None) -> list[tuple[str, str]]:
    """Các cặp cần kiểm, đã sắp sao cho `reference` luôn đứng ở vị trí A.

    `mode = vs_reference` bỏ cặp không chứa tham chiếu. `mode = contrasts` chỉ
    kiểm các cặp KHAI TRƯỚC trong `stats.contrasts` (mỗi cặp một câu hỏi của
    bài; phần tử đầu là A, phần tử sau là B, hiệu số B − A). Cặp có candidate
    không chạy lần này (vd. `--arms` lọc bớt) bị bỏ qua; không còn cặp nào thì
    DỪNG, vì một bảng kiểm định rỗng đọc như "không có khác biệt".
    """
    ids = list(arm_ids)
    mode = str(mode or "all").lower()
    if mode not in ("all", "vs_reference", "contrasts"):
        raise ValueError(
            f"stats.pairs = {mode!r}; hợp lệ: 'all' | 'vs_reference' | "
            f"'contrasts'.")
    if mode == "contrasts":
        out = []
        for item in contrasts or []:
            if len(item) != 2 or item[0] == item[1]:
                raise ValueError(f"stats.contrasts: cặp sai: {item!r}")
            a, b = str(item[0]), str(item[1])
            if a in ids and b in ids and (a, b) not in out:
                out.append((a, b))
        if not out:
            raise ValueError(
                "stats.pairs = contrasts nhưng không cặp nào trong "
                "stats.contrasts có đủ cả hai candidate đã chấm điểm "
                f"({ids}).")
        return out
    if reference is not None and reference not in ids:
        raise ValueError(
            f"stats.reference = {reference!r} không nằm trong các candidate có "
            f"điểm ({ids}). Tham chiếu quyết định DẤU của mọi hiệu số, nên một "
            f"giá trị sai ở đây làm cả bảng đọc ngược mà vẫn hợp lệ.")
    out = []
    for a, b in combinations(ids, 2):
        if mode == "vs_reference" and reference not in (a, b):
            continue
        if reference is not None and b == reference:
            a, b = b, a
        out.append((a, b))
    return out


def compare_pairs(scores: dict[str, pd.DataFrame], metrics: list[str],
                  cfg_stats: dict, arm_ids: list[str],
                  seed: int = 42) -> pd.DataFrame:
    """Một dòng cho mỗi (cặp × metric). Hiệu số là B − A."""
    alpha = float(cfg_stats.get("alpha", 0.05))
    n_perm = int(cfg_stats.get("n_perm", 10000))
    n_boot = int(cfg_stats.get("n_boot", 10000))
    boot_size = int(cfg_stats.get("bootstrap_sample_size", 0))
    ci_method = str(cfg_stats.get("ci_method", "bca"))
    dz_floor = float(cfg_stats.get("dz_floor", 0.10))
    scope = str(cfg_stats.get("adjust_scope", "pairs")).lower()
    pairs = pair_list(arm_ids, cfg_stats.get("pairs", "all"),
                      cfg_stats.get("reference"), cfg_stats.get("contrasts"))

    # MỘT generator cho cả bảng, tiêu tuần tự theo (cặp, metric). Một generator
    # mới cho mỗi cặp nghĩa là ba cặp dùng CÙNG dãy hoán vị — các p-value khi ấy
    # tương quan với nhau theo một cách không ai khai báo, và Holm giả định
    # ngược lại.
    rng = np.random.default_rng(seed)
    n_rows = len(next(iter(scores.values())))
    sample_size = boot_size or n_rows
    if sample_size < 2:
        raise ValueError("stats.bootstrap_sample_size must be >= 2")
    # One independently seeded draw per bootstrap subset. Reuse each resulting
    # subset across candidates, pairs, and metrics for paired comparisons.
    configured_seeds = cfg_stats.get("bootstrap_seeds")
    if configured_seeds is None:
        bootstrap_seeds = [int(seed) + i for i in range(n_boot)]
    else:
        bootstrap_seeds = [int(s) for s in configured_seeds]
        if len(bootstrap_seeds) != n_boot:
            raise ValueError(
                "stats.bootstrap_seeds must contain exactly stats.n_boot seeds")
        if len(set(bootstrap_seeds)) != len(bootstrap_seeds):
            raise ValueError("stats.bootstrap_seeds must all be different")
    sample_indices = np.stack([
        np.random.default_rng(boot_seed).integers(0, n_rows, size=sample_size)
        for boot_seed in bootstrap_seeds
    ])

    rows = []
    for arm_a, arm_b in pairs:
        for m in metrics:
            a = scores[arm_a][m].to_numpy(dtype=np.float64)
            b = scores[arm_b][m].to_numpy(dtype=np.float64)
            d = b - a
            has_chexbert_counts = all(
                c in scores[arm_a].columns and c in scores[arm_b].columns
                for c in ("chexbert_tp", "chexbert_fp", "chexbert_fn"))
            if m == "chexbert_f1" and has_chexbert_counts:
                boot_a = _chexbert_bootstrap_scores(scores[arm_a], sample_indices)
                boot_b = _chexbert_bootstrap_scores(scores[arm_b], sample_indices)
                boot_d = boot_b - boot_a
                observed = float(boot_d.mean())
                centered = boot_d - observed
                p_boot = float(min(1.0, (np.count_nonzero(
                    np.abs(centered) >= abs(observed) - 1e-15) + 1) / (len(boot_d) + 1)))
                p_perm = _chexbert_permutation_p(
                    scores[arm_a], scores[arm_b], sample_indices, observed,
                    n_perm, rng)
                n_tie = int(np.count_nonzero(d == 0.0))
                rows.append({
                    "pair": f"{arm_b}_vs_{arm_a}", "arm_a": arm_a, "arm_b": arm_b,
                    "metric": m, "n": d.size,
                    "mean_a": float(boot_a.mean()), "mean_b": float(boot_b.mean()),
                    "mean_diff": observed,
                    "rel_diff_pct": (100.0 * observed / abs(float(boot_a.mean()))
                                     if boot_a.mean() else float("nan")),
                    "ci_lo": float(np.quantile(boot_d, alpha / 2)),
                    "ci_hi": float(np.quantile(boot_d, 1 - alpha / 2)),
                    "ci_method": "percentile over shared bootstrap F1s",
                    "boot_se": float(boot_d.std(ddof=1)) if len(boot_d) > 1 else float("nan"),
                    "p_perm": p_perm, "p_boot": p_boot,
                    "cohen_dz": cohen_dz(d),
                    "win_rate_b": float(np.mean(boot_d > 0)),
                    "win_b": int(np.count_nonzero(boot_d > 0)),
                    "win_a": int(np.count_nonzero(boot_d < 0)),
                    "tie": int(np.count_nonzero(boot_d == 0.0)),
                    "n_tied": n_tie, "frac_tied": n_tie / max(1, d.size),
                })
                continue
            bs = paired_bootstrap(d, n_boot, rng, alpha=alpha, method=ci_method,
                                  sample_indices=sample_indices)
            boot_mean_a = float(a[sample_indices].mean())
            boot_mean_b = float(b[sample_indices].mean())
            pm = permutation_paired(d, n_perm, rng)
            n_tie = int(np.count_nonzero(d == 0.0))
            rows.append({
                "pair": f"{arm_b}_vs_{arm_a}", "arm_a": arm_a, "arm_b": arm_b,
                "metric": m, "n": d.size,
                "mean_a": boot_mean_a, "mean_b": boot_mean_b,
                "mean_diff": bs["mean_diff"],
                "rel_diff_pct": (100.0 * bs["mean_diff"] / abs(a.mean())
                                 if a.mean() else float("nan")),
                "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                "ci_method": bs["ci_method"], "boot_se": bs["boot_se"],
                "p_perm": pm["p_perm"], "p_boot": bs["p_boot"],
                "cohen_dz": cohen_dz(d),
                "win_rate_b": float(np.mean(d > 0)),
                "win_b": int(np.count_nonzero(d > 0)),
                "win_a": int(np.count_nonzero(d < 0)),
                "tie": n_tie, "n_tied": n_tie,
                "frac_tied": n_tie / max(1, d.size),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ── Holm, theo họ đã khai ──────────────────────────────────────────────
    if scope == "pairs":
        group = df["metric"]
    elif scope == "metrics":
        group = df["pair"]
    elif scope == "all":
        group = pd.Series("all", index=df.index)
    else:
        raise ValueError(
            f"stats.adjust_scope = {scope!r}; hợp lệ: 'pairs' | 'metrics' | "
            f"'all'. Không có mặc định ngầm ở đây — họ hiệu chỉnh quyết định "
            f"bao nhiêu dòng sống sót, và một mặc định lặng lẽ là một bảng "
            f"p-value không tái lập được.")
    for col in ("p_perm", "p_boot"):
        adj = np.empty(len(df), dtype=np.float64)
        for _key, idx in df.groupby(group).groups.items():
            pos = df.index.get_indexer(idx)
            adj[pos] = holm(df.loc[idx, col].to_numpy())
        df[f"{col}_holm"] = adj

    # Quy ước §2.3: ý nghĩa = permutation p thô dưới alpha VÀ CI bootstrap
    # không chứa 0. p bootstrap không phải cổng kết luận; với bốn subset, p_boot
    # đặc biệt thô và không được dùng để chặn kết quả.
    df["sig_perm"] = df["p_perm"] < alpha
    df["sig_boot"] = df["p_boot_holm"] < alpha
    df["material"] = df["cohen_dz"].abs() >= dz_floor
    df["ci_excludes_zero"] = (df["ci_lo"] > 0) | (df["ci_hi"] < 0)
    df["sig_ci"] = df["ci_excludes_zero"]
    df["significant"] = df["sig_perm"] & df["sig_ci"]
    df["significant_and_material"] = df["significant"] & df["material"]
    df["winner"] = np.where(
        ~df["significant_and_material"], "—",
        np.where(df["mean_diff"] > 0, df["arm_b"], df["arm_a"]))
    df["dz_warning"] = [
        materiality_warning(int(r.n), float(r.cohen_dz), dz_floor)
        if bool(r.significant) else None for r in df.itertuples(index=False)]
    df["adjust_scope"] = scope
    # Bí danh cho phía báo cáo — `report_recomb.normalize_pw` của §2.3 đọc
    # `diff`/`better`, và giữ chúng ở đây nghĩa là một bảng CSV của §2.5 vẫn mở
    # được bằng công cụ cũ mà không phải đổi tên cột.
    df["diff"] = df["mean_diff"]
    df["better"] = df["winner"]
    return df


def summary_lines(df: pd.DataFrame, primary: str, names: dict[str, str]) -> list[str]:
    """Một câu kết luận cho MỖI cặp, trên metric chính."""
    out = []
    sub = df[df["metric"] == primary]
    if sub.empty:
        return [f"Không có metric chính `{primary}` trong bảng."]
    for r in sub.itertuples(index=False):
        na = names.get(r.arm_a, r.arm_a)
        nb = names.get(r.arm_b, r.arm_b)
        head = (f"`{primary}` · {nb} − {na} = {r.mean_diff:+.4f} "
                f"(95% CI [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], dz={r.cohen_dz:.3f})")
        if not bool(r.significant):
            out.append(f"{head} — KHÔNG có ý nghĩa theo quy ước §2.3 "
                       f"(p_perm={r.p_perm:.3g}; CI loại 0={bool(r.ci_excludes_zero)}).")
        elif not bool(r.material):
            out.append(f"{head} — có ý nghĩa thống kê nhưng |dz| dưới sàn: ở "
                       f"n={int(r.n)} đó là điều phải chờ đợi, không phải một "
                       f"phát hiện.")
        else:
            out.append(f"{head} — **{names.get(r.winner, r.winner)} thắng**, cả "
                       f"p permutation < α và CI bootstrap loại 0 "
                       f"(thắng {100 * r.win_rate_b:.1f}% số ảnh).")
    return out
