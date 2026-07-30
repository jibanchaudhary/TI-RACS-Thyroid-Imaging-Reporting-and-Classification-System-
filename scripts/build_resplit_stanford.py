"""
Clip-grouped, class-stratified re-split of the Stanford dataset.

The original split (data_pipeline/dataset_conversion.py:split_dataset_csv) shuffles
rows at the FRAME level, so frames from the same clip (video) scatter across
train/val/test — leaking every clip into every split. This script instead treats
each clip as one atomic, single-label unit and splits the *clips*, so no clip can
cross splits.

Reads   : stanford_dataset/{train,val,test}.csv        (left untouched)
Writes  : resplit_stanford_dataset/{train,val,test}.csv
          resplit_stanford_dataset/split_manifest.csv  (clip,label,split)
          resplit_stanford_dataset/split_report.txt

Notes
-----
- Clip id  = basename(img_path).split('_')[0]   (e.g. 138_69.jpg -> clip "138").
- TR-1 has only ONE clip in the whole dataset; a clip lives in exactly one split,
  so it is forced entirely into TRAIN. The split stays 5-class; TR-1 is simply
  unevaluable in val/test.
- Images (stanford_dataset/images) and boundary masks (stanford_dataset/med_sam)
  are the same frames regardless of split, so they are referenced in place — this
  script only reassigns rows, it does not copy pixel data.
"""
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------- config
REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "stanford_dataset"
OUT_DIR = REPO / "resplit_stanford_dataset"
SPLITS = ("train", "val", "test")

TEST_FRAC = 0.15  # fraction of (TR-2..TR-5) clips held out for test
VAL_FRAC = 0.15  # fraction of ALL clips targeted for val
SEED = 42
TR1_LABEL = "TR-1"  # the single-clip class, forced into train
EXPECTED_FRAMES = 17412  # sanity total across the three source CSVs


def clip_of(img_path: str) -> str:
    """138_69.jpg (or a/138_69.jpg) -> '138'."""
    return os.path.basename(str(img_path)).split("_")[0]


def load_full_frame_table() -> pd.DataFrame:
    frames = []
    for s in SPLITS:
        df = pd.read_csv(SRC_DIR / f"{s}.csv", dtype=str, keep_default_na=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    assert list(df.columns[:3]) == [
        "img_path",
        "label",
        "mask",
    ], f"unexpected columns: {list(df.columns)}"
    assert len(df) == EXPECTED_FRAMES, f"expected {EXPECTED_FRAMES} frames, got {len(df)}"
    df["clip"] = df["img_path"].map(clip_of)
    return df


def build_clip_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per clip: clip, label. Asserts each clip is single-label."""
    per_clip = df.groupby("clip")["label"].agg(["nunique", "first"])
    multi = per_clip[per_clip["nunique"] > 1]
    assert multi.empty, f"clips with >1 label (cannot group-split cleanly): {list(multi.index)}"
    return (
        per_clip.rename(columns={"first": "label"})[["label"]]
        .reset_index()
        .sort_values("clip")
        .reset_index(drop=True)
    )


def assign_splits(clips: pd.DataFrame) -> pd.DataFrame:
    """Return the clip table with a new 'split' column, stratified by label."""
    # TR-1: single clip -> forced into train, excluded from the stratified split.
    tr1 = clips[clips["label"] == TR1_LABEL].copy()
    tr1["split"] = "train"
    rest = clips[clips["label"] != TR1_LABEL].copy()

    # Peel off test (15% of rest), stratified on class.
    rest_idx, test_idx = train_test_split(
        rest.index,
        test_size=TEST_FRAC,
        random_state=SEED,
        stratify=rest["label"],
    )
    # Peel off val from the remainder. Target VAL_FRAC of ALL clips; convert to a
    # fraction of the post-test remainder.
    val_of_remainder = VAL_FRAC / (1.0 - TEST_FRAC)
    train_idx, val_idx = train_test_split(
        rest_idx,
        test_size=val_of_remainder,
        random_state=SEED,
        stratify=rest.loc[rest_idx, "label"],
    )

    rest.loc[train_idx, "split"] = "train"
    rest.loc[val_idx, "split"] = "val"
    rest.loc[test_idx, "split"] = "test"

    out = pd.concat([tr1, rest], ignore_index=True).sort_values("clip").reset_index(drop=True)
    assert out["split"].notna().all(), "some clip was left unassigned"
    return out


def verify(df: pd.DataFrame, clip_split: pd.DataFrame) -> None:
    """Fail loudly on any leakage or lost frame."""
    sets = {s: set(clip_split.loc[clip_split["split"] == s, "clip"]) for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = sets[a] & sets[b]
        assert not inter, f"clip leakage {a}∩{b}: {sorted(inter)}"

    # every frame accounted for exactly once
    merged = df.merge(clip_split[["clip", "split"]], on="clip", how="left", validate="many_to_one")
    assert merged["split"].notna().all(), "some frame's clip has no split"
    assert len(merged) == EXPECTED_FRAMES

    # TR-1 only in train
    tr1_splits = set(clip_split.loc[clip_split["label"] == TR1_LABEL, "split"])
    assert tr1_splits <= {"train"}, f"TR-1 leaked outside train: {tr1_splits}"


def render_report(df: pd.DataFrame, clip_split: pd.DataFrame) -> str:
    merged = df.merge(clip_split[["clip", "split"]], on="clip", how="left")
    labels = sorted(df["label"].unique())
    lines = ["Clip-grouped stratified re-split of the Stanford dataset", "=" * 58, ""]

    lines.append(f"{'':8s}{'clips':>8s}{'frames':>9s}")
    for s in SPLITS:
        n_clip = (clip_split["split"] == s).sum()
        n_frame = (merged["split"] == s).sum()
        lines.append(f"{s:8s}{n_clip:8d}{n_frame:9d}")
    lines.append(f"{'total':8s}{len(clip_split):8d}{len(merged):9d}")
    lines.append("")

    lines.append("CLIPS per class per split:")
    lines.append(f"{'label':8s}" + "".join(f"{s:>8s}" for s in SPLITS))
    for lab in labels:
        row = clip_split[clip_split["label"] == lab]
        counts = Counter(row["split"])
        lines.append(f"{lab:8s}" + "".join(f"{counts.get(s, 0):8d}" for s in SPLITS))
    lines.append("")

    lines.append("FRAMES per class per split:")
    lines.append(f"{'label':8s}" + "".join(f"{s:>8s}" for s in SPLITS))
    for lab in labels:
        row = merged[merged["label"] == lab]
        counts = Counter(row["split"])
        lines.append(f"{lab:8s}" + "".join(f"{counts.get(s, 0):8d}" for s in SPLITS))
    lines.append("")

    lines.append("Leakage check: train∩val, train∩test, val∩test all empty  ✓")
    lines.append("TR-1 (single clip) forced into: train  ✓")
    lines.append(f"seed={SEED}  test_frac={TEST_FRAC}  val_frac={VAL_FRAC}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_full_frame_table()
    clips = build_clip_table(df)
    clip_split = assign_splits(clips)
    verify(df, clip_split)

    merged = df.merge(clip_split[["clip", "split"]], on="clip", how="left")
    for s in SPLITS:
        out = merged.loc[merged["split"] == s, ["img_path", "label", "mask"]]
        out.to_csv(OUT_DIR / f"{s}.csv", index=False)
        print(
            f"{s:5s}: {len(out):6d} frames  {(clip_split['split'] == s).sum():4d} clips  -> {OUT_DIR / f'{s}.csv'}"
        )

    clip_split[["clip", "label", "split"]].to_csv(OUT_DIR / "split_manifest.csv", index=False)

    report = render_report(df, clip_split)
    (OUT_DIR / "split_report.txt").write_text(report)
    print("\n" + report)
    print(f"All outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
