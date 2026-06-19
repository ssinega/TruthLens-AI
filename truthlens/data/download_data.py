"""
TruthLens — Dataset Downloader
================================
Downloads fake news datasets from Kaggle (with HuggingFace fallbacks).

Supported datasets:
    welfake  — WELFake (72,134 articles, Kaggle: saurabhshahane/fake-news-classification)
    isot     — ISOT Fake & Real News (44,898 articles, Kaggle: clmentbisaillon/fake-and-real-news-dataset)
    liar     — LIAR benchmark (12,836 statements, HuggingFace: liar)

Usage:
    python data/download_data.py --dataset welfake
    python data/download_data.py --dataset isot
    python data/download_data.py --dataset liar
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("truthlens.data")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent

# ── Label maps ────────────────────────────────────────────────────────────────
# TruthLens 3-class: 0 = Fake, 1 = Suspicious, 2 = Real

# WELFake: 0 = Fake, 1 = Real → TruthLens: 0=Fake, 2=Real
_WELFAKE_LABEL_MAP = {0: 0, 1: 2}

# ISOT: Fake.csv=0, True.csv=2 (assigned directly)

# LIAR 6-class → TruthLens 3-class
_LIAR_LABEL_MAP = {
    "false":         0,   # Fake
    "pants-fire":    0,   # Fake (pants on fire)
    "barely-true":   1,   # Suspicious
    "half-true":     1,   # Suspicious
    "mostly-true":   2,   # Real
    "true":          2,   # Real
}


# ══════════════════════════════════════════════════════════════════════════════
# WELFake
# ══════════════════════════════════════════════════════════════════════════════

def download_welfake() -> pd.DataFrame:
    """
    Download WELFake dataset.

    Primary:  Kaggle → saurabhshahane/fake-news-classification
    Fallback: HuggingFace → fabiochiu/WELFake_Dataset

    Returns
    -------
    pd.DataFrame with columns: text, title, label  (label: 0=Fake, 2=Real)
    """
    # ── Try Kaggle first ──────────────────────────────────────────────────────
    try:
        import kagglehub  # type: ignore
        logger.info("Attempting WELFake download from Kaggle...")
        path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
        dataset_path = Path(path)

        # Find the CSV (WELFake_Dataset.csv or similar)
        csvs = list(dataset_path.glob("**/*.csv"))
        if not csvs:
            raise FileNotFoundError("No CSV found in Kaggle WELFake download.")
        df = pd.read_csv(csvs[0])

        # Standardize columns
        col_map = {}
        for c in df.columns:
            lc = c.lower()
            if "text" in lc and "text" not in col_map:
                col_map[c] = "text"
            elif "title" in lc and "title" not in col_map:
                col_map[c] = "title"
            elif "label" in lc and "label" not in col_map:
                col_map[c] = "label"
        df = df.rename(columns=col_map)

        if "label" in df.columns:
            df["label"] = df["label"].map(_WELFAKE_LABEL_MAP).fillna(0).astype(int)
        if "title" not in df.columns:
            df["title"] = ""

        df = df[["text", "title", "label"]].dropna(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 10].reset_index(drop=True)
        logger.info(f"WELFake (Kaggle) downloaded: {len(df):,} records.")
        return df

    except Exception as exc:
        logger.warning(f"Kaggle WELFake failed ({exc}). Trying HuggingFace...")

    # ── Fallback: HuggingFace ─────────────────────────────────────────────────
    try:
        from datasets import load_dataset as hf_load  # type: ignore
        logger.info("Downloading WELFake from HuggingFace (fabiochiu/WELFake_Dataset)...")
        dataset = hf_load("fabiochiu/WELFake_Dataset", split="train")
        df = dataset.to_pandas()

        if "text" not in df.columns and "statement" in df.columns:
            df["text"] = df["statement"]
        if "title" not in df.columns:
            df["title"] = ""
        if "label" in df.columns:
            df["label"] = df["label"].map(_WELFAKE_LABEL_MAP).fillna(0).astype(int)

        df = df[["text", "title", "label"]].dropna(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 10].reset_index(drop=True)
        logger.info(f"WELFake (HuggingFace) downloaded: {len(df):,} records.")
        return df

    except Exception as exc:
        logger.warning(f"HuggingFace WELFake failed ({exc}). Using synthetic data.")
        return generate_synthetic_dataset(n_samples=1000)


# ══════════════════════════════════════════════════════════════════════════════
# ISOT (Clement Bisaillon — Fake and Real News Dataset)
# ══════════════════════════════════════════════════════════════════════════════

def download_isot() -> pd.DataFrame:
    """
    Download ISOT Fake & Real News dataset.

    Primary:  Kaggle → clmentbisaillon/fake-and-real-news-dataset
              (Fake.csv label=0, True.csv label=2)
    Fallback: HuggingFace → GonzaloA/fake_news

    Returns
    -------
    pd.DataFrame with columns: text, title, label  (label: 0=Fake, 2=Real)
    """
    # ── Try Kaggle first ──────────────────────────────────────────────────────
    try:
        import kagglehub  # type: ignore
        logger.info("Attempting ISOT download from Kaggle...")
        path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
        dataset_path = Path(path)

        fake_csv = None
        real_csv = None
        for csv in dataset_path.glob("**/*.csv"):
            name_lc = csv.name.lower()
            if "fake" in name_lc:
                fake_csv = csv
            elif "true" in name_lc or "real" in name_lc:
                real_csv = csv

        if fake_csv is None or real_csv is None:
            raise FileNotFoundError(
                f"Could not find Fake.csv / True.csv in {dataset_path}. "
                f"Found: {[c.name for c in dataset_path.glob('**/*.csv')]}"
            )

        df_fake = pd.read_csv(fake_csv)
        df_fake["label"] = 0

        df_real = pd.read_csv(real_csv)
        df_real["label"] = 2

        df = pd.concat([df_fake, df_real], ignore_index=True)

        # ISOT columns: title, text, subject, date
        if "text" not in df.columns:
            # Try combining subject + title as text
            df["text"] = df.get("title", pd.Series([""] * len(df))).astype(str)
        if "title" not in df.columns:
            df["title"] = ""

        df = df[["text", "title", "label"]].dropna(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 10].reset_index(drop=True)
        logger.info(f"ISOT (Kaggle) downloaded: {len(df):,} records.")
        return df

    except Exception as exc:
        logger.warning(f"Kaggle ISOT failed ({exc}). Trying HuggingFace GonzaloA/fake_news...")

    # ── Fallback: HuggingFace ─────────────────────────────────────────────────
    try:
        from datasets import load_dataset as hf_load  # type: ignore
        logger.info("Downloading ISOT fallback from HuggingFace (GonzaloA/fake_news)...")
        dataset = hf_load("GonzaloA/fake_news", split="train")
        df = dataset.to_pandas()

        if "text" not in df.columns and "article" in df.columns:
            df["text"] = df["article"]
        if "title" not in df.columns:
            df["title"] = ""
        # GonzaloA uses label 0=Fake, 1=Real → map to TruthLens
        if "label" in df.columns:
            df["label"] = df["label"].map({0: 0, 1: 2}).fillna(0).astype(int)

        df = df[["text", "title", "label"]].dropna(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 10].reset_index(drop=True)
        logger.info(f"ISOT fallback (HuggingFace) downloaded: {len(df):,} records.")
        return df

    except Exception as exc:
        logger.warning(f"HuggingFace ISOT fallback failed ({exc}). Using synthetic data.")
        return generate_synthetic_dataset(n_samples=1000)


# ══════════════════════════════════════════════════════════════════════════════
# LIAR Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def download_liar() -> pd.DataFrame:
    """
    Download the LIAR benchmark dataset from HuggingFace.

    Maps 6 fine-grained truthfulness labels → TruthLens 3-class:
        false / pants-fire   → 0  (Fake)
        barely-true / half-true → 1  (Suspicious)
        mostly-true / true   → 2  (Real)

    Returns
    -------
    pd.DataFrame with columns: text, title, label
    """
    try:
        from datasets import load_dataset as hf_load  # type: ignore
        logger.info("Downloading LIAR dataset from HuggingFace (UKPLab/liar)...")
        dataset = hf_load("UKPLab/liar")

        frames = []
        for split_name in ["train", "validation", "test"]:
            if split_name in dataset:
                frames.append(dataset[split_name].to_pandas())
        df = pd.concat(frames, ignore_index=True)

        if "text" not in df.columns and "statement" in df.columns:
            df["text"] = df["statement"].astype(str)
        if "title" not in df.columns:
            df["title"] = df.get("context", pd.Series([""] * len(df))).astype(str)

        if "label_text" in df.columns:
            df["label"] = df["label_text"].astype(str).str.lower().map({
                "false statement": 0,
                "true statement": 2,
            })
        elif "labels" in df.columns:
            # UKPLab/liar uses 0=true, 1=false
            df["label"] = df["labels"].map({0: 2, 1: 0})
        elif "label" in df.columns:
            if df["label"].dtype == object:
                df["label"] = df["label"].str.lower().map(_LIAR_LABEL_MAP)
            else:
                liar_int_map = {
                    0: 0, 1: 1, 2: 2, 3: 2, 4: 1, 5: 0,
                }
                df["label"] = df["label"].map(liar_int_map)

        df["label"] = df["label"].fillna(1).astype(int)
        df = df[["text", "title", "label"]].dropna(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 5].reset_index(drop=True)
        logger.info(f"LIAR downloaded: {len(df):,} records.")
        return df

    except Exception as exc:
        logger.warning(f"HuggingFace LIAR failed ({exc}). Trying GitHub TSV mirror...")

    try:
        base_url = "https://raw.githubusercontent.com/tfs4/liar_dataset/master"
        frames = []
        for split_name in ["train", "valid", "test"]:
            url = f"{base_url}/{split_name}.tsv"
            df_split = pd.read_csv(url, sep="\t", header=None)
            frames.append(df_split)

        df = pd.concat(frames, ignore_index=True)
        df["text"] = df[2].astype(str)
        df["title"] = df[3].fillna("").astype(str)
        df["label"] = df[1].astype(str).str.lower().map(_LIAR_LABEL_MAP).fillna(1).astype(int)
        df = df[["text", "title", "label"]].dropna(subset=["text"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"].str.len() > 5].reset_index(drop=True)
        logger.info(f"LIAR (GitHub mirror) downloaded: {len(df):,} records.")
        return df

    except Exception as exc:
        logger.warning(f"LIAR download failed ({exc}). Using synthetic data.")
        return generate_synthetic_dataset(n_samples=1000)


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic Fallback
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate a synthetic fake news dataset for demo/fallback purposes.

    Parameters
    ----------
    n_samples : int

    Returns
    -------
    pd.DataFrame with columns: text, title, label
    """
    import random
    random.seed(42)
    np.random.seed(42)

    logger.info(f"Generating {n_samples} synthetic articles...")

    fake_templates = [
        "SHOCKING: {topic} exposed as massive government cover-up! You won't believe what they found!",
        "BREAKING: Miracle {topic} cure doctors don't want you to know about. Leaked documents reveal all.",
        "The elites are hiding the truth about {topic}. Bombshell revelation destroys mainstream narrative.",
        "ALERT: Scientists baffled by {topic}. Secret conspiracy involves world leaders.",
        "EXCLUSIVE: {topic} scandal reveals deep state corruption. Anonymous whistleblower speaks out!",
    ]
    suspicious_templates = [
        "New study suggests possible link between {topic} and health outcomes, though researchers caution more data needed.",
        "Unverified reports indicate {topic} may have significant implications, according to anonymous sources.",
        "Some experts claim {topic} findings are preliminary. Critics dispute the methodology used.",
        "Controversial claims about {topic} emerge from new research. Debate continues among specialists.",
        "Preliminary data shows {topic} trends, but scientists say more investigation is required.",
    ]
    real_templates = [
        "Peer-reviewed study published in Nature confirms {topic} findings across multiple cohort studies.",
        "Official government report outlines new {topic} policy framework, citing verified census data.",
        "According to WHO and CDC, {topic} recommendations are based on systematic review of evidence.",
        "Court documents filed in federal court reveal {topic} details, corroborated by multiple sources.",
        "Data from the Bureau of Statistics shows {topic} trends, verified by independent researchers.",
    ]

    topics = [
        "climate change", "vaccine safety", "election integrity",
        "economic policy", "public health measures", "AI regulation",
        "immigration policy", "crime statistics", "education funding",
        "renewable energy", "social media censorship", "pharmaceutical approvals"
    ]

    rows = []
    per_class = n_samples // 3
    remainder = n_samples - 2 * per_class

    for label, templates, count in [
        (0, fake_templates, per_class),
        (1, suspicious_templates, per_class),
        (2, real_templates, remainder),
    ]:
        for _ in range(count):
            topic = random.choice(topics)
            template = random.choice(templates)
            text = template.format(topic=topic)
            body_sentences = [
                f"This development concerning {topic} has drawn significant attention.",
                "Experts from various institutions have weighed in on the matter.",
                f"The implications of this for {topic} policy remain to be seen.",
                "Further analysis is expected in the coming weeks."
            ]
            full_text = text + " " + " ".join(random.sample(body_sentences, k=random.randint(2, 4)))
            rows.append({
                "text": full_text,
                "title": text[:80],
                "label": label
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Synthetic dataset created: {len(df)} records.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Dataset saved → {path}  ({len(df):,} rows)")


def load_dataset_csv(path: Path) -> pd.DataFrame:
    """
    Load cached dataset CSV.  Falls back to synthetic data if not found.

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame
    """
    if path.exists():
        logger.info(f"Loading cached dataset from {path}")
        return pd.read_csv(path)
    else:
        logger.warning(f"Dataset not found at {path}. Generating synthetic data.")
        df = generate_synthetic_dataset()
        save_dataset(df, path)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# Dataset registry
# ══════════════════════════════════════════════════════════════════════════════

DATASETS = {
    "welfake": {
        "fn":       download_welfake,
        "out":      DATA_DIR / "welfake_dataset.csv",
        "display":  "WELFake",
        "expected": 72134,
    },
    "isot": {
        "fn":       download_isot,
        "out":      DATA_DIR / "isot_dataset.csv",
        "display":  "ISOT Fake/Real News",
        "expected": 44898,
    },
    "liar": {
        "fn":       download_liar,
        "out":      DATA_DIR / "liar_dataset.csv",
        "display":  "LIAR Benchmark",
        "expected": 12836,
    },
    "all": {
        "fn":       None,
        "out":      DATA_DIR / "all_datasets_combined.csv",
        "display":  "All Supported Datasets",
        "expected": 72134 + 44898 + 12836,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthLens Dataset Downloader")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="welfake",
        help="Which dataset to download (default: welfake)",
    )
    args = parser.parse_args()

    info = DATASETS[args.dataset]
    logger.info(f"=== TruthLens Dataset Downloader — {info['display']} ===")

    if args.dataset == "all":
        frames = []
        for dataset_key in ["welfake", "isot", "liar"]:
            dataset_info = DATASETS[dataset_key]
            logger.info(f"Downloading {dataset_info['display']}...")
            df_part = dataset_info["fn"]()
            save_dataset(df_part, dataset_info["out"])
            frames.append(df_part.assign(source_dataset=dataset_key))

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        save_dataset(df, info["out"])
    else:
        df = info["fn"]()
        save_dataset(df, info["out"])

    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    logger.info(
        f"Done!  Saved {len(df):,} rows to {info['out']}\n"
        f"Next: python models/train_models.py --dataset {args.dataset}"
    )
