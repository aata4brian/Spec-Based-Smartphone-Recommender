"""
Smartphone Recommender Web App - Full Fuzzy Mamdani Engine
=========================================================

File ini mempertahankan inti engine fuzzy dari fuzzy_smartphone_mamdani.py:
- pembacaan CSV aman
- ekstraksi fitur numerik
- processor scoring
- triangular dan trapezoidal membership function
- inferensi Mamdani: AND=min, OR=max, implication=min, aggregation=max
- defuzzifikasi centroid

Tambahan untuk web app:
- FastAPI endpoint /recommend
- CORS agar frontend HTML lokal bisa memanggil backend
- input preferensi user: budget, priority, min_storage
- rule-base diperluas tanpa menghapus rule asli
- output Top 20 smartphone siap divisualisasikan di frontend
"""

import argparse
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


# ==========================================================
# 1. Fungsi bantu pembacaan data
# ==========================================================

def read_csv_safely(path: str) -> pd.DataFrame:
    """Membaca CSV dengan beberapa opsi encoding agar tidak mudah error."""
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252", "ISO-8859-1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Gagal membaca file CSV. Error terakhir: {last_error}")


# ==========================================================
# 2. Ekstraksi angka dari teks
# ==========================================================

def extract_first_number(value) -> float:
    """Mengambil angka pertama dari teks, misalnya '6.1 inches' menjadi 6.1."""
    if pd.isna(value):
        return np.nan

    text = str(value).replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if not numbers:
        return np.nan

    return float(numbers[0])


def extract_max_number(value) -> float:
    """
    Mengambil angka terbesar dari teks.
    Cocok untuk RAM '8GB / 12GB' -> 12 dan kamera '50MP + 12MP' -> 50.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)

    if not numbers:
        return np.nan

    return max(float(num) for num in numbers)


def extract_price_usd(value) -> float:
    """Mengambil angka harga USD, misalnya 'USD 1,299' menjadi 1299."""
    return extract_first_number(value)


def extract_storage_gb(row: pd.Series) -> float:
    """Mengambil storage dari kolom Storage_GB jika tersedia, atau dari nama model/RAM sebagai fallback."""
    if "Storage_GB" in row and not pd.isna(row["Storage_GB"]):
        return float(row["Storage_GB"])

    candidates = []
    for col in ["Model Name", "RAM"]:
        if col in row and not pd.isna(row[col]):
            text = str(row[col]).replace(",", "")
            # Ambil pola umum 128GB, 256 GB, 1TB, dst.
            for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(TB|GB)", text, flags=re.I):
                value = float(num)
                if unit.lower() == "tb":
                    value *= 1024
                candidates.append(value)
    return max(candidates) if candidates else np.nan


# ==========================================================
# 3. Skoring processor
# ==========================================================

def processor_score(processor) -> float:
    """
    Mengubah nama processor menjadi skor 0-100.
    Skor ini berbasis aturan sederhana dari segmentasi umum chipset.
    """
    if pd.isna(processor):
        return 50.0

    p = str(processor).lower()

    # Apple Bionic / Apple Silicon
    apple_scores = {
        "a18": 98, "a17": 95, "a16": 92, "a15": 88, "a14": 82,
        "a13": 76, "a12z": 76, "a12": 70, "a11": 62,
    }
    for key, score in apple_scores.items():
        if key in p:
            return float(score)

    # Snapdragon
    if "snapdragon" in p or "sd " in p:
        if "elite" in p or "8 gen 3" in p or "8 gen 2" in p or "8 gen 1" in p:
            return 92.0
        if any(x in p for x in ["888", "870", "865", "855", "845"]):
            return 84.0
        if any(x in p for x in ["7 gen", "778", "780", "782", "765", "750"]):
            return 74.0
        if any(x in p for x in ["6 gen", "695", "690", "680", "675", "665", "660"]):
            return 60.0
        if any(x in p for x in ["4 gen", "480", "460", "450", "439", "435"]):
            return 45.0
        return 65.0

    # MediaTek Dimensity
    if "dimensity" in p:
        if any(x in p for x in ["9400", "9300", "9200", "9100", "9000"]):
            return 90.0
        if any(x in p for x in ["8400", "8300", "8200", "8100", "8000"]):
            return 82.0
        if any(x in p for x in ["1300", "1200", "1100"]):
            return 75.0
        if any(x in p for x in ["1080", "1050", "1000", "920", "900"]):
            return 66.0
        if any(x in p for x in ["800", "810", "720", "700", "610", "608", "6020"]):
            return 55.0
        return 62.0

    # Exynos
    if "exynos" in p:
        if "2400" in p:
            return 90.0
        if "2200" in p:
            return 82.0
        if "2100" in p:
            return 78.0
        if "1380" in p:
            return 65.0
        if "1280" in p or "1080" in p:
            return 60.0
        if "850" in p:
            return 45.0
        return 60.0

    # Google Tensor
    if "tensor" in p:
        if "g4" in p:
            return 88.0
        if "g3" in p:
            return 84.0
        if "g2" in p:
            return 80.0
        return 75.0

    # Kirin
    if "kirin" in p:
        if "9000" in p:
            return 85.0
        if "990" in p or "980" in p:
            return 78.0
        if "820" in p or "810" in p:
            return 62.0
        if "710" in p:
            return 50.0
        return 58.0

    # Helio
    if "helio" in p:
        if "g99" in p:
            return 60.0
        if any(x in p for x in ["g96", "g95", "g90"]):
            return 56.0
        if any(x in p for x in ["g88", "g85", "g80"]):
            return 50.0
        if any(x in p for x in ["g70", "g35", "g25", "p35", "p22"]):
            return 38.0
        return 45.0

    # Unisoc
    if "unisoc" in p:
        if "t618" in p or "t616" in p:
            return 45.0
        if "t606" in p or "t610" in p:
            return 38.0
        return 35.0

    return 50.0


# ==========================================================
# 4. Fuzzy membership function
# ==========================================================

def trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function."""
    if pd.isna(x):
        return 0.0

    x = float(x)

    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if b < x < c:
        return (c - x) / (c - b)

    return 0.0


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership function."""
    if pd.isna(x):
        return 0.0

    x = float(x)

    # Dibuat aman untuk shoulder set seperti (0,0,...) dan (...,100,100).
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a <= x < b:
        return (x - a) / (b - a) if b != a else 1.0
    if c < x <= d:
        return (d - x) / (d - c) if d != c else 1.0

    return 0.0


def fuzzify(row: pd.Series) -> Dict[str, Dict[str, float]]:
    """Mengubah nilai numerik menjadi derajat keanggotaan fuzzy."""
    ram = row["ram_gb"]
    battery = row["battery_mah"]
    camera = row["back_camera_mp"]
    price = row["price_usd"]
    year = row["launched_year"]
    processor = row["processor_score"]
    storage = row.get("storage_gb", np.nan)

    fuzzy = {
        "ram": {
            "low": trapmf(ram, 0, 0, 2, 4),
            "medium": trimf(ram, 3, 6, 8),
            "high": trapmf(ram, 6, 8, 16, 24),
        },
        "battery": {
            "small": trapmf(battery, 0, 0, 3000, 4000),
            "medium": trimf(battery, 3500, 4500, 5500),
            "large": trapmf(battery, 5000, 6000, 8000, 10000),
        },
        "camera": {
            "low": trapmf(camera, 0, 0, 12, 32),
            "medium": trimf(camera, 24, 50, 64),
            "high": trapmf(camera, 50, 108, 200, 250),
        },
        "price": {
            "cheap": trapmf(price, 0, 0, 250, 500),
            "medium": trimf(price, 350, 700, 1000),
            "expensive": trapmf(price, 800, 1200, 2000, 3000),
        },
        "year": {
            "old": trapmf(year, 2014, 2014, 2018, 2020),
            "recent": trimf(year, 2019, 2022, 2024),
            "new": trapmf(year, 2023, 2024, 2025, 2026),
        },
        "processor": {
            "entry": trapmf(processor, 0, 0, 35, 50),
            "mid": trimf(processor, 40, 60, 75),
            "flagship": trapmf(processor, 70, 85, 100, 100),
        },
        "storage": {
            "small": trapmf(storage, 0, 0, 64, 128),
            "medium": trimf(storage, 64, 128, 256),
            "large": trapmf(storage, 128, 256, 512, 1024),
        },
    }

    return fuzzy


def output_membership(y: np.ndarray, label: str) -> np.ndarray:
    """Membership function untuk output skor rekomendasi."""
    if label == "low":
        return np.array([trapmf(v, 0, 0, 30, 45) for v in y])
    if label == "medium":
        return np.array([trimf(v, 35, 55, 75) for v in y])
    if label == "high":
        return np.array([trapmf(v, 65, 80, 100, 100) for v in y])

    raise ValueError(f"Label output tidak dikenal: {label}")


# ==========================================================
# 5. Fuzzy Mamdani inference
# ==========================================================

def build_rule_base(f: Dict[str, Dict[str, float]], budget: str = "medium", priorities: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    """
    Membentuk rule base Mamdani.

    Bagian pertama adalah rule asli dari engine awal.
    Bagian berikutnya adalah ekspansi rule agar input budget, storage, dan priority user
    benar-benar memengaruhi ranking. Jumlah rule aktif/tersedia > 50 tanpa menghapus rule asli.
    """
    priorities = [p.strip().lower() for p in (priorities or [])]
    if not priorities:
        priorities = ["ram", "camera", "battery", "processor"]

    # Shortcut variabel agar rule mudah dibaca.
    ram_low = f["ram"]["low"]
    ram_med = f["ram"]["medium"]
    ram_high = f["ram"]["high"]

    bat_small = f["battery"]["small"]
    bat_med = f["battery"]["medium"]
    bat_large = f["battery"]["large"]

    cam_low = f["camera"]["low"]
    cam_med = f["camera"]["medium"]
    cam_high = f["camera"]["high"]

    price_cheap = f["price"]["cheap"]
    price_med = f["price"]["medium"]
    price_exp = f["price"]["expensive"]

    year_old = f["year"]["old"]
    year_recent = f["year"]["recent"]
    year_new = f["year"]["new"]

    proc_entry = f["processor"]["entry"]
    proc_mid = f["processor"]["mid"]
    proc_flagship = f["processor"]["flagship"]

    storage_small = f["storage"]["small"]
    storage_med = f["storage"]["medium"]
    storage_large = f["storage"]["large"]

    not_expensive = max(price_cheap, price_med)
    not_old = max(year_recent, year_new)
    capable_ram = max(ram_med, ram_high)
    capable_battery = max(bat_med, bat_large)
    capable_camera = max(cam_med, cam_high)
    capable_processor = max(proc_mid, proc_flagship)
    capable_storage = max(storage_med, storage_large)

    rules: List[Tuple[str, float]] = []

    # ------------------------------------------------------
    # A. Rule asli engine awal, tidak dihapus.
    # ------------------------------------------------------
    rules.append(("high", min(ram_high, proc_flagship)))
    rules.append(("high", min(ram_high, bat_large, max(price_cheap, price_med))))
    rules.append(("high", min(cam_high, proc_flagship, max(year_recent, year_new))))
    rules.append(("high", min(price_cheap, ram_high, bat_large)))
    rules.append(("high", min(year_new, proc_flagship)))
    rules.append(("high", min(ram_high, cam_high, max(price_cheap, price_med))))

    rules.append(("medium", min(ram_med, max(bat_med, bat_large), max(price_cheap, price_med))))
    rules.append(("medium", min(max(cam_med, cam_high), price_med, max(year_recent, year_new))))
    rules.append(("medium", min(proc_mid, ram_med, bat_med)))
    rules.append(("medium", min(price_exp, proc_flagship, ram_high)))
    rules.append(("medium", min(year_recent, proc_mid, max(ram_med, ram_high))))
    rules.append(("medium", min(price_cheap, proc_mid, bat_large)))

    rules.append(("low", min(ram_low, proc_entry)))
    rules.append(("low", min(price_exp, ram_low)))
    rules.append(("low", min(bat_small, year_old)))
    rules.append(("low", min(year_old, proc_entry)))
    rules.append(("low", min(price_exp, cam_low)))
    rules.append(("low", min(ram_low, bat_small)))
    rules.append(("low", min(price_exp, proc_entry)))

    # ------------------------------------------------------
    # B. Rule keseimbangan spesifikasi umum.
    # ------------------------------------------------------
    rules.extend([
        ("high", min(cappable, not_expensive, not_old))
        for cappable in [ram_high, bat_large, cam_high, proc_flagship, storage_large]
    ])
    rules.extend([
        ("medium", min(cappable, price_med, not_old))
        for cappable in [capable_ram, capable_battery, capable_camera, capable_processor, capable_storage]
    ])
    rules.extend([
        ("low", weak)
        for weak in [ram_low, bat_small, cam_low, proc_entry, storage_small]
    ])

    # Kombinasi performa tinggi.
    rules.extend([
        ("high", min(ram_high, proc_flagship, cam_high)),
        ("high", min(ram_high, proc_flagship, bat_large)),
        ("high", min(ram_high, proc_flagship, storage_large)),
        ("high", min(cam_high, proc_flagship, bat_large)),
        ("high", min(cam_high, storage_large, year_new)),
        ("high", min(storage_large, bat_large, not_expensive)),
        ("high", min(year_new, ram_high, capable_camera)),
        ("high", min(year_new, bat_large, capable_processor)),
        ("high", min(not_expensive, capable_ram, capable_processor, capable_battery)),
        ("high", min(price_cheap, capable_ram, capable_storage, capable_battery)),
    ])

    # Kombinasi menengah.
    rules.extend([
        ("medium", min(ram_med, proc_mid, capable_storage)),
        ("medium", min(cam_med, bat_med, price_med)),
        ("medium", min(storage_med, ram_med, not_old)),
        ("medium", min(proc_mid, capable_camera, not_expensive)),
        ("medium", min(year_recent, capable_battery, capable_storage)),
        ("medium", min(price_cheap, ram_med, proc_mid)),
        ("medium", min(price_exp, proc_flagship, year_new)),
        ("medium", min(price_exp, cam_high, ram_high)),
        ("medium", min(price_med, bat_large, storage_large)),
        ("medium", min(price_med, proc_flagship, storage_med)),
    ])

    # Kombinasi rendah.
    rules.extend([
        ("low", min(year_old, ram_low, bat_small)),
        ("low", min(year_old, storage_small)),
        ("low", min(price_exp, year_old)),
        ("low", min(price_exp, storage_small)),
        ("low", min(proc_entry, cam_low)),
        ("low", min(proc_entry, storage_small)),
        ("low", min(ram_low, cam_low)),
        ("low", min(bat_small, storage_small)),
        ("low", min(price_exp, bat_small)),
        ("low", min(year_old, cam_low)),
    ])

    # ------------------------------------------------------
    # C. Budget-specific rules.
    # ------------------------------------------------------
    if budget == "low":
        rules.extend([
            ("high", min(price_cheap, capable_ram, capable_battery)),
            ("high", min(price_cheap, capable_processor, capable_storage)),
            ("high", min(price_cheap, cam_high, not_old)),
            ("medium", min(price_med, ram_high, proc_flagship)),
            ("medium", min(price_med, cam_high, bat_large)),
            ("low", price_exp),
            ("low", min(price_exp, not_old)),
            ("low", min(price_exp, proc_mid)),
        ])
    elif budget == "high":
        rules.extend([
            ("high", min(max(price_med, price_exp), proc_flagship, ram_high)),
            ("high", min(max(price_med, price_exp), cam_high, year_new)),
            ("high", min(price_exp, storage_large, proc_flagship)),
            ("high", min(price_exp, bat_large, ram_high)),
            ("medium", min(price_cheap, capable_processor, capable_storage)),
            ("medium", min(price_med, capable_ram, capable_camera)),
            ("low", min(price_exp, ram_low)),
            ("low", min(price_exp, proc_entry)),
        ])
    else:  # medium
        rules.extend([
            ("high", min(price_med, capable_ram, capable_processor, capable_storage)),
            ("high", min(price_med, cam_high, bat_large)),
            ("high", min(price_cheap, ram_high, proc_flagship)),
            ("medium", min(price_cheap, capable_ram, capable_battery)),
            ("medium", min(price_exp, proc_flagship, ram_high)),
            ("medium", min(price_med, capable_camera, capable_storage)),
            ("low", min(price_exp, proc_entry)),
            ("low", min(price_exp, ram_low)),
        ])

    # ------------------------------------------------------
    # D. Priority-specific rules.
    # Semakin diprioritaskan user, semakin besar pengaruh fitur itu.
    # ------------------------------------------------------
    priority_map = {
        "ram": (ram_low, ram_med, ram_high),
        "camera": (cam_low, cam_med, cam_high),
        "battery": (bat_small, bat_med, bat_large),
        "processor": (proc_entry, proc_mid, proc_flagship),
        "storage": (storage_small, storage_med, storage_large),
    }
    for p in priorities:
        if p not in priority_map:
            continue
        low_v, med_v, high_v = priority_map[p]
        rules.extend([
            ("high", min(high_v, not_expensive, not_old)),
            ("high", min(high_v, capable_storage)),
            ("medium", min(med_v, not_old)),
            ("medium", min(high_v, price_exp)),
            ("low", low_v),
            ("low", min(low_v, price_exp)),
        ])

    # Prioritas ganda: pasangan fitur penting harus saling menguatkan.
    for i, p1 in enumerate(priorities):
        for p2 in priorities[i + 1:]:
            if p1 in priority_map and p2 in priority_map:
                _, _, high_1 = priority_map[p1]
                _, med_2, high_2 = priority_map[p2]
                rules.append(("high", min(high_1, high_2, not_old)))
                rules.append(("medium", min(high_1, med_2, not_expensive)))

    return rules


def infer_mamdani_score(row: pd.Series, budget: str = "medium", priorities: Optional[List[str]] = None) -> Tuple[float, str]:
    """
    Menghitung skor rekomendasi dengan metode Mamdani.
    Operator AND = minimum.
    Operator OR = maximum.
    Implikasi = minimum.
    Agregasi = maximum.
    Defuzzifikasi = centroid.
    """
    f = fuzzify(row)
    rules = build_rule_base(f, budget=budget, priorities=priorities)

    # Agregasi output
    y = np.linspace(0, 100, 1001)
    aggregated = np.zeros_like(y)

    for label, strength in rules:
        if strength <= 0:
            continue

        mf = output_membership(y, label)
        clipped = np.minimum(strength, mf)
        aggregated = np.maximum(aggregated, clipped)

    # Defuzzifikasi centroid
    denominator = np.sum(aggregated)
    if denominator == 0:
        score = fallback_score(row)
    else:
        score = float(np.sum(y * aggregated) / denominator)

    category = recommendation_category(score)
    return round(score, 2), category


def fallback_score(row: pd.Series) -> float:
    """Skor cadangan jika rule fuzzy tidak aktif."""
    ram = np.clip((row["ram_gb"] / 16) * 100, 0, 100)
    battery = np.clip((row["battery_mah"] / 7000) * 100, 0, 100)
    camera = np.clip((row["back_camera_mp"] / 200) * 100, 0, 100)
    processor = row["processor_score"]
    storage = np.clip((row.get("storage_gb", 0) / 512) * 100, 0, 100)
    year = np.clip(((row["launched_year"] - 2014) / (2025 - 2014)) * 100, 0, 100)

    # Harga dibalik: semakin murah semakin baik
    price = row["price_usd"]
    price_score = 100 - np.clip(((price - 100) / (2000 - 100)) * 100, 0, 100)

    return float(
        0.18 * ram
        + 0.14 * battery
        + 0.14 * camera
        + 0.20 * processor
        + 0.10 * storage
        + 0.12 * year
        + 0.12 * price_score
    )


def recommendation_category(score: float) -> str:
    if score >= 75:
        return "Sangat Direkomendasikan"
    if score >= 55:
        return "Direkomendasikan"
    if score >= 40:
        return "Cukup Direkomendasikan"
    return "Kurang Direkomendasikan"


# ==========================================================
# 6. Cleaning dan preprocessing
# ==========================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan dataset dan membuat kolom numerik yang dibutuhkan sistem."""
    data = df.copy()

    required_columns = [
        "Company Name",
        "Model Name",
        "Mobile Weight",
        "RAM",
        "Front Camera",
        "Back Camera",
        "Processor",
        "Battery Capacity",
        "Screen Size",
        "Launched Price (USA)",
        "Launched Year",
    ]

    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan dalam dataset: {missing}")

    # Hapus device non-smartphone berdasarkan nama model
    non_phone_keywords = r"ipad|tablet|tab|pad"
    is_non_phone = data["Model Name"].astype(str).str.lower().str.contains(non_phone_keywords, regex=True, na=False)
    data = data.loc[~is_non_phone].copy()

    # Ekstraksi fitur numerik. Kolom hasil cleaning bawaan dataset digunakan jika tersedia.
    data["ram_gb"] = data["RAM_GB"] if "RAM_GB" in data.columns else data["RAM"].apply(extract_max_number)
    data["front_camera_mp"] = data["Front Camera"].apply(extract_max_number)
    data["back_camera_mp"] = data["Back_Camera_MP"] if "Back_Camera_MP" in data.columns else data["Back Camera"].apply(extract_max_number)
    data["battery_mah"] = data["Battery_mAh"] if "Battery_mAh" in data.columns else data["Battery Capacity"].apply(extract_first_number)

    # Beberapa dataset menyimpan 3600 mAh sebagai 3.0 karena koma ribuan. Koreksi otomatis.
    data["battery_mah"] = pd.to_numeric(data["battery_mah"], errors="coerce")
    data.loc[data["battery_mah"].between(1, 20, inclusive="both"), "battery_mah"] *= 1000

    data["screen_size_inch"] = data["Screen_inch"] if "Screen_inch" in data.columns else data["Screen Size"].apply(extract_first_number)
    data["price_usd"] = data["Price_USD"] if "Price_USD" in data.columns else data["Launched Price (USA)"].apply(extract_price_usd)
    data["storage_gb"] = data.apply(extract_storage_gb, axis=1)
    data["launched_year"] = pd.to_numeric(data["Launched Year"], errors="coerce")
    data["processor_score"] = data["Processor"].apply(processor_score)

    numeric_features = [
        "ram_gb",
        "back_camera_mp",
        "battery_mah",
        "price_usd",
        "storage_gb",
        "launched_year",
        "processor_score",
    ]
    for col in numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=numeric_features).copy()
    return data


def add_recommendation_scores(data: pd.DataFrame, budget: str = "medium", priorities: Optional[List[str]] = None) -> pd.DataFrame:
    """Menambahkan skor dan kategori rekomendasi pada dataset."""
    scores = data.apply(lambda row: infer_mamdani_score(row, budget=budget, priorities=priorities), axis=1, result_type="expand")
    data = data.copy()
    data["fuzzy_score"] = scores[0]
    data["recommendation_category"] = scores[1]

    data = data.sort_values(
        by=["fuzzy_score", "launched_year", "processor_score", "ram_gb", "battery_mah", "storage_gb"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    data["rank"] = np.arange(1, len(data) + 1)
    return data


def select_output_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Memilih kolom output agar hasil lebih mudah dibaca."""
    columns = [
        "rank",
        "Company Name",
        "Model Name",
        "RAM",
        "ram_gb",
        "Processor",
        "processor_score",
        "Back Camera",
        "back_camera_mp",
        "Battery Capacity",
        "battery_mah",
        "storage_gb",
        "Screen Size",
        "screen_size_inch",
        "Launched Price (USA)",
        "price_usd",
        "Launched Year",
        "launched_year",
        "fuzzy_score",
        "recommendation_category",
    ]
    available = [col for col in columns if col in data.columns]
    return data[available]


def normalize_for_radar(row: pd.Series) -> Dict[str, float]:
    """Nilai 0-100 untuk Chart.js radar chart."""
    return {
        "RAM": round(float(np.clip((row["ram_gb"] / 16) * 100, 0, 100)), 2),
        "Camera": round(float(np.clip((row["back_camera_mp"] / 200) * 100, 0, 100)), 2),
        "Battery": round(float(np.clip((row["battery_mah"] / 7000) * 100, 0, 100)), 2),
        "Processor": round(float(np.clip(row["processor_score"], 0, 100)), 2),
        "Storage": round(float(np.clip((row["storage_gb"] / 512) * 100, 0, 100)), 2),
    }


def row_to_recommendation(row: pd.Series) -> Dict[str, object]:
    return {
        "Rank": int(row["rank"]),
        "Brand": str(row["Company Name"]),
        "Model": str(row["Model Name"]),
        "Score": round(float(row["fuzzy_score"]), 2),
        "Category": str(row["recommendation_category"]),
        "RAM": float(row["ram_gb"]),
        "Camera": float(row["back_camera_mp"]),
        "Battery": int(round(float(row["battery_mah"]))),
        "Processor": round(float(row["processor_score"]), 2),
        "ProcessorName": str(row["Processor"]),
        "Storage": int(round(float(row["storage_gb"]))),
        "Price": round(float(row["price_usd"]), 2),
        "Year": int(row["launched_year"]),
        "Radar": normalize_for_radar(row),
    }


# ==========================================================
# 7. FastAPI Web App
# ==========================================================

class RecommendationRequest(BaseModel):
    budget: str = Field(default="medium", description="low|medium|high")
    priority: List[str] = Field(default_factory=lambda: ["RAM", "Camera", "Battery", "Processor"])
    min_storage: int = Field(default=128, ge=0)

    @field_validator("budget")
    @classmethod
    def validate_budget(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"low", "medium", "high"}:
            raise ValueError("budget harus low, medium, atau high")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, values: List[str]) -> List[str]:
        allowed = {"ram", "camera", "battery", "processor", "storage"}
        cleaned = []
        for item in values:
            key = item.lower().strip()
            if key not in allowed:
                raise ValueError(f"priority tidak valid: {item}. Pilihan: RAM, Camera, Battery, Processor, Storage")
            cleaned.append(key)
        return cleaned or ["ram", "camera", "battery", "processor"]


class RecommendationResponse(BaseModel):
    status: str
    total_data: int
    total_after_filter: int
    recommendations: List[Dict[str, object]]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "Mobiles_Cleaned.csv"
DATASET_PATH = Path(os.getenv("DATASET_PATH", str(DEFAULT_DATASET_PATH)))


@lru_cache(maxsize=1)
def load_clean_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")
    raw = read_csv_safely(str(DATASET_PATH))
    return clean_dataset(raw)


def apply_budget_filter(data: pd.DataFrame, budget: str) -> pd.DataFrame:
    """Filter harga bertahap. Kalau terlalu sempit, sistem akan fallback agar Top 20 tetap bisa muncul."""
    if budget == "low":
        filtered = data[data["price_usd"] <= 500]
    elif budget == "medium":
        filtered = data[(data["price_usd"] >= 300) & (data["price_usd"] <= 1000)]
    else:
        filtered = data[data["price_usd"] >= 700]

    # Jangan sampai hasil kosong gara-gara filter terlalu kaku. Hidup sudah cukup kaku.
    return filtered if len(filtered) >= 20 else data


app = FastAPI(
    title="Smartphone Recommender - Fuzzy Mamdani",
    version="1.0.0",
    description="Web API rekomendasi smartphone berbasis Fuzzy Inference System metode Mamdani.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "OK",
        "message": "Smartphone Recommender API aktif. Gunakan POST /recommend.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    try:
        data = load_clean_data()
        return {"status": "OK", "dataset_rows": int(len(data)), "dataset_path": str(DATASET_PATH)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/meta")
def meta():
    data = load_clean_data()
    scored = add_recommendation_scores(data.copy())
    return {
        "status": "OK",
        "brands": sorted(scored["Company Name"].dropna().astype(str).unique().tolist()),
        "categories": sorted(scored["recommendation_category"].dropna().astype(str).unique().tolist()),
        "total_data": int(len(scored)),
        "rule_base_minimum": ">50 rules including original engine rules + preference-aware rules",
    }


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest):
    try:
        data = load_clean_data().copy()
        filtered = data[data["storage_gb"] >= payload.min_storage].copy()
        filtered = apply_budget_filter(filtered, payload.budget)

        if filtered.empty:
            return {
                "status": "EMPTY",
                "total_data": int(len(data)),
                "total_after_filter": 0,
                "recommendations": [],
            }

        scored = add_recommendation_scores(filtered, budget=payload.budget, priorities=payload.priority)
        top = scored.head(20).copy()
        top["rank"] = np.arange(1, len(top) + 1)

        return {
            "status": "OK",
            "total_data": int(len(data)),
            "total_after_filter": int(len(scored)),
            "recommendations": [row_to_recommendation(row) for _, row in top.iterrows()],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ==========================================================
# 8. CLI tetap tersedia seperti file original
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sistem rekomendasi smartphone berbasis Fuzzy Inference System metode Mamdani."
    )
    parser.add_argument("--input", required=True, help="Path file CSV dataset.")
    parser.add_argument("--output", default="hasil_rekomendasi_smartphone.csv", help="Path file output CSV.")
    parser.add_argument("--top", type=int, default=20, help="Jumlah ranking teratas yang ditampilkan di terminal.")
    parser.add_argument("--budget", choices=["low", "medium", "high"], default="medium", help="Preferensi budget.")
    parser.add_argument("--priority", nargs="*", default=["RAM", "Camera", "Battery", "Processor"], help="Prioritas fitur.")
    parser.add_argument("--min-storage", type=int, default=128, help="Storage minimum GB.")

    args = parser.parse_args()

    raw = read_csv_safely(args.input)
    cleaned = clean_dataset(raw)
    cleaned = cleaned[cleaned["storage_gb"] >= args.min_storage].copy()
    cleaned = apply_budget_filter(cleaned, args.budget)
    result = add_recommendation_scores(cleaned, budget=args.budget, priorities=args.priority)
    output = select_output_columns(result)

    output.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n=== RINGKASAN DATA ===")
    print(f"Jumlah data awal       : {len(raw)}")
    print(f"Jumlah data smartphone : {len(output)}")
    print(f"Budget                 : {args.budget}")
    print(f"Priority               : {', '.join(args.priority)}")
    print(f"Minimum storage        : {args.min_storage} GB")
    print(f"Output tersimpan       : {args.output}")

    print(f"\n=== TOP {args.top} REKOMENDASI SMARTPHONE ===")
    preview_cols = [
        "rank", "Company Name", "Model Name", "ram_gb", "storage_gb", "processor_score",
        "back_camera_mp", "battery_mah", "price_usd", "launched_year",
        "fuzzy_score", "recommendation_category"
    ]
    print(output[preview_cols].head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
