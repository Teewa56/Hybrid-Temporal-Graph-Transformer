import uuid
import random
import numpy as np
from dataclasses import dataclass
from datetime import datetime, date

from synthetic_data_generator.config import CONFIG, NIGERIAN_STATES


# Nigerian ID document specifications (calibrated from NIMC/FRSC standards)
DOCUMENT_SPECS = {
    "NIN_slip": {
        "dpi_mean": 300, "dpi_std": 5,
        "file_size_kb_mean": 180, "file_size_kb_std": 30,
        "color_depth": 24,
        "compression_ratio_mean": 0.85, "compression_ratio_std": 0.03,
    },
    "drivers_license": {
        "dpi_mean": 600, "dpi_std": 10,
        "file_size_kb_mean": 350, "file_size_kb_std": 50,
        "color_depth": 24,
        "compression_ratio_mean": 0.90, "compression_ratio_std": 0.02,
    },
    "voters_card": {
        "dpi_mean": 200, "dpi_std": 15,
        "file_size_kb_mean": 120, "file_size_kb_std": 25,
        "color_depth": 24,
        "compression_ratio_mean": 0.80, "compression_ratio_std": 0.05,
    },
    "international_passport": {
        "dpi_mean": 600, "dpi_std": 5,
        "file_size_kb_mean": 500, "file_size_kb_std": 60,
        "color_depth": 24,
        "compression_ratio_mean": 0.92, "compression_ratio_std": 0.02,
    },
}

FONT_METRICS = {
    "NIN_slip":               {"font_size": 10, "font_family_hash": 0.42, "line_spacing": 1.15},
    "drivers_license":        {"font_size": 8,  "font_family_hash": 0.71, "line_spacing": 1.00},
    "voters_card":            {"font_size": 11, "font_family_hash": 0.55, "line_spacing": 1.20},
    "international_passport": {"font_size": 9,  "font_family_hash": 0.88, "line_spacing": 1.05},
}


@dataclass
class DocumentMetadata:
    doc_id: str
    doc_type: str
    dpi: float
    file_size_kb: float
    color_depth: int
    compression_ratio: float
    font_size: float
    font_family_hash: float
    line_spacing: float
    exif_creation_date: str
    exif_modification_date: str
    biometric_hash: float          # Normalized face feature hash
    issuing_state: str
    issue_year: int
    expiry_year: int
    pixel_uniformity: float        # High = suspiciously clean (AI-generated)
    noise_level: float             # Low noise = suspicious
    label: int = 0                 # 0=legit, 1=forged


def _sample_biometric_hash() -> float:
    """Realistic biometric hash — normally distributed around 0.5."""
    return float(np.clip(np.random.normal(0.5, 0.1), 0.1, 0.9))


class DocumentMetadataGenerator:
    """
    Generates synthetic document metadata for NIN slips, BVN documents,
    driver's licenses, and international passports.
    Legitimate documents follow tight statistical distributions based
    on known Nigerian ID document specifications.
    """

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.kyc.random_seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_one(self, doc_type: str = None) -> DocumentMetadata:
        doc_type = doc_type or random.choice(list(DOCUMENT_SPECS.keys()))
        spec = DOCUMENT_SPECS[doc_type]
        fonts = FONT_METRICS[doc_type]

        dpi = max(72, np.random.normal(spec["dpi_mean"], spec["dpi_std"]))
        file_size = max(10, np.random.normal(spec["file_size_kb_mean"], spec["file_size_kb_std"]))
        compression = float(np.clip(
            np.random.normal(spec["compression_ratio_mean"], spec["compression_ratio_std"]),
            0.5, 1.0
        ))

        # EXIF dates: creation before modification, modification recent
        issue_year = random.randint(2018, 2024)
        expiry_year = issue_year + random.choice([5, 10])
        creation_date = datetime(issue_year, random.randint(1, 12), random.randint(1, 28))
        mod_date = creation_date  # Legit docs: EXIF modification = creation

        return DocumentMetadata(
            doc_id=str(uuid.uuid4())[:12],
            doc_type=doc_type,
            dpi=round(dpi, 1),
            file_size_kb=round(file_size, 1),
            color_depth=spec["color_depth"],
            compression_ratio=round(compression, 4),
            font_size=fonts["font_size"] + random.uniform(-0.2, 0.2),
            font_family_hash=fonts["font_family_hash"] + random.uniform(-0.01, 0.01),
            line_spacing=fonts["line_spacing"] + random.uniform(-0.02, 0.02),
            exif_creation_date=creation_date.strftime("%Y:%m:%d %H:%M:%S"),
            exif_modification_date=mod_date.strftime("%Y:%m:%d %H:%M:%S"),
            biometric_hash=_sample_biometric_hash(),
            issuing_state=random.choice(NIGERIAN_STATES),
            issue_year=issue_year,
            expiry_year=expiry_year,
            pixel_uniformity=float(np.clip(np.random.normal(0.15, 0.05), 0.01, 0.5)),
            noise_level=float(np.clip(np.random.normal(0.25, 0.05), 0.05, 0.6)),
            label=0,
        )

    def generate_batch(self, n: int = None) -> list[DocumentMetadata]:
        n = n or CONFIG.kyc.n_legitimate_docs
        return [self.generate_one() for _ in range(n)]

    def to_feature_vector(self, doc: DocumentMetadata) -> np.ndarray:
        """Convert document metadata to a fixed-size feature vector (128-dim)."""
        doc_type_enc = [0.0] * len(DOCUMENT_SPECS)
        doc_types = list(DOCUMENT_SPECS.keys())
        if doc.doc_type in doc_types:
            doc_type_enc[doc_types.index(doc.doc_type)] = 1.0

        features = [
            doc.dpi / 600.0,
            doc.file_size_kb / 600.0,
            doc.color_depth / 32.0,
            doc.compression_ratio,
            doc.font_size / 14.0,
            doc.font_family_hash,
            doc.line_spacing / 2.0,
            doc.biometric_hash,
            doc.pixel_uniformity,
            doc.noise_level,
            (doc.expiry_year - doc.issue_year) / 10.0,
            (2025 - doc.issue_year) / 10.0,
        ] + doc_type_enc

        features += [0.0] * (CONFIG.kyc.feature_dim - len(features))
        return np.array(features[:CONFIG.kyc.feature_dim], dtype=np.float32)