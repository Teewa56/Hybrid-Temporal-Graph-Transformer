import random
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from synthetic_data_generator.config import CONFIG
from synthetic_data_generator.kyc.document_metadata_generator import (
    DocumentMetadata,
    DocumentMetadataGenerator,
    DOCUMENT_SPECS,
)


class ForgerySimulator:
    """
    Generates labeled forged document metadata by injecting anomalies
    into legitimate document records.

    Forgery types:
    - dpi_inconsistency      : DPI claims 300 but file size suggests 72
    - exif_mismatch          : EXIF modification date before creation date
    - font_anomaly           : Font metrics deviate from spec
    - biometric_divergence   : Face hash inconsistent with document identity
    - ai_generated           : Suspiciously perfect pixel uniformity, zero noise
    """

    FORGERY_TYPES = list(CONFIG.kyc.forgery_type_distribution.keys())

    def __init__(self, seed: int = None):
        seed = seed or CONFIG.kyc.random_seed
        random.seed(seed)
        np.random.seed(seed)
        self.legit_gen = DocumentMetadataGenerator(seed=seed)

    def inject_dpi_inconsistency(self, doc: DocumentMetadata) -> DocumentMetadata:
        """
        Claims high DPI but file size is consistent with a low-DPI scan.
        Real scanners at 300 DPI produce much larger files than at 72 DPI.
        """
        d = deepcopy(doc)
        # Claim 300 DPI but file size is more like 72 DPI
        d.dpi = DOCUMENT_SPECS[doc.doc_type]["dpi_mean"]
        d.file_size_kb = random.uniform(20, 60)   # Too small for claimed DPI
        d.label = 1
        d.doc_id = d.doc_id + "_FORGED"
        return d

    def inject_exif_mismatch(self, doc: DocumentMetadata) -> DocumentMetadata:
        """EXIF modification timestamp predates creation — impossible."""
        d = deepcopy(doc)
        creation = datetime.strptime(d.exif_creation_date, "%Y:%m:%d %H:%M:%S")
        # Modification before creation = red flag
        earlier = creation - timedelta(days=random.randint(1, 365))
        d.exif_modification_date = earlier.strftime("%Y:%m:%d %H:%M:%S")
        d.label = 1
        d.doc_id = d.doc_id + "_FORGED"
        return d

    def inject_font_anomaly(self, doc: DocumentMetadata) -> DocumentMetadata:
        """Font metrics deviate significantly from the expected spec."""
        d = deepcopy(doc)
        # Inject large deviations from expected font metrics
        d.font_size += random.choice([-4.0, 4.0, 6.0])
        d.font_family_hash += random.choice([-0.3, 0.3, 0.5])
        d.font_family_hash = float(np.clip(d.font_family_hash, 0.0, 1.0))
        d.line_spacing += random.choice([-0.5, 0.5, 0.8])
        d.label = 1
        d.doc_id = d.doc_id + "_FORGED"
        return d

    def inject_biometric_divergence(self, doc: DocumentMetadata) -> DocumentMetadata:
        """
        Biometric hash is inconsistent with the document's identity.
        Simulates face-swap or photo substitution.
        """
        d = deepcopy(doc)
        # Drive the biometric hash far from the legitimate distribution
        d.biometric_hash = float(np.clip(
            np.random.choice([
                np.random.uniform(0.0, 0.1),   # Too low
                np.random.uniform(0.9, 1.0),   # Too high
            ]),
            0.0, 1.0
        ))
        d.label = 1
        d.doc_id = d.doc_id + "_FORGED"
        return d

    def inject_ai_generated(self, doc: DocumentMetadata) -> DocumentMetadata:
        """
        AI-generated document: suspiciously uniform pixels, near-zero noise.
        Real scanned documents always have some noise and imperfection.
        """
        d = deepcopy(doc)
        d.pixel_uniformity = float(np.clip(np.random.normal(0.90, 0.03), 0.80, 1.0))
        d.noise_level = float(np.clip(np.random.normal(0.01, 0.005), 0.001, 0.05))
        # AI generators also often produce perfect DPI values
        d.dpi = float(DOCUMENT_SPECS[doc.doc_type]["dpi_mean"])
        d.compression_ratio = float(np.clip(np.random.normal(0.99, 0.005), 0.97, 1.0))
        d.label = 1
        d.doc_id = d.doc_id + "_FORGED"
        return d

    def inject(
        self,
        doc: DocumentMetadata,
        forgery_type: Optional[str] = None,
    ) -> DocumentMetadata:
        """Inject a random (or specified) forgery pattern."""
        if forgery_type is None:
            dist = CONFIG.kyc.forgery_type_distribution
            forgery_type = random.choices(
                list(dist.keys()), weights=list(dist.values()), k=1
            )[0]

        handler = {
            "dpi_inconsistency":    self.inject_dpi_inconsistency,
            "exif_mismatch":        self.inject_exif_mismatch,
            "font_anomaly":         self.inject_font_anomaly,
            "biometric_divergence": self.inject_biometric_divergence,
            "ai_generated":         self.inject_ai_generated,
        }
        return handler[forgery_type](doc)

    def generate_forged_batch(
        self,
        legit_docs: list[DocumentMetadata],
        n: int = None,
    ) -> list[DocumentMetadata]:
        """Generate N forged documents from legitimate base documents."""
        n = n or CONFIG.kyc.n_forged_docs
        result = []
        for _ in range(n):
            base = deepcopy(random.choice(legit_docs))
            result.append(self.inject(base))
        return result