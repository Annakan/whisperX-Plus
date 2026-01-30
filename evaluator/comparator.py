from __future__ import annotations

from dataclasses import dataclass

from jiwer import cer, wer
from rapidfuzz import fuzz


@dataclass
class MetricsResult:
    """Metrics comparing generated text against reference."""

    levenshtein_ratio: float
    wer: float | None = None
    cer: float | None = None


def compare(generated: str, reference: str) -> MetricsResult:
    """
    Compare generated text against reference using multiple metrics.

    Args:
        generated: Generated transcript text
        reference: Reference transcript text

    Returns:
        MetricsResult with Levenshtein ratio, WER, and CER
    """
    if not reference or not generated:
        return MetricsResult(
            levenshtein_ratio=0.0 if reference != generated else 1.0,
            wer=1.0 if reference or generated else 0.0,
            cer=1.0 if reference or generated else 0.0,
        )

    lev_ratio = fuzz.ratio(generated, reference) / 100.0

    try:
        wer_score = wer(reference, generated)
    except Exception:
        wer_score = None

    try:
        cer_score = cer(reference, generated)
    except Exception:
        cer_score = None

    return MetricsResult(
        levenshtein_ratio=lev_ratio,
        wer=wer_score,
        cer=cer_score,
    )


def compare_all_variants(
    generated_variants: dict[str, str],
    reference_variants: dict[str, str],
) -> dict[str, MetricsResult]:
    """
    Compare all variants between generated and reference.

    Args:
        generated_variants: Dict of variant name to generated text (VA, VB, VC, VD)
        reference_variants: Dict of variant name to reference text (VA, VB, VC, VD)

    Returns:
        Dict mapping variant name to MetricsResult
    """
    results: dict[str, MetricsResult] = {}

    for variant_name in generated_variants:
        if variant_name in reference_variants:
            results[variant_name] = compare(
                generated_variants[variant_name],
                reference_variants[variant_name],
            )

    return results
