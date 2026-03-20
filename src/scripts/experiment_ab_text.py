"""Experimento A/B para comparar descripcion zero-shot vs asistida con Ground Truth AD."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.inference.vlm_runner import VLMLoader


PROMPT_A = (
    "Eres un medico. Describe detalladamente la textura, el color y la morfologia "
    "de la lesion en esta imagen."
)

PROMPT_B = (
    "Eres un medico. Sabemos que la lesion en esta imagen es un Adenoma (AD). "
    "Describe detalladamente la textura, color y morfologia que justifican este diagnostico."
)


class PolypDescription(BaseModel):
    """Esquema estructurado para descripcion cualitativa de polipo."""

    texture: str = Field(..., description="Descripcion de la textura observada")
    color: str = Field(..., description="Descripcion de la coloracion observada")
    morphology: str = Field(..., description="Descripcion de la morfologia observada")
    conclusion: str = Field(..., description="Conclusion clinica breve")


@dataclass(frozen=True)
class ABInputSample:
    """Entrada seleccionada para el experimento A/B."""

    image_path: Path
    image_name: str
    ground_truth_cls: str
    predicted_cls: str
    previous_status: str


@dataclass(frozen=True)
class ModelVariant:
    """Describe una variante de inferencia detectada en JSONL previo."""

    model_id: str
    include_reasoning: bool | None
    schema_name: str | None


@dataclass(frozen=True)
class ABVariantPlan:
    """Plan de ejecucion para una variante concreta."""

    variant: ModelVariant
    samples: list[ABInputSample]


@dataclass(frozen=True)
class ABResult:
    """Resultado final por imagen con comparativa A/B."""

    sample: ABInputSample
    description_a: dict[str, str]
    description_b: dict[str, str]


@dataclass(frozen=True)
class ABVariantRunResult:
    """Resultados A/B para una variante concreta."""

    variant: ModelVariant
    items: list[ABResult]


@dataclass(frozen=True)
class LegacyRunContext:
    """Compatibilidad con wrapper previo."""

    model_id: str
    schema_name: str | None
    include_reasoning: bool | None


def _normalize_class_label(value: Any) -> str:
    """Normaliza etiquetas de clase para comparaciones robustas."""
    text = str(value or "").strip().upper()
    if text.startswith("AD") or "ADENOMA" in text:
        return "AD"
    if text.startswith("HP") or "HIPER" in text:
        return "HP"
    if text.startswith("ASS") or "SERR" in text:
        return "ASS"
    if text.startswith("UNKNOWN"):
        return "UNKNOWN"
    return text


def _extract_predicted_class(record: dict[str, Any]) -> str:
    """Extrae clase predicha desde payload o campos planos de respaldo."""
    payload = record.get("payload")
    if isinstance(payload, dict):
        candidate = (
            payload.get("predicted_class")
            or payload.get("final_diagnosis")
            or payload.get("diagnosis")
            or payload.get("class")
            or payload.get("label")
        )
        normalized = _normalize_class_label(candidate)
        if normalized:
            return normalized

    for key in ("predicted_class", "final_diagnosis", "diagnosis", "class", "label"):
        normalized = _normalize_class_label(record.get(key))
        if normalized:
            return normalized

    return ""


def _extract_ground_truth_class(record: dict[str, Any]) -> str:
    """Extrae clase real desde campos conocidos o metadata anidada."""
    for key in ("ground_truth_cls", "ground_truth", "gt_cls", "cls", "label"):
        normalized = _normalize_class_label(record.get(key))
        if normalized:
            return normalized

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("ground_truth_cls", "ground_truth", "gt_cls", "cls"):
            normalized = _normalize_class_label(metadata.get(key))
            if normalized:
                return normalized

    return ""


def _variant_sort_key(variant: ModelVariant) -> tuple[str, int, str]:
    """Clave estable para ordenar variantes detectadas."""
    reasoning_rank = 2
    if variant.include_reasoning is True:
        reasoning_rank = 1
    elif variant.include_reasoning is False:
        reasoning_rank = 0
    return (
        variant.model_id.lower(),
        reasoning_rank,
        str(variant.schema_name or "").lower(),
    )


def _reasoning_label(value: bool | None) -> str:
    """Etiqueta legible del modo de razonamiento."""
    if value is True:
        return "con razonamiento"
    if value is False:
        return "sin razonamiento"
    return "desconocido"


def _autodetect_results_file(results_dir: Path) -> Path:
    """Detecta el JSONL de batch mas reciente para PolypClassification."""
    preferred = sorted(
        results_dir.glob("batch_*PolypClassification*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if preferred:
        return preferred[0]

    fallback = sorted(
        results_dir.glob("batch_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not fallback:
        raise FileNotFoundError(
            f"No se encontraron archivos batch JSONL en: {results_dir}"
        )
    return fallback[0]


def detect_results_file(results_file: Path | None, results_dir: Path) -> Path:
    """Resuelve ruta de resultados previos con autodeteccion opcional."""
    return results_file if results_file is not None else _autodetect_results_file(results_dir)


def _iter_data_records(results_file: Path) -> list[dict[str, Any]]:
    """Carga un JSONL y devuelve solo registros de inferencia (sin metadatos)."""
    records: list[dict[str, Any]] = []
    with results_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if any(key.startswith("__") and key.endswith("__") for key in record):
                continue
            records.append(record)
    return records


def _resolve_variant_from_record(record: dict[str, Any]) -> ModelVariant | None:
    """Resuelve datos de variante (modelo + reasoning + schema) desde un registro."""
    model_id = str(record.get("model_id") or "").strip()
    if not model_id:
        return None
    schema_name_raw = str(record.get("schema_name") or "").strip()
    schema_name = schema_name_raw if schema_name_raw else None
    include_reasoning: bool | None = None
    if "include_reasoning" in record:
        include_reasoning = bool(record.get("include_reasoning"))
    return ModelVariant(
        model_id=model_id,
        include_reasoning=include_reasoning,
        schema_name=schema_name,
    )


def _build_variant_candidates(
    records: list[dict[str, Any]],
    *,
    override_model: str | None,
) -> dict[ModelVariant, list[ABInputSample]]:
    """Agrupa candidatos AD por variante detectada."""
    grouped: dict[ModelVariant, list[ABInputSample]] = {}

    for record in records:
        status = str(record.get("status") or "").strip().lower()
        if status != "ok":
            continue

        variant = _resolve_variant_from_record(record)
        if variant is None:
            continue

        if override_model and variant.model_id != override_model.strip():
            continue

        ground_truth = _extract_ground_truth_class(record)
        if ground_truth != "AD":
            continue

        predicted = _extract_predicted_class(record)
        if not predicted:
            continue

        image_path_value = str(record.get("image_path") or "").strip()
        if not image_path_value:
            continue

        image_path = Path(image_path_value)
        if not image_path.exists():
            continue

        previous_status = "Acierto" if predicted == "AD" else "Fallo"
        sample = ABInputSample(
            image_path=image_path,
            image_name=image_path.name,
            ground_truth_cls=ground_truth,
            predicted_cls=predicted,
            previous_status=previous_status,
        )
        grouped.setdefault(variant, []).append(sample)

    return grouped


def _select_samples(
    candidates: list[ABInputSample],
    *,
    n_correct: int,
    n_incorrect: int,
    seed: int,
) -> list[ABInputSample]:
    """Selecciona 5 AD aciertos y 5 AD fallos de forma reproducible."""
    unique_by_path: dict[str, ABInputSample] = {}
    for item in candidates:
        unique_by_path[str(item.image_path)] = item
    deduped = list(unique_by_path.values())

    correct = sorted(
        [item for item in deduped if item.previous_status == "Acierto"],
        key=lambda item: str(item.image_path).lower(),
    )
    incorrect = sorted(
        [item for item in deduped if item.previous_status == "Fallo"],
        key=lambda item: str(item.image_path).lower(),
    )

    if len(correct) < n_correct or len(incorrect) < n_incorrect:
        raise RuntimeError(
            "No hay suficientes casos AD para el experimento. "
            f"Disponibles -> Aciertos: {len(correct)}, Fallos: {len(incorrect)}. "
            f"Requeridos -> Aciertos: {n_correct}, Fallos: {n_incorrect}."
        )

    rng = random.Random(seed)
    selected_correct = rng.sample(correct, n_correct)
    selected_incorrect = rng.sample(incorrect, n_incorrect)

    selected = selected_correct + selected_incorrect
    rng.shuffle(selected)
    return selected


def build_execution_plans(
    *,
    results_file: Path,
    n_correct: int,
    n_incorrect: int,
    seed: int,
    override_model: str | None = None,
) -> list[ABVariantPlan]:
    """Construye planes por variante detectada en JSONL."""
    records = _iter_data_records(results_file)
    grouped = _build_variant_candidates(
        records,
        override_model=override_model,
    )

    if not grouped:
        model_hint = f" para el modelo '{override_model}'" if override_model else ""
        raise RuntimeError(
            "No se detectaron candidatos AD validos"
            f"{model_hint} en {results_file}."
        )

    plans: list[ABVariantPlan] = []
    for variant in sorted(grouped.keys(), key=_variant_sort_key):
        selected = _select_samples(
            grouped[variant],
            n_correct=n_correct,
            n_incorrect=n_incorrect,
            seed=seed,
        )
        plans.append(ABVariantPlan(variant=variant, samples=selected))

    return plans


def _extract_previous_run_context(
    results_file: Path,
    *,
    override_model: str | None = None,
) -> LegacyRunContext:
    """Compatibilidad para wrappers existentes: usa la primera variante del plan."""
    plans = build_execution_plans(
        results_file=results_file,
        n_correct=1,
        n_incorrect=1,
        seed=0,
        override_model=override_model,
    )
    first = plans[0].variant
    return LegacyRunContext(
        model_id=first.model_id,
        schema_name=first.schema_name,
        include_reasoning=first.include_reasoning,
    )


def _safe_text(value: Any) -> str:
    """Sanitiza texto para celdas Markdown."""
    text = str(value or "").strip()
    if not text:
        return "-"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return text


def _run_single_inference(
    *,
    loader: VLMLoader,
    image_path: Path,
    prompt: str,
    temperature: float,
    seed: int | None,
) -> dict[str, str]:
    """Ejecuta una inferencia y devuelve dict serializable para markdown."""
    result = loader.inference(
        image_path=image_path,
        prompt=prompt,
        schema=PolypDescription,
        temperature=temperature,
        seed=seed,
    )
    payload = result.model_dump()
    return {
        "texture": str(payload.get("texture") or ""),
        "color": str(payload.get("color") or ""),
        "morphology": str(payload.get("morphology") or ""),
        "conclusion": str(payload.get("conclusion") or ""),
    }


def _run_ab_inference(
    *,
    model_id: str,
    samples: list[ABInputSample],
    temperature: float,
    seed: int | None,
    verbose: bool,
    server_api_host: str | None,
    api_token: str | None,
) -> list[ABResult]:
    """Ejecuta ambos prompts por muestra y asegura descarga de VRAM al finalizar."""
    results: list[ABResult] = []
    loader = VLMLoader(
        model_path=model_id,
        verbose=verbose,
        server_api_host=server_api_host,
        api_token=api_token,
    )

    try:
        loader.load_model()
        for index, sample in enumerate(samples, start=1):
            print(
                f"[{index}/{len(samples)}] Procesando {sample.image_name} "
                f"({sample.previous_status} previo)..."
            )

            description_a: dict[str, str]
            description_b: dict[str, str]

            try:
                description_a = _run_single_inference(
                    loader=loader,
                    image_path=sample.image_path,
                    prompt=PROMPT_A,
                    temperature=temperature,
                    seed=seed,
                )
            except Exception as error:
                description_a = {
                    "texture": "",
                    "color": "",
                    "morphology": "",
                    "conclusion": f"ERROR: {error}",
                }

            try:
                description_b = _run_single_inference(
                    loader=loader,
                    image_path=sample.image_path,
                    prompt=PROMPT_B,
                    temperature=temperature,
                    seed=seed,
                )
            except Exception as error:
                description_b = {
                    "texture": "",
                    "color": "",
                    "morphology": "",
                    "conclusion": f"ERROR: {error}",
                }

            results.append(
                ABResult(
                    sample=sample,
                    description_a=description_a,
                    description_b=description_b,
                )
            )
    finally:
        loader.unload_model()

    return results


def _render_markdown(
    *,
    runs: list[ABVariantRunResult],
    seed: int,
    temperature: float,
    results_file: Path,
) -> str:
    """Construye reporte markdown comparativo del experimento A/B."""
    lines: list[str] = []
    lines.append("# Results A/B Experiment")
    lines.append("")
    lines.append("## Configuracion")
    lines.append("")
    lines.append("| Campo | Valor |")
    lines.append("|---|---|")
    lines.append(f"| Registros de origen | `{results_file}` |")
    lines.append(f"| Semilla de seleccion | `{seed}` |")
    lines.append(f"| Temperatura | `{temperature}` |")
    lines.append(f"| Variantes evaluadas | `{len(runs)}` |")
    lines.append(f"| Total de imagenes AD evaluadas | `{sum(len(run.items) for run in runs)}` |")
    lines.append("")
    lines.append("## Resumen de Variantes")
    lines.append("")
    lines.append("| Modelo | Modo origen | Esquema origen | Casos evaluados |")
    lines.append("|---|---|---|---|")
    for run in runs:
        variant = run.variant
        lines.append(
            "| "
            f"{_safe_text(variant.model_id)} | "
            f"{_safe_text(_reasoning_label(variant.include_reasoning))} | "
            f"{_safe_text(variant.schema_name or 'N/A')} | "
            f"{len(run.items)} |"
        )
    lines.append("")
    lines.append("## Hipotesis")
    lines.append("")
    lines.append(
        "Comparar si la inyeccion del Ground Truth (Prompt B) cambia cualitativamente la "
        "descripcion visual frente al enfoque zero-shot (Prompt A)."
    )
    lines.append("")

    for run in runs:
        variant = run.variant
        lines.append(f"## Variante: {variant.model_id}")
        lines.append("")
        lines.append(f"- Modo origen: **{_reasoning_label(variant.include_reasoning)}**")
        if variant.schema_name:
            lines.append(f"- Esquema origen: **{variant.schema_name}**")
        lines.append(f"- Casos evaluados: **{len(run.items)}**")
        lines.append("")

        for index, item in enumerate(run.items, start=1):
            sample = item.sample
            lines.append(f"### Caso {index}: {sample.image_name}")
            lines.append("")
            lines.append(f"- Estado previo: **{sample.previous_status}**")
            lines.append(f"- Ground Truth: **{sample.ground_truth_cls}**")
            lines.append(f"- Prediccion previa: **{sample.predicted_cls}**")
            lines.append(f"- Imagen: `{sample.image_path}`")
            lines.append("")
            lines.append("| Campo | Descripcion A (Zero-Shot) | Descripcion B (Asistido AD) |")
            lines.append("|---|---|---|")
            lines.append(
                f"| Textura | {_safe_text(item.description_a.get('texture'))} | "
                f"{_safe_text(item.description_b.get('texture'))} |"
            )
            lines.append(
                f"| Color | {_safe_text(item.description_a.get('color'))} | "
                f"{_safe_text(item.description_b.get('color'))} |"
            )
            lines.append(
                f"| Morfologia | {_safe_text(item.description_a.get('morphology'))} | "
                f"{_safe_text(item.description_b.get('morphology'))} |"
            )
            lines.append(
                f"| Conclusion | {_safe_text(item.description_a.get('conclusion'))} | "
                f"{_safe_text(item.description_b.get('conclusion'))} |"
            )
            lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Crea parser de argumentos CLI para experimento A/B."""
    parser = argparse.ArgumentParser(
        description="Ejecuta experimento A/B: zero-shot vs prompt asistido con AD."
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help="ID/tag del modelo en LM Studio (opcional, filtra variantes detectadas)",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Ruta JSONL de resultados previos (si no se indica, se autodetecta)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/processed/batch_results"),
        help="Directorio para autodeteccion de JSONL previos",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/processed/ab_results/results_ab_experiment.md"),
        help="Ruta del Markdown de salida",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatura de inferencia para ambos prompts",
    )
    parser.add_argument(
        "--n-correct",
        type=int,
        default=5,
        help="Numero de AD acertados a seleccionar",
    )
    parser.add_argument(
        "--n-incorrect",
        type=int,
        default=5,
        help="Numero de AD fallados a seleccionar",
    )
    parser.add_argument(
        "--server-api-host",
        default=None,
        help="Host opcional de API de LM Studio",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="Token opcional de API de LM Studio",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activa logs verbosos de VLMLoader",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI."""
    args = build_parser().parse_args(argv)

    results_file = detect_results_file(args.results_file, args.results_dir)
    print(f"Usando resultados previos: {results_file}")

    plans = build_execution_plans(
        results_file=results_file,
        n_correct=args.n_correct,
        n_incorrect=args.n_incorrect,
        seed=args.seed,
        override_model=args.model,
    )

    print(f"Variantes detectadas para ejecutar: {len(plans)}")
    for plan in plans:
        print(
            f"- {plan.variant.model_id} | {_reasoning_label(plan.variant.include_reasoning)} "
            f"| schema={plan.variant.schema_name or 'N/A'} | muestras={len(plan.samples)}"
        )

    runs: list[ABVariantRunResult] = []
    for plan in plans:
        variant = plan.variant
        print(
            "Ejecutando variante -> "
            f"model={variant.model_id}, reasoning={_reasoning_label(variant.include_reasoning)}, "
            f"schema={variant.schema_name or 'N/A'}"
        )
        items = _run_ab_inference(
            model_id=variant.model_id,
            samples=plan.samples,
            temperature=args.temperature,
            seed=args.seed,
            verbose=args.verbose,
            server_api_host=args.server_api_host,
            api_token=args.api_token,
        )
        runs.append(ABVariantRunResult(variant=variant, items=items))

    markdown = _render_markdown(
        runs=runs,
        seed=args.seed,
        temperature=args.temperature,
        results_file=results_file,
    )

    output_path: Path = args.output_md
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Reporte generado: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
