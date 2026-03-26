"""Utilidades para generar reportes Markdown dinamicos con secciones combinables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence


class MarkdownSection(Protocol):
    """Contrato de una seccion renderizable en Markdown."""

    def render(self, report_path: Path) -> str:
        """Renderiza el bloque Markdown de la seccion."""
        ...


def _to_heading(level: int, text: str) -> str:
    """Construye una cabecera Markdown de nivel configurable."""
    safe_level = min(max(level, 1), 6)
    return f"{'#' * safe_level} {text}"


def _as_report_relative(path: str | Path, report_path: Path) -> str:
    """Convierte una ruta absoluta o relativa a una ruta relativa al .md de salida."""
    candidate = Path(path)
    if not candidate.is_absolute():
        # Si llega relativa (p. ej. desde JSONL), se interpreta respecto al cwd
        # del proceso y luego se convierte a relativa al reporte.
        candidate = (Path.cwd() / candidate).resolve()

    try:
        return candidate.relative_to(report_path.parent).as_posix()
    except ValueError:
        # Fallback para unidades distintas en Windows o rutas fuera del arbol del reporte.
        from os.path import relpath

        return relpath(str(candidate), str(report_path.parent)).replace("\\", "/")


@dataclass(slots=True)
class TextSection:
    """Seccion de texto libre con cabecera opcional."""

    text: str
    heading: str | None = None
    heading_level: int = 2

    def render(self, report_path: Path) -> str:
        _ = report_path
        lines: list[str] = []
        if self.heading:
            lines.append(_to_heading(self.heading_level, self.heading))
            lines.append("")
        lines.append(self.text.rstrip())
        return "\n".join(lines)


@dataclass(slots=True)
class RawMarkdownSection:
    """Seccion para inyectar Markdown crudo ya preparado."""

    markdown: str

    def render(self, report_path: Path) -> str:
        _ = report_path
        return self.markdown.rstrip()


@dataclass(slots=True)
class ListSection:
    """Seccion de lista (ordenada o no ordenada) con cabecera opcional."""

    items: Sequence[str]
    heading: str | None = None
    ordered: bool = False
    heading_level: int = 2

    def render(self, report_path: Path) -> str:
        _ = report_path
        lines: list[str] = []
        if self.heading:
            lines.append(_to_heading(self.heading_level, self.heading))
            lines.append("")

        for idx, item in enumerate(self.items, start=1):
            marker = f"{idx}." if self.ordered else "-"
            lines.append(f"{marker} {item}")

        return "\n".join(lines)


@dataclass(slots=True)
class CodeBlockSection:
    """Seccion de bloque de codigo con lenguaje opcional."""

    code: str
    language: str = ""
    heading: str | None = None
    heading_level: int = 2

    def render(self, report_path: Path) -> str:
        _ = report_path
        lines: list[str] = []
        if self.heading:
            lines.append(_to_heading(self.heading_level, self.heading))
            lines.append("")

        lines.append(f"```{self.language}".rstrip())
        lines.append(self.code.rstrip())
        lines.append("```")
        return "\n".join(lines)


@dataclass(slots=True)
class TableSection:
    """Seccion de tabla Markdown."""

    headers: Sequence[str]
    rows: Sequence[Sequence[str]]
    heading: str | None = None
    heading_level: int = 2

    def render(self, report_path: Path) -> str:
        _ = report_path
        if not self.headers:
            raise ValueError("TableSection requiere al menos una cabecera.")

        lines: list[str] = []
        if self.heading:
            lines.append(_to_heading(self.heading_level, self.heading))
            lines.append("")

        header_line = "| " + " | ".join(self.headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(self.headers)) + " |"
        lines.append(header_line)
        lines.append(sep_line)

        for row in self.rows:
            padded = list(row)[: len(self.headers)]
            if len(padded) < len(self.headers):
                padded.extend([""] * (len(self.headers) - len(padded)))
            lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(lines)


@dataclass(slots=True)
class ImageItem:
    """Elemento de imagen para grupos visuales dentro del reporte."""

    path: str | Path
    alt_text: str
    caption: str | None = None


@dataclass(slots=True)
class ImageGroupSection:
    """Seccion de grupo de imagenes con cabecera opcional."""

    images: Sequence[ImageItem]
    heading: str | None = None
    heading_level: int = 3
    layout: str = "vertical"
    columns: int = 3
    image_width_percent: int = 32

    def render(self, report_path: Path) -> str:
        lines: list[str] = []
        if self.heading:
            lines.append(_to_heading(self.heading_level, self.heading))
            lines.append("")

        if self.layout not in {"vertical", "grid"}:
            raise ValueError("ImageGroupSection.layout debe ser 'vertical' o 'grid'.")

        if self.layout == "grid":
            lines.append("<p>")
            for image in self.images:
                rel_path = _as_report_relative(image.path, report_path)
                title_attr = f' title="{image.caption}"' if image.caption else ""
                lines.append(
                    f'<img src="{rel_path}" alt="{image.alt_text}" width="{self.image_width_percent}%"{title_attr} />'
                )
            lines.append("</p>")
            return "\n".join(lines).rstrip()

        for image in self.images:
            rel_path = _as_report_relative(image.path, report_path)
            lines.append(f"![{image.alt_text}]({rel_path})")
            if image.caption:
                lines.append("")
                lines.append(f"_{image.caption}_")
            lines.append("")

        return "\n".join(lines).rstrip()


@dataclass(slots=True)
class SectionGroup:
    """Seccion compuesta para combinar bloques heterogeneos en cascada."""

    sections: Sequence[MarkdownSection] = field(default_factory=list)
    heading: str | None = None
    heading_level: int = 2

    def render(self, report_path: Path) -> str:
        blocks: list[str] = []
        if self.heading:
            blocks.append(_to_heading(self.heading_level, self.heading))

        for section in self.sections:
            if blocks:
                blocks.append("")
            blocks.append(section.render(report_path).rstrip())

        return "\n".join(blocks).rstrip()


@dataclass(slots=True)
class SectionListItem:
    """Elemento compuesto para listas de secciones."""

    title: str
    section: MarkdownSection


@dataclass(slots=True)
class SectionListSection:
    """Lista de secciones combinables (texto, imagenes, tablas, etc.)."""

    items: Sequence[SectionListItem]
    heading: str | None = None
    heading_level: int = 2
    ordered: bool = True

    def render(self, report_path: Path) -> str:
        lines: list[str] = []
        if self.heading:
            lines.append(_to_heading(self.heading_level, self.heading))
            lines.append("")

        for idx, item in enumerate(self.items, start=1):
            marker = f"{idx}." if self.ordered else "-"
            lines.append(f"{marker} **{item.title}**")

            rendered = item.section.render(report_path).rstrip()
            if rendered:
                for raw_line in rendered.splitlines():
                    if raw_line.strip():
                        lines.append(f"   {raw_line}")
                    else:
                        lines.append("")

        return "\n".join(lines).rstrip()


def build_markdown_document(
    *,
    title: str,
    sections: Sequence[MarkdownSection],
    metadata: dict[str, str] | None = None,
    report_path: Path,
) -> str:
    """Construye el contenido completo del reporte Markdown."""
    blocks: list[str] = [_to_heading(1, title)]

    if metadata:
        blocks.append("")
        for key, value in metadata.items():
            blocks.append(f"- **{key}:** {value}")

    for section in sections:
        blocks.append("")
        blocks.append(section.render(report_path).rstrip())

    blocks.append("")
    return "\n".join(blocks)


def write_markdown_report(
    *,
    report_path: str | Path,
    title: str,
    sections: Sequence[MarkdownSection],
    metadata: dict[str, str] | None = None,
) -> Path:
    """Escribe en disco un reporte Markdown generado dinamicamente."""
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown_document(
        title=title,
        sections=sections,
        metadata=metadata,
        report_path=destination,
    )
    destination.write_text(markdown, encoding="utf-8")
    return destination
