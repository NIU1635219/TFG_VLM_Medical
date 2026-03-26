from __future__ import annotations

from pathlib import Path
import os

from src.utils.tests_ui.markdown_report import (
    CodeBlockSection,
    ImageGroupSection,
    ImageItem,
    ListSection,
    SectionGroup,
    SectionListItem,
    SectionListSection,
    TableSection,
    TextSection,
    write_markdown_report,
)


def test_write_markdown_report_with_composed_sections(tmp_path: Path) -> None:
    report_path = tmp_path / "report" / "result.md"
    image_path = tmp_path / "assets" / "img.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake")

    sections = [
        TextSection(text="Intro", heading="Resumen"),
        ListSection(items=["uno", "dos"], heading="Checklist", ordered=True),
        TableSection(
            headers=["name", "value"],
            rows=[["iou", "0.75"], ["count", "3"]],
            heading="Tabla",
        ),
        CodeBlockSection(code="print('ok')", language="python", heading="Snippet"),
        ImageGroupSection(
            images=[ImageItem(path=image_path, alt_text="sample", caption="captura")],
            heading="Imagenes",
            layout="grid",
        ),
    ]

    out_path = write_markdown_report(
        report_path=report_path,
        title="Demo Report",
        metadata={"Model": "demo-model"},
        sections=sections,
    )

    content = out_path.read_text(encoding="utf-8")
    assert "# Demo Report" in content
    assert "## Resumen" in content
    assert "1. uno" in content
    assert "| name | value |" in content
    assert "```python" in content
    assert "<img src=" in content


def test_section_list_section_renders_nested_sections(tmp_path: Path) -> None:
    report_path = tmp_path / "nested.md"

    nested = SectionListSection(
        heading="Bloques",
        items=[
            SectionListItem(title="Texto", section=TextSection(text="Linea A")),
            SectionListItem(
                title="Grupo",
                section=SectionGroup(
                    sections=[
                        ListSection(items=["a", "b"], ordered=False),
                        TextSection(text="Linea B"),
                    ]
                ),
            ),
        ],
        ordered=True,
    )

    out_path = write_markdown_report(
        report_path=report_path,
        title="Nested",
        sections=[nested],
    )

    content = out_path.read_text(encoding="utf-8")
    assert "## Bloques" in content
    assert "1. **Texto**" in content
    assert "2. **Grupo**" in content
    assert "Linea A" in content
    assert "- a" in content


def test_image_paths_are_rebased_to_report_directory_for_relative_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    image_path = Path("data/processed/bbox_results/run_x/annotated/sample.jpg")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake")

    report_path = Path("data/processed/bbox_results/run_x/results.md")

    out_path = write_markdown_report(
        report_path=report_path,
        title="Path Rebase",
        sections=[
            ImageGroupSection(
                images=[ImageItem(path=image_path, alt_text="img")],
                heading="Visual",
            )
        ],
    )

    content = out_path.read_text(encoding="utf-8")
    expected_rel = os.path.relpath(image_path.resolve(), report_path.parent.resolve()).replace("\\", "/")
    assert f"![img]({expected_rel})" in content
