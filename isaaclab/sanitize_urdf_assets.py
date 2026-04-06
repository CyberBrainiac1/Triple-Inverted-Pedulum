from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "_extracted_urdf" / "finaltripleinvertedpendulum"
DST_ROOT = REPO_ROOT / "_sanitized_urdf" / "finaltripleinvertedpendulum"
SRC_URDF = SRC_ROOT / "urdf" / "finaltripleinvertedpendulum.urdf"
DST_URDF = DST_ROOT / "urdf" / "finaltripleinvertedpendulum_sanitized.urdf"


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        sanitized = "unnamed"
    if sanitized[0].isdigit():
        sanitized = f"a_{sanitized}"
    return sanitized


def sanitize_mesh_name(filename: str) -> str:
    path = Path(filename)
    stem = sanitize_name(path.stem)
    return f"{stem}{path.suffix.lower()}"


def main() -> None:
    if not SRC_URDF.exists():
        raise FileNotFoundError(f"Source URDF not found: {SRC_URDF}")

    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    (DST_ROOT / "meshes").mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "urdf").mkdir(parents=True, exist_ok=True)

    tree = ET.parse(SRC_URDF)
    root = tree.getroot()

    link_map: dict[str, str] = {}
    material_map: dict[str, str] = {}

    for link in root.findall("link"):
        old = link.attrib["name"]
        new = sanitize_name(old)
        link_map[old] = new
        link.attrib["name"] = new

    for material in root.findall(".//material"):
        name = material.attrib.get("name")
        if name:
            material_map[name] = sanitize_name(name)
            material.attrib["name"] = material_map[name]

    for parent in root.findall(".//parent"):
        link = parent.attrib.get("link")
        if link in link_map:
            parent.attrib["link"] = link_map[link]

    for child in root.findall(".//child"):
        link = child.attrib.get("link")
        if link in link_map:
            child.attrib["link"] = link_map[link]

    for visual in root.findall(".//visual"):
        material = visual.find("material")
        if material is not None and material.attrib.get("name") in material_map:
            material.attrib["name"] = material_map[material.attrib["name"]]

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        original_name = filename.split("/")[-1]
        sanitized_name = sanitize_mesh_name(original_name)
        src_mesh = SRC_ROOT / "meshes" / original_name
        dst_mesh = DST_ROOT / "meshes" / sanitized_name
        shutil.copy2(src_mesh, dst_mesh)
        mesh.attrib["filename"] = f"package://finaltripleinvertedpendulum/meshes/{sanitized_name}"

    tree.write(DST_URDF, encoding="utf-8", xml_declaration=False)
    print(DST_URDF)


if __name__ == "__main__":
    main()
