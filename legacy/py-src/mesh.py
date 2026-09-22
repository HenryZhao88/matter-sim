"""Mesh data structures and importers (OBJ / STL).

Provides a lightweight triangle-mesh representation that can be rendered by
the main simulator's PIL pipeline or the raytracer module.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class Vertex:
    """A single 3D position."""
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Face:
    """Triangle (or n-gon) face — list of vertex indices + optional per-vertex
    texture-coordinate indices and normal indices."""
    vertex_indices: list[int]
    normal_indices: list[int] = field(default_factory=list)
    uv_indices: list[int] = field(default_factory=list)
    material_name: str = ""


@dataclass
class MeshMaterial:
    """Basic PBR material definition (matches Blender Principled BSDF subset)."""
    name: str = "default"
    albedo: tuple[float, float, float] = (0.8, 0.8, 0.8)  # linear RGB 0-1
    roughness: float = 0.5
    metallic: float = 0.0
    emission: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emission_strength: float = 0.0
    opacity: float = 1.0
    ior: float = 1.45

    def albedo_rgb(self) -> tuple[int, int, int]:
        """Return albedo as 0-255 sRGB tuple."""
        return (
            max(0, min(255, int(self.albedo[0] * 255))),
            max(0, min(255, int(self.albedo[1] * 255))),
            max(0, min(255, int(self.albedo[2] * 255))),
        )


@dataclass
class Mesh:
    """In-memory triangle mesh with optional materials."""
    name: str = "Mesh"
    vertices: list[Vertex] = field(default_factory=list)
    normals: list[Vertex] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    materials: dict[str, MeshMaterial] = field(default_factory=dict)
    source_path: str = ""

    # Computed bounding box (call compute_bounds after loading)
    bb_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bb_max: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # -----------------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------------

    def compute_bounds(self) -> None:
        if not self.vertices:
            self.bb_min = (0.0, 0.0, 0.0)
            self.bb_max = (0.0, 0.0, 0.0)
            return
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        self.bb_min = (min(xs), min(ys), min(zs))
        self.bb_max = (max(xs), max(ys), max(zs))

    def center(self) -> tuple[float, float, float]:
        self.compute_bounds()
        return (
            (self.bb_min[0] + self.bb_max[0]) * 0.5,
            (self.bb_min[1] + self.bb_max[1]) * 0.5,
            (self.bb_min[2] + self.bb_max[2]) * 0.5,
        )

    def extent(self) -> float:
        """Return the max half-extent (for normalizing to a target size)."""
        self.compute_bounds()
        dx = self.bb_max[0] - self.bb_min[0]
        dy = self.bb_max[1] - self.bb_min[1]
        dz = self.bb_max[2] - self.bb_min[2]
        return max(dx, dy, dz) * 0.5 if max(dx, dy, dz) > 1e-12 else 1.0

    def normalize(self, target_size: float = 1.0) -> None:
        """Centre mesh at origin and scale so max extent == target_size."""
        cx, cy, cz = self.center()
        scale = target_size / (2.0 * self.extent())
        for v in self.vertices:
            v.x = (v.x - cx) * scale
            v.y = (v.y - cy) * scale
            v.z = (v.z - cz) * scale
        self.compute_bounds()

    def triangulate(self) -> None:
        """Fan-triangulate any faces with more than 3 vertices (in-place)."""
        new_faces: list[Face] = []
        for f in self.faces:
            n = len(f.vertex_indices)
            if n < 3:
                continue
            if n == 3:
                new_faces.append(f)
                continue
            # Fan from vertex 0
            for i in range(1, n - 1):
                nf = Face(
                    vertex_indices=[f.vertex_indices[0], f.vertex_indices[i], f.vertex_indices[i + 1]],
                    normal_indices=([f.normal_indices[0], f.normal_indices[i], f.normal_indices[i + 1]]
                                   if len(f.normal_indices) >= n else []),
                    uv_indices=([f.uv_indices[0], f.uv_indices[i], f.uv_indices[i + 1]]
                                if len(f.uv_indices) >= n else []),
                    material_name=f.material_name,
                )
                new_faces.append(nf)
        self.faces = new_faces

    def compute_face_normal(self, face: Face) -> tuple[float, float, float]:
        """Compute flat face normal from the first 3 vertices (not normalised)."""
        if len(face.vertex_indices) < 3:
            return (0.0, 0.0, 1.0)
        v0 = self.vertices[face.vertex_indices[0]]
        v1 = self.vertices[face.vertex_indices[1]]
        v2 = self.vertices[face.vertex_indices[2]]
        ax, ay, az = v1.x - v0.x, v1.y - v0.y, v1.z - v0.z
        bx, by, bz = v2.x - v0.x, v2.y - v0.y, v2.z - v0.z
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        if ln < 1e-12:
            return (0.0, 0.0, 1.0)
        return (nx / ln, ny / ln, nz / ln)

    def face_center(self, face: Face) -> tuple[float, float, float]:
        """Average position of the face's vertices."""
        n = len(face.vertex_indices)
        if n == 0:
            return (0.0, 0.0, 0.0)
        sx = sy = sz = 0.0
        for vi in face.vertex_indices:
            v = self.vertices[vi]
            sx += v.x
            sy += v.y
            sz += v.z
        return (sx / n, sy / n, sz / n)

    def vertex_count(self) -> int:
        return len(self.vertices)

    def face_count(self) -> int:
        return len(self.faces)

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict (for scene save/load)."""
        return {
            "name": self.name,
            "source_path": self.source_path,
            "vertices": [(v.x, v.y, v.z) for v in self.vertices],
            "normals": [(n.x, n.y, n.z) for n in self.normals],
            "uvs": list(self.uvs),
            "faces": [
                {
                    "vi": f.vertex_indices,
                    "ni": f.normal_indices,
                    "ui": f.uv_indices,
                    "mat": f.material_name,
                }
                for f in self.faces
            ],
            "materials": {
                k: {
                    "albedo": list(m.albedo),
                    "roughness": m.roughness,
                    "metallic": m.metallic,
                    "emission": list(m.emission),
                    "emission_strength": m.emission_strength,
                    "opacity": m.opacity,
                    "ior": m.ior,
                }
                for k, m in self.materials.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Mesh":
        """Deserialise from a dict (scene load)."""
        m = cls(name=d.get("name", "Mesh"), source_path=d.get("source_path", ""))
        for vx, vy, vz in d.get("vertices", []):
            m.vertices.append(Vertex(vx, vy, vz))
        for nx, ny, nz in d.get("normals", []):
            m.normals.append(Vertex(nx, ny, nz))
        m.uvs = [tuple(uv) for uv in d.get("uvs", [])]
        for fd in d.get("faces", []):
            m.faces.append(Face(
                vertex_indices=fd["vi"],
                normal_indices=fd.get("ni", []),
                uv_indices=fd.get("ui", []),
                material_name=fd.get("mat", ""),
            ))
        for mk, md in d.get("materials", {}).items():
            m.materials[mk] = MeshMaterial(
                name=mk,
                albedo=tuple(md.get("albedo", [0.8, 0.8, 0.8])),
                roughness=md.get("roughness", 0.5),
                metallic=md.get("metallic", 0.0),
                emission=tuple(md.get("emission", [0.0, 0.0, 0.0])),
                emission_strength=md.get("emission_strength", 0.0),
                opacity=md.get("opacity", 1.0),
                ior=md.get("ior", 1.45),
            )
        m.compute_bounds()
        return m


# ---------------------------------------------------------------------------
# OBJ importer
# ---------------------------------------------------------------------------

def _parse_mtl_file(path: str) -> dict[str, MeshMaterial]:
    """Parse a Wavefront .mtl material library file."""
    materials: dict[str, MeshMaterial] = {}
    current: MeshMaterial | None = None
    if not os.path.isfile(path):
        return materials
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0].lower()
            if key == "newmtl" and len(parts) >= 2:
                name = " ".join(parts[1:])
                current = MeshMaterial(name=name)
                materials[name] = current
            elif current is None:
                continue
            elif key == "kd" and len(parts) >= 4:
                current.albedo = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif key == "ke" and len(parts) >= 4:
                current.emission = (float(parts[1]), float(parts[2]), float(parts[3]))
                current.emission_strength = max(current.emission)
            elif key in ("ns",) and len(parts) >= 2:
                # Specular exponent → roughness approximation
                ns = float(parts[1])
                current.roughness = max(0.0, min(1.0, 1.0 - math.sqrt(ns / 1000.0)))
            elif key == "d" and len(parts) >= 2:
                current.opacity = float(parts[1])
            elif key == "ni" and len(parts) >= 2:
                current.ior = float(parts[1])
    return materials


def load_obj(path: str, normalize_size: float | None = None) -> Mesh:
    """Load a Wavefront OBJ file into a Mesh.

    Args:
        path: Path to .obj file.
        normalize_size: If given, normalise mesh so max extent = this value.

    Returns:
        Populated Mesh with triangulated faces.
    """
    mesh = Mesh(name=os.path.splitext(os.path.basename(path))[0], source_path=path)
    current_material = ""
    obj_dir = os.path.dirname(os.path.abspath(path))

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            key = parts[0].lower()

            if key == "v" and len(parts) >= 4:
                mesh.vertices.append(Vertex(float(parts[1]), float(parts[2]), float(parts[3])))

            elif key == "vn" and len(parts) >= 4:
                mesh.normals.append(Vertex(float(parts[1]), float(parts[2]), float(parts[3])))

            elif key == "vt" and len(parts) >= 3:
                mesh.uvs.append((float(parts[1]), float(parts[2])))

            elif key == "f":
                vis: list[int] = []
                nis: list[int] = []
                uis: list[int] = []
                for token in parts[1:]:
                    # Format: v, v/vt, v/vt/vn, v//vn
                    segs = token.split("/")
                    vi = int(segs[0])
                    vis.append(vi - 1 if vi > 0 else len(mesh.vertices) + vi)
                    if len(segs) >= 2 and segs[1]:
                        ui = int(segs[1])
                        uis.append(ui - 1 if ui > 0 else len(mesh.uvs) + ui)
                    if len(segs) >= 3 and segs[2]:
                        ni = int(segs[2])
                        nis.append(ni - 1 if ni > 0 else len(mesh.normals) + ni)
                mesh.faces.append(Face(
                    vertex_indices=vis,
                    normal_indices=nis,
                    uv_indices=uis,
                    material_name=current_material,
                ))

            elif key == "mtllib" and len(parts) >= 2:
                mtl_name = " ".join(parts[1:])
                mtl_path = os.path.join(obj_dir, mtl_name)
                mesh.materials.update(_parse_mtl_file(mtl_path))

            elif key == "usemtl" and len(parts) >= 2:
                current_material = " ".join(parts[1:])

    mesh.triangulate()
    mesh.compute_bounds()
    if normalize_size is not None:
        mesh.normalize(normalize_size)

    # Ensure default material exists
    if not mesh.materials:
        mesh.materials["default"] = MeshMaterial()

    return mesh


# ---------------------------------------------------------------------------
# STL importer (binary and ASCII)
# ---------------------------------------------------------------------------

def load_stl(path: str, normalize_size: float | None = None) -> Mesh:
    """Load a STL file (binary or ASCII) into a Mesh."""
    with open(path, "rb") as f:
        header = f.read(80)
        if header[:5] == b"solid":
            # Could be ASCII STL — check if the rest looks textual
            f.seek(0)
            try:
                text = f.read().decode("ascii")
                if "facet normal" in text.lower():
                    return _load_stl_ascii(path, text, normalize_size)
            except UnicodeDecodeError:
                pass
            f.seek(80)
        return _load_stl_binary(path, f, normalize_size)


def _load_stl_binary(path: str, f, normalize_size: float | None) -> Mesh:
    mesh = Mesh(name=os.path.splitext(os.path.basename(path))[0], source_path=path)
    num_tris = struct.unpack("<I", f.read(4))[0]
    for _ in range(num_tris):
        data = f.read(50)  # 12 floats + 2 byte attribute
        if len(data) < 50:
            break
        vals = struct.unpack("<12fH", data)
        nx, ny, nz = vals[0], vals[1], vals[2]
        ni = len(mesh.normals)
        mesh.normals.append(Vertex(nx, ny, nz))
        vi_start = len(mesh.vertices)
        for j in range(3):
            off = 3 + j * 3
            mesh.vertices.append(Vertex(vals[off], vals[off + 1], vals[off + 2]))
        mesh.faces.append(Face(
            vertex_indices=[vi_start, vi_start + 1, vi_start + 2],
            normal_indices=[ni, ni, ni],
        ))
    mesh.compute_bounds()
    if normalize_size is not None:
        mesh.normalize(normalize_size)
    mesh.materials["default"] = MeshMaterial()
    return mesh


def _load_stl_ascii(path: str, text: str, normalize_size: float | None) -> Mesh:
    mesh = Mesh(name=os.path.splitext(os.path.basename(path))[0], source_path=path)
    lines = text.strip().splitlines()
    current_normal: Vertex | None = None
    tri_verts: list[int] = []
    for raw in lines:
        line = raw.strip().lower()
        if line.startswith("facet normal"):
            parts = line.split()
            if len(parts) >= 5:
                current_normal = Vertex(float(parts[2]), float(parts[3]), float(parts[4]))
        elif line.startswith("vertex"):
            parts = line.split()
            if len(parts) >= 4:
                vi = len(mesh.vertices)
                mesh.vertices.append(Vertex(float(parts[1]), float(parts[2]), float(parts[3])))
                tri_verts.append(vi)
        elif line.startswith("endfacet"):
            if len(tri_verts) >= 3:
                ni = len(mesh.normals)
                if current_normal:
                    mesh.normals.append(current_normal)
                else:
                    mesh.normals.append(Vertex(0.0, 0.0, 1.0))
                mesh.faces.append(Face(
                    vertex_indices=tri_verts[:3],
                    normal_indices=[ni, ni, ni],
                ))
            tri_verts = []
            current_normal = None
    mesh.compute_bounds()
    if normalize_size is not None:
        mesh.normalize(normalize_size)
    mesh.materials["default"] = MeshMaterial()
    return mesh


# ---------------------------------------------------------------------------
# Primitive mesh generators
# ---------------------------------------------------------------------------

def make_cube(size: float = 1.0) -> Mesh:
    """Generate a unit cube mesh centred at origin."""
    h = size * 0.5
    m = Mesh(name="Cube")
    m.vertices = [
        Vertex(-h, -h, -h), Vertex(h, -h, -h), Vertex(h, h, -h), Vertex(-h, h, -h),
        Vertex(-h, -h, h), Vertex(h, -h, h), Vertex(h, h, h), Vertex(-h, h, h),
    ]
    m.normals = [
        Vertex(0, 0, -1), Vertex(0, 0, 1), Vertex(-1, 0, 0),
        Vertex(1, 0, 0), Vertex(0, -1, 0), Vertex(0, 1, 0),
    ]
    # 6 faces, each quad → 2 tris
    quads = [
        ([0, 3, 2, 1], 0), ([4, 5, 6, 7], 1), ([0, 4, 7, 3], 2),
        ([1, 2, 6, 5], 3), ([0, 1, 5, 4], 4), ([3, 7, 6, 2], 5),
    ]
    for vi_list, ni in quads:
        m.faces.append(Face(vertex_indices=[vi_list[0], vi_list[1], vi_list[2]], normal_indices=[ni, ni, ni]))
        m.faces.append(Face(vertex_indices=[vi_list[0], vi_list[2], vi_list[3]], normal_indices=[ni, ni, ni]))
    m.materials["default"] = MeshMaterial()
    m.compute_bounds()
    return m


def make_sphere(radius: float = 0.5, rings: int = 16, segments: int = 24) -> Mesh:
    """Generate a UV sphere mesh."""
    m = Mesh(name="Sphere")
    # Top pole
    m.vertices.append(Vertex(0, 0, radius))
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        sp = math.sin(phi)
        cp = math.cos(phi)
        for seg in range(segments):
            theta = 2.0 * math.pi * seg / segments
            m.vertices.append(Vertex(radius * sp * math.cos(theta), radius * sp * math.sin(theta), radius * cp))
    # Bottom pole
    m.vertices.append(Vertex(0, 0, -radius))

    # Top cap triangles
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        m.faces.append(Face(vertex_indices=[0, 1 + seg, 1 + next_seg]))
    # Middle quads as triangle pairs
    for ring in range(rings - 2):
        base = 1 + ring * segments
        for seg in range(segments):
            next_seg = (seg + 1) % segments
            i0 = base + seg
            i1 = base + next_seg
            i2 = base + segments + next_seg
            i3 = base + segments + seg
            m.faces.append(Face(vertex_indices=[i0, i3, i2]))
            m.faces.append(Face(vertex_indices=[i0, i2, i1]))
    # Bottom cap
    bottom_idx = len(m.vertices) - 1
    base = 1 + (rings - 2) * segments
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        m.faces.append(Face(vertex_indices=[bottom_idx, base + next_seg, base + seg]))

    m.materials["default"] = MeshMaterial()
    m.compute_bounds()
    return m


def make_plane(size: float = 1.0) -> Mesh:
    """Generate a flat plane in the XY plane."""
    h = size * 0.5
    m = Mesh(name="Plane")
    m.vertices = [Vertex(-h, -h, 0), Vertex(h, -h, 0), Vertex(h, h, 0), Vertex(-h, h, 0)]
    m.normals = [Vertex(0, 0, 1)]
    m.faces.append(Face(vertex_indices=[0, 1, 2], normal_indices=[0, 0, 0]))
    m.faces.append(Face(vertex_indices=[0, 2, 3], normal_indices=[0, 0, 0]))
    m.materials["default"] = MeshMaterial()
    m.compute_bounds()
    return m
