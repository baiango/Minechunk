from __future__ import annotations

import cProfile
import ctypes
import os
import math
import subprocess
import struct
import time
import sys
from collections import OrderedDict, deque
from dataclasses import dataclass
from pstats import Stats

import numpy as np
import wgpu
try:
    from wgpu.backends.wgpu_native import multi_draw_indirect as wgpu_native_multi_draw_indirect
except Exception:
    wgpu_native_multi_draw_indirect = None
from rendercanvas.auto import RenderCanvas, loop

from terrain_kernels import build_chunk_vertex_array_from_voxels
from voxel_world import CHUNK_SIZE, WORLD_HEIGHT, VoxelWorld
from terrain_backend import ChunkSurfaceGpuBatch, ChunkVoxelResult

try:
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover - only used outside kernprof
    def profile(func):
        return func


CHUNK_SAMPLE_SIZE = CHUNK_SIZE + 2
MAX_FACES_PER_CELL = 5
VERTICES_PER_FACE = 6
MAX_VERTICES_PER_CHUNK = CHUNK_SIZE * CHUNK_SIZE * MAX_FACES_PER_CELL * VERTICES_PER_FACE
VERTEX_STRIDE = 48
LIGHT_DIRECTION = (0.35, 0.90, 0.25)
DEPTH_FORMAT = "depth24plus"
CHUNK_PREP_RATE = 1000.0
CHUNK_PREP_TOKEN_CAP = 500.0
CHUNK_FORWARD_CONE_LATERAL_RATIO = 0.5
CHUNK_PREP_REQUEST_BUDGET_CAP = 16
CHUNK_PREP_REQUEST_BATCH_SIZE = 16
CHUNK_PREP_REORDER_YAW_DELTA = math.radians(10.0)
ENGINE_MODE_CPU = "cpu"
ENGINE_MODE_GPU = "gpu"
ENGINE_MODE = ENGINE_MODE_GPU
DEFAULT_RENDER_DISTANCE_BLOCKS = CHUNK_SIZE * 32
MERGED_TILE_SIZE_CHUNKS = 4
MERGED_TILE_MIN_AGE_SECONDS = 2.0
MERGED_TILE_MAX_CHUNKS = MERGED_TILE_SIZE_CHUNKS * MERGED_TILE_SIZE_CHUNKS
MAX_CACHED_CHUNKS = 4225
DEFAULT_MESH_BATCH_SIZE = 128
MESH_OUTPUT_SLAB_MIN_BYTES = 16 * 1024 * 1024
INDIRECT_DRAW_COMMAND_STRIDE = 16
GPU_VISIBILITY_WORKGROUP_SIZE = 64
MESH_VISIBILITY_RECORD_DTYPE = np.dtype([("bounds", np.float32, 4), ("draw", np.uint32, 4)])

# Indirect draws use first_vertex, which is counted in whole vertices, not bytes.
# Persistent slab suballocation can start at byte offsets that are aligned for storage
# bindings (for example 256 B) but not necessarily aligned to VERTEX_STRIDE (48 B).
# To preserve exact mesh slices with indirect draws, bind each slab at a shared byte
# offset equal to vertex_offset % VERTEX_STRIDE, then keep first_vertex relative to
# that binding offset.


def _build_tile_merge_shader() -> str:
    source_bindings = "\n".join(
        f"@group(0) @binding({index}) var<storage, read> src_{index}: VertexBuffer;"
        for index in range(MERGED_TILE_MAX_CHUNKS)
    )
    source_cases = "\n".join(
        f"        case {index}u: {{ return src_{index}.values[local_vertex]; }}"
        for index in range(MERGED_TILE_MAX_CHUNKS)
    )
    return f"""
struct Vertex {{
    position: vec4f,
    normal: vec4f,
    color: vec4f,
}}

struct MergeMeta {{
    vertex_count: u32,
    dst_first_vertex: u32,
    pad0: u32,
    pad1: u32,
}}

struct MergeMetaBuffer {{
    values: array<MergeMeta>,
}}

struct MergeParams {{
    chunk_count: u32,
    total_vertices: u32,
    pad0: u32,
    pad1: u32,
}}

struct VertexBuffer {{
    values: array<Vertex>,
}}

{source_bindings}
@group(0) @binding({MERGED_TILE_MAX_CHUNKS}) var<storage, read> merge_meta: MergeMetaBuffer;
@group(0) @binding({MERGED_TILE_MAX_CHUNKS + 1}) var<uniform> merge_params: MergeParams;
@group(0) @binding({MERGED_TILE_MAX_CHUNKS + 2}) var<storage, read_write> merged_vertices: VertexBuffer;

fn read_source_vertex(chunk_index: u32, local_vertex: u32) -> Vertex {{
    switch (chunk_index) {{
{source_cases}
        default: {{
            return src_0.values[local_vertex];
        }}
    }}
}}

@compute @workgroup_size(64)
fn combine_main(@builtin(global_invocation_id) gid: vec3u) {{
    if (gid.x >= merge_params.total_vertices) {{
        return;
    }}

    var chunk_index: u32 = 0u;
    var local_vertex: u32 = gid.x;
    var i: u32 = 0u;
    loop {{
        if (i >= merge_params.chunk_count) {{
            return;
        }}
        let vertex_count = merge_meta.values[i].vertex_count;
        if (local_vertex < vertex_count) {{
            chunk_index = i;
            break;
        }}
        local_vertex -= vertex_count;
        i += 1u;
    }}

    let dst_first_vertex = merge_meta.values[chunk_index].dst_first_vertex;
    merged_vertices.values[dst_first_vertex + local_vertex] = read_source_vertex(chunk_index, local_vertex);
}}
"""


TILE_MERGE_SHADER = _build_tile_merge_shader()
HUD_FONT_SCALE = 2
HUD_FONT_CHAR_WIDTH = 5
HUD_FONT_CHAR_HEIGHT = 9
HUD_PANEL_PADDING = 10
HUD_LINE_SPACING = 3
HUD_GLYPH_SPACING = 1
PROFILE_REPORT_INTERVAL = 1.0
FRAME_BREAKDOWN_SAMPLE_WINDOW = 120
SWAPCHAIN_MAX_FPS = 320
SWAPCHAIN_USE_VSYNC = False
SPRINT_FLY_SPEED = 500.0


COMPUTE_SHADER = """
struct ChunkParams {
    origin_and_scale: vec4f,
    counts_and_flags: vec4u,
}

struct Vertex {
    position: vec4f,
    normal: vec4f,
    color: vec4f,
}

struct HeightBuffer {
    values: array<u32>,
}

struct MaterialBuffer {
    values: array<u32>,
}

struct VertexBuffer {
    values: array<Vertex>,
}

struct CounterBuffer {
    value: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> heights: HeightBuffer;
@group(0) @binding(1) var<storage, read> materials: MaterialBuffer;
@group(0) @binding(2) var<storage, read_write> vertices: VertexBuffer;
@group(0) @binding(3) var<storage, read_write> vertex_count: CounterBuffer;
@group(0) @binding(4) var<uniform> params: ChunkParams;

fn material_color(material: u32, height: u32) -> vec3f {
    let altitude = clamp((f32(height) - 30.0) / 56.0, 0.0, 1.0);
    var color = vec3f(0.42, 0.68, 0.36);
    if (material == 1u) {
        color = vec3f(0.24, 0.22, 0.20);
    } else if (material == 2u) {
        color = vec3f(0.42, 0.40, 0.38);
    } else if (material == 3u) {
        color = vec3f(0.47, 0.31, 0.18);
    } else if (material == 4u) {
        color = mix(vec3f(0.18, 0.53, 0.18), vec3f(0.31, 0.68, 0.24), altitude);
    } else if (material == 5u) {
        color = vec3f(0.78, 0.71, 0.49);
    } else if (material == 6u) {
        color = vec3f(0.95, 0.97, 0.98);
    }
    return mix(color, vec3f(1.0, 1.0, 1.0), altitude * 0.08);
}

fn face_color(material: u32, height: u32, shade: f32) -> vec3f {
    return material_color(material, height) * shade;
}

fn emit_vertex(position: vec3f, normal: vec3f, color: vec3f, slot: u32) {
    vertices.values[slot] = Vertex(
        vec4f(position, 1.0),
        vec4f(normal, 0.0),
        vec4f(color, 1.0),
    );
}

fn emit_triangle(
    a: vec3f,
    b: vec3f,
    c: vec3f,
    normal: vec3f,
    color: vec3f,
) {
    let base = atomicAdd(&vertex_count.value, 3u);
    emit_vertex(a, normal, color, base + 0u);
    emit_vertex(b, normal, color, base + 1u);
    emit_vertex(c, normal, color, base + 2u);
}

fn emit_quad(
    p0: vec3f,
    p1: vec3f,
    p2: vec3f,
    p3: vec3f,
    normal: vec3f,
    color: vec3f,
) {
    emit_triangle(p0, p1, p2, normal, color);
    emit_triangle(p0, p2, p3, normal, color);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let sample_stride = params.counts_and_flags.x;
    let sample_size = params.counts_and_flags.y;
    let emit_west_face = params.counts_and_flags.z != 0u;
    let emit_north_face = params.counts_and_flags.w != 0u;
    if (gid.x >= sample_size - 1u || gid.y >= sample_size - 1u) {
        return;
    }

    let cell_x = gid.x;
    let cell_z = gid.y;
    let cell_index = cell_z * sample_stride + cell_x;
    let height = heights.values[cell_index];
    if (height == 0u) {
        return;
    }

    let east_height = heights.values[cell_index + 1u];
    let south_height = heights.values[cell_index + sample_stride];
    let material = materials.values[cell_index];
    let origin_x = params.origin_and_scale.x;
    let origin_z = params.origin_and_scale.y;
    let cell_size = params.origin_and_scale.z;

    let x0 = origin_x + f32(cell_x) * cell_size;
    let x1 = x0 + cell_size;
    let z0 = origin_z + f32(cell_z) * cell_size;
    let z1 = z0 + cell_size;
    let y0 = f32(height);
    let east_y = f32(east_height);
    let south_y = f32(south_height);

    let top = face_color(material, height, 1.0);
    let east = face_color(material, height, 0.80);
    let south = face_color(material, height, 0.72);
    let west = face_color(material, height, 0.64);
    let north = face_color(material, height, 0.60);

    emit_quad(
        vec3f(x0, y0, z0),
        vec3f(x1, y0, z0),
        vec3f(x1, y0, z1),
        vec3f(x0, y0, z1),
        vec3f(0.0, 1.0, 0.0),
        top,
    );

    if (height > east_height) {
        emit_quad(
            vec3f(x1, east_y, z0),
            vec3f(x1, y0, z0),
            vec3f(x1, y0, z1),
            vec3f(x1, east_y, z1),
            vec3f(1.0, 0.0, 0.0),
            east,
        );
    }

    if (height > south_height) {
        emit_quad(
            vec3f(x0, south_y, z1),
            vec3f(x0, y0, z1),
            vec3f(x1, y0, z1),
            vec3f(x1, south_y, z1),
            vec3f(0.0, 0.0, 1.0),
            south,
        );
    }

    if (emit_west_face && cell_x == 0u) {
        emit_quad(
            vec3f(x0, 0.0, z0),
            vec3f(x0, y0, z0),
            vec3f(x0, y0, z1),
            vec3f(x0, 0.0, z1),
            vec3f(-1.0, 0.0, 0.0),
            west,
        );
    }

    if (emit_north_face && cell_z == 0u) {
        emit_quad(
            vec3f(x0, 0.0, z0),
            vec3f(x0, y0, z0),
            vec3f(x1, y0, z0),
            vec3f(x1, 0.0, z0),
            vec3f(0.0, 0.0, -1.0),
            north,
        );
    }
}
"""


RENDER_SHADER = """
struct CameraUniform {
    position: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
    proj: vec4f,
}

struct VertexInput {
    @location(0) position: vec4f,
    @location(1) normal: vec4f,
    @location(2) color: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) color: vec3f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = input.position.xyz;
    let to_point = world - camera.position.xyz;
    let view_x = dot(to_point, camera.right.xyz);
    let view_y = dot(to_point, camera.up.xyz);
    let view_z = dot(to_point, camera.forward.xyz);

    if (view_z <= camera.proj.z) {
        out.position = vec4f(2.0, 2.0, 2.0, 1.0);
        out.normal = normalize(input.normal.xyz);
        out.color = input.color.xyz;
        return out;
    }

    let focal = camera.proj.x;
    let aspect = camera.proj.y;
    let near = camera.proj.z;
    let far = camera.proj.w;
    let clip_x = view_x * focal / aspect;
    let clip_y = view_y * focal;
    let clip_z = view_z * (far / (far - near)) - (near * far) / (far - near);
    out.position = vec4f(clip_x, clip_y, clip_z, view_z);
    out.normal = normalize(input.normal.xyz);
    out.color = input.color.xyz;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let light = normalize(vec3f(0.35, 0.90, 0.25));
    let diffuse = max(dot(input.normal, light), 0.0);
    let brightness = 0.32 + 0.68 * diffuse;
    return vec4f(input.color * brightness, 1.0);
}
"""


HUD_SHADER = """
struct VertexInput {
    @location(0) position: vec4f,
    @location(1) normal: vec4f,
    @location(2) color: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = input.position;
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}
"""


VOXEL_MESH_SHADER = """
struct ChunkParams {
    origin_and_scale: vec4f,
    counts_and_flags: vec4u,
}

struct Vertex {
    position: vec4f,
    normal: vec4f,
    color: vec4f,
}

struct BlockBuffer {
    values: array<u32>,
}

struct MaterialBuffer {
    values: array<u32>,
}

struct VertexBuffer {
    values: array<Vertex>,
}

struct CounterBuffer {
    value: atomic<u32>,
}

@group(0) @binding(0) var<storage, read> blocks: BlockBuffer;
@group(0) @binding(1) var<storage, read> materials: MaterialBuffer;
@group(0) @binding(2) var<storage, read_write> vertices: VertexBuffer;
@group(0) @binding(3) var<storage, read_write> vertex_count: CounterBuffer;
@group(0) @binding(4) var<uniform> params: ChunkParams;

fn terrain_color(height: u32) -> vec3f {
    let altitude = clamp((f32(height) - 30.0) / 56.0, 0.0, 1.0);
    if (height <= 14u) {
        return vec3f(0.78, 0.71, 0.49);
    }
    if (height >= 90u) {
        return vec3f(0.95, 0.97, 0.98);
    }
    return mix(vec3f(0.18, 0.53, 0.18), vec3f(0.31, 0.68, 0.24), altitude);
}

fn material_color(material: u32, height: u32) -> vec3f {
    if (material == 1u) {
        return vec3f(0.24, 0.22, 0.20);
    }
    if (material == 2u) {
        return vec3f(0.42, 0.40, 0.38);
    }
    if (material == 3u) {
        return vec3f(0.47, 0.31, 0.18);
    }
    if (material == 4u) {
        return terrain_color(height);
    }
    if (material == 5u) {
        return vec3f(0.78, 0.71, 0.49);
    }
    if (material == 6u) {
        return vec3f(0.95, 0.97, 0.98);
    }
    return terrain_color(height);
}

fn face_color(material: u32, height: u32, shade: f32) -> vec3f {
    return material_color(material, height) * shade;
}

fn emit_vertex(position: vec3f, normal: vec3f, color: vec3f, slot: u32) {
    vertices.values[slot] = Vertex(
        vec4f(position, 1.0),
        vec4f(normal, 0.0),
        vec4f(color, 1.0),
    );
}

fn emit_triangle(a: vec3f, b: vec3f, c: vec3f, normal: vec3f, color: vec3f) {
    let base = atomicAdd(&vertex_count.value, 3u);
    emit_vertex(a, normal, color, base + 0u);
    emit_vertex(b, normal, color, base + 1u);
    emit_vertex(c, normal, color, base + 2u);
}

fn emit_quad(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, normal: vec3f, color: vec3f) {
    emit_triangle(p0, p1, p2, normal, color);
    emit_triangle(p0, p2, p3, normal, color);
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let sample_size = params.counts_and_flags.x;
    let height_limit = params.counts_and_flags.y;
    if (gid.x >= sample_size - 2u || gid.y >= sample_size - 2u || gid.z >= height_limit) {
        return;
    }

    let local_x = gid.x + 1u;
    let local_z = gid.y + 1u;
    let y = gid.z;
    let plane = sample_size * sample_size;
    let cell_index = y * plane + local_z * sample_size + local_x;
    if (blocks.values[cell_index] == 0u) {
        return;
    }

    let origin_x = params.origin_and_scale.x;
    let origin_z = params.origin_and_scale.y;
    let cell_size = params.origin_and_scale.z;
    let x0 = origin_x + f32(local_x - 1u) * cell_size;
    let x1 = x0 + cell_size;
    let z0 = origin_z + f32(local_z - 1u) * cell_size;
    let z1 = z0 + cell_size;
    let y0 = f32(y);
    let y1 = y0 + cell_size;

    let material = materials.values[cell_index];
    let top = face_color(material, y, 1.0);
    let east = face_color(material, y, 0.80);
    let west = face_color(material, y, 0.64);
    let south = face_color(material, y, 0.72);
    let north = face_color(material, y, 0.60);
    let bottom = face_color(material, y, 0.50);

    if (y == height_limit - 1u || blocks.values[cell_index + plane] == 0u) {
        emit_quad(
            vec3f(x0, y1, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y1, z1),
            vec3f(x0, y1, z1),
            vec3f(0.0, 1.0, 0.0),
            top,
        );
    }

    if (y == 0u || blocks.values[cell_index - plane] == 0u) {
        emit_quad(
            vec3f(x0, y0, z0),
            vec3f(x0, y0, z1),
            vec3f(x1, y0, z1),
            vec3f(x1, y0, z0),
            vec3f(0.0, -1.0, 0.0),
            bottom,
        );
    }

    if (blocks.values[cell_index + 1u] == 0u) {
        emit_quad(
            vec3f(x1, y0, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y1, z1),
            vec3f(x1, y0, z1),
            vec3f(1.0, 0.0, 0.0),
            east,
        );
    }

    if (blocks.values[cell_index - 1u] == 0u) {
        emit_quad(
            vec3f(x0, y0, z0),
            vec3f(x0, y0, z1),
            vec3f(x0, y1, z1),
            vec3f(x0, y1, z0),
            vec3f(-1.0, 0.0, 0.0),
            west,
        );
    }

    if (blocks.values[cell_index + sample_size] == 0u) {
        emit_quad(
            vec3f(x0, y0, z1),
            vec3f(x1, y0, z1),
            vec3f(x1, y1, z1),
            vec3f(x0, y1, z1),
            vec3f(0.0, 0.0, 1.0),
            south,
        );
    }

    if (blocks.values[cell_index - sample_size] == 0u) {
        emit_quad(
            vec3f(x0, y0, z0),
            vec3f(x0, y1, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y0, z0),
            vec3f(0.0, 0.0, -1.0),
            north,
        );
    }
}
"""


VOXEL_SURFACE_EXPAND_SHADER = """
struct ExpandParams {
    sample_size: u32,
    height_limit: u32,
    chunk_count: u32,
    chunk_size: u32,
}

struct HeightBuffer {
    values: array<u32>,
}

struct MaterialBuffer {
    values: array<u32>,
}

struct BlockBuffer {
    values: array<u32>,
}

struct VoxelMaterialBuffer {
    values: array<u32>,
}

@group(0) @binding(0) var<storage, read> surface_heights: HeightBuffer;
@group(0) @binding(1) var<storage, read> surface_materials: MaterialBuffer;
@group(0) @binding(2) var<storage, read_write> blocks: BlockBuffer;
@group(0) @binding(3) var<storage, read_write> voxel_materials: VoxelMaterialBuffer;
@group(0) @binding(4) var<uniform> params: ExpandParams;

@compute @workgroup_size(8, 8, 1)
fn expand_main(@builtin(global_invocation_id) gid: vec3u) {
    let sample_size = params.sample_size;
    let height_limit = params.height_limit;
    let chunk_count = params.chunk_count;
    let chunk_stride = height_limit * sample_size * sample_size;
    if (gid.x >= sample_size || gid.y >= sample_size || gid.z >= chunk_count * height_limit) {
        return;
    }

    let chunk_index = gid.z / height_limit;
    let y = gid.z - chunk_index * height_limit;
    let cell_index = gid.y * sample_size + gid.x;
    let surface_index = chunk_index * (sample_size * sample_size) + cell_index;
    let voxel_index = chunk_index * chunk_stride + y * sample_size * sample_size + cell_index;
    let surface_height = surface_heights.values[surface_index];

    if (y >= surface_height) {
        blocks.values[voxel_index] = 0u;
        voxel_materials.values[voxel_index] = 0u;
        return;
    }

    var material = surface_materials.values[surface_index];
    if (y == 0u) {
        material = 1u;
    } else if (y < surface_height - 4u) {
        material = 2u;
    } else if (y < surface_height - 1u) {
        material = 3u;
    }

    blocks.values[voxel_index] = 1u;
    voxel_materials.values[voxel_index] = material;
}
"""


VOXEL_MESH_BATCH_SHADER = """
struct BatchParams {
    counts_and_flags: vec4u,
}

struct Vertex {
    position: vec4f,
    normal: vec4f,
    color: vec4f,
}

struct BlockBuffer {
    values: array<u32>,
}

struct MaterialBuffer {
    values: array<u32>,
}

struct ChunkCoordBuffer {
    values: array<vec2i>,
}

struct CountBuffer {
    values: array<u32>,
}

struct AtomicCountBuffer {
    values: array<atomic<u32>>,
}

struct VertexBuffer {
    values: array<Vertex>,
}

@group(0) @binding(0) var<storage, read> blocks: BlockBuffer;
@group(0) @binding(1) var<storage, read> materials: MaterialBuffer;
@group(0) @binding(2) var<storage, read> chunk_coords: ChunkCoordBuffer;
@group(0) @binding(3) var<storage, read_write> column_totals: CountBuffer;
@group(0) @binding(4) var<storage, read_write> chunk_totals: AtomicCountBuffer;
@group(0) @binding(5) var<storage, read_write> chunk_offsets: CountBuffer;
@group(0) @binding(6) var<storage, read_write> vertices: VertexBuffer;
@group(0) @binding(8) var<uniform> params: BatchParams;

var<workgroup> face_counts: array<u32, 128>;
var<workgroup> prefix_offsets: array<u32, 128>;

fn terrain_color(height: u32) -> vec3f {
    let altitude = clamp((f32(height) - 30.0) / 56.0, 0.0, 1.0);
    if (height <= 14u) {
        return vec3f(0.78, 0.71, 0.49);
    }
    if (height >= 90u) {
        return vec3f(0.95, 0.97, 0.98);
    }
    return mix(vec3f(0.18, 0.53, 0.18), vec3f(0.31, 0.68, 0.24), altitude);
}

fn material_color(material: u32, height: u32) -> vec3f {
    if (material == 1u) {
        return vec3f(0.24, 0.22, 0.20);
    }
    if (material == 2u) {
        return vec3f(0.42, 0.40, 0.38);
    }
    if (material == 3u) {
        return vec3f(0.47, 0.31, 0.18);
    }
    if (material == 4u) {
        return terrain_color(height);
    }
    if (material == 5u) {
        return vec3f(0.78, 0.71, 0.49);
    }
    if (material == 6u) {
        return vec3f(0.95, 0.97, 0.98);
    }
    return terrain_color(height);
}

fn face_color(material: u32, height: u32, shade: f32) -> vec3f {
    return material_color(material, height) * shade;
}

fn emit_vertex(position: vec3f, normal: vec3f, color: vec3f, slot: u32) {
    vertices.values[slot] = Vertex(vec4f(position, 1.0), vec4f(normal, 0.0), vec4f(color, 1.0));
}

fn emit_triangle_at(base: u32, a: vec3f, b: vec3f, c: vec3f, normal: vec3f, color: vec3f) {
    emit_vertex(a, normal, color, base + 0u);
    emit_vertex(b, normal, color, base + 1u);
    emit_vertex(c, normal, color, base + 2u);
}

fn emit_quad_at(base: u32, p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, normal: vec3f, color: vec3f) {
    emit_triangle_at(base + 0u, p0, p1, p2, normal, color);
    emit_triangle_at(base + 3u, p0, p2, p3, normal, color);
}

fn voxel_face_count(
    chunk_base: u32,
    sample_size: u32,
    plane: u32,
    local_x: u32,
    local_z: u32,
    y: u32,
    height_limit: u32,
) -> u32 {
    let cell_index = chunk_base + y * plane + local_z * sample_size + local_x;
    if (blocks.values[cell_index] == 0u) {
        return 0u;
    }
    var count = 0u;
    if (y == height_limit - 1u || blocks.values[cell_index + plane] == 0u) {
        count = count + 6u;
    }
    if (y == 0u || blocks.values[cell_index - plane] == 0u) {
        count = count + 6u;
    }
    if (blocks.values[cell_index + 1u] == 0u) {
        count = count + 6u;
    }
    if (blocks.values[cell_index - 1u] == 0u) {
        count = count + 6u;
    }
    if (blocks.values[cell_index + sample_size] == 0u) {
        count = count + 6u;
    }
    if (blocks.values[cell_index - sample_size] == 0u) {
        count = count + 6u;
    }
    return count;
}

fn emit_voxel_faces(
    base: u32,
    chunk_base: u32,
    sample_size: u32,
    plane: u32,
    local_x: u32,
    local_z: u32,
    y: u32,
    height_limit: u32,
    origin_x: f32,
    origin_z: f32,
) {
    let cell_index = chunk_base + y * plane + local_z * sample_size + local_x;
    if (blocks.values[cell_index] == 0u) {
        return;
    }

    let x0 = origin_x + f32(local_x - 1u);
    let x1 = x0 + 1.0;
    let z0 = origin_z + f32(local_z - 1u);
    let z1 = z0 + 1.0;
    let y0 = f32(y);
    let y1 = y0 + 1.0;

    let material = materials.values[cell_index];
    let top = face_color(material, y, 1.0);
    let east = face_color(material, y, 0.80);
    let west = face_color(material, y, 0.64);
    let south = face_color(material, y, 0.72);
    let north = face_color(material, y, 0.60);
    let bottom = face_color(material, y, 0.50);
    var face_base = base;

    if (y == height_limit - 1u || blocks.values[cell_index + plane] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x0, y1, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y1, z1),
            vec3f(x0, y1, z1),
            vec3f(0.0, 1.0, 0.0),
            top,
        );
        face_base = face_base + 6u;
    }
    if (y == 0u || blocks.values[cell_index - plane] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z0),
            vec3f(x0, y0, z1),
            vec3f(x1, y0, z1),
            vec3f(x1, y0, z0),
            vec3f(0.0, -1.0, 0.0),
            bottom,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index + 1u] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x1, y0, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y1, z1),
            vec3f(x1, y0, z1),
            vec3f(1.0, 0.0, 0.0),
            east,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index - 1u] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z0),
            vec3f(x0, y0, z1),
            vec3f(x0, y1, z1),
            vec3f(x0, y1, z0),
            vec3f(-1.0, 0.0, 0.0),
            west,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index + sample_size] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z1),
            vec3f(x1, y0, z1),
            vec3f(x1, y1, z1),
            vec3f(x0, y1, z1),
            vec3f(0.0, 0.0, 1.0),
            south,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index - sample_size] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z0),
            vec3f(x0, y1, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y0, z0),
            vec3f(0.0, 0.0, -1.0),
            north,
        );
    }
}

@compute @workgroup_size(1, 1, 128)
fn count_main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let sample_size = params.counts_and_flags.x;
    let height_limit = params.counts_and_flags.y;
    let chunk_count = params.counts_and_flags.z;
    let columns_per_side = params.counts_and_flags.w;
    if (wid.x >= columns_per_side || wid.y >= columns_per_side || wid.z >= chunk_count) {
        return;
    }
    let y = lid.z;
    if (y >= height_limit) {
        return;
    }
    let plane = sample_size * sample_size;
    let chunk_stride = height_limit * plane;
    let chunk_base = wid.z * chunk_stride;
    let local_x = wid.x + 1u;
    let local_z = wid.y + 1u;
    face_counts[y] = voxel_face_count(chunk_base, sample_size, plane, local_x, local_z, y, height_limit);
    workgroupBarrier();
    if (y == 0u) {
        var total = 0u;
        for (var i = 0u; i < height_limit; i = i + 1u) {
            total = total + face_counts[i];
        }
        let column_plane = columns_per_side * columns_per_side;
        let column_index = wid.y * columns_per_side + wid.x;
        column_totals.values[wid.z * column_plane + column_index] = total;
        atomicAdd(&chunk_totals.values[wid.z], total);
    }
}

@compute @workgroup_size(1, 1, 1)
fn scan_main(
    @builtin(workgroup_id) wid: vec3u,
) {
    let chunk_count = params.counts_and_flags.z;
    let columns_per_side = params.counts_and_flags.w;
    if (wid.z >= chunk_count) {
        return;
    }
    let chunk_index = wid.z;
    var chunk_prefix = 0u;
    for (var i = 0u; i < chunk_index; i = i + 1u) {
        chunk_prefix = chunk_prefix + atomicLoad(&chunk_totals.values[i]);
    }
    chunk_offsets.values[chunk_index] = chunk_prefix;
}

@compute @workgroup_size(1, 1, 128)
fn emit_main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let sample_size = params.counts_and_flags.x;
    let height_limit = params.counts_and_flags.y;
    let chunk_count = params.counts_and_flags.z;
    let columns_per_side = params.counts_and_flags.w;
    if (wid.x >= columns_per_side || wid.y >= columns_per_side || wid.z >= chunk_count) {
        return;
    }
    let y = lid.z;
    if (y >= height_limit) {
        return;
    }

    let plane = sample_size * sample_size;
    let chunk_stride = height_limit * plane;
    let chunk_index = wid.z;
    let chunk_base = chunk_index * chunk_stride;
    let local_x = wid.x + 1u;
    let local_z = wid.y + 1u;
    face_counts[y] = voxel_face_count(chunk_base, sample_size, plane, local_x, local_z, y, height_limit);
    workgroupBarrier();
    if (y == 0u) {
        var prefix = 0u;
        for (var i = 0u; i < height_limit; i = i + 1u) {
            prefix_offsets[i] = prefix;
            prefix = prefix + face_counts[i];
        }
    }
    workgroupBarrier();

    let column_plane = columns_per_side * columns_per_side;
    let column_index = wid.y * columns_per_side + wid.x;
    var column_prefix = 0u;
    let column_base_index = chunk_index * column_plane;
    for (var i = 0u; i < column_index; i = i + 1u) {
        column_prefix = column_prefix + column_totals.values[column_base_index + i];
    }
    let column_base = chunk_offsets.values[chunk_index] + column_prefix;
    let origin = chunk_coords.values[chunk_index];
    let origin_x = f32(origin.x * i32(params.counts_and_flags.w));
    let origin_z = f32(origin.y * i32(params.counts_and_flags.w));
    emit_voxel_faces(column_base + prefix_offsets[y], chunk_base, sample_size, plane, local_x, local_z, y, height_limit, origin_x, origin_z);
}
"""


GPU_VISIBILITY_SHADER = """
struct CameraUniform {
    position: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
    proj: vec4f,
}

struct VisibilityParams {
    mesh_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct MeshRecord {
    bounds: vec4f,
    draw: vec4u,
}

struct MeshRecordBuffer {
    values: array<MeshRecord>,
}

struct IndirectCommandBuffer {
    values: array<vec4u>,
}

@group(0) @binding(0) var<storage, read> mesh_records: MeshRecordBuffer;
@group(0) @binding(1) var<storage, read_write> indirect_commands: IndirectCommandBuffer;
@group(0) @binding(2) var<uniform> camera: CameraUniform;
@group(0) @binding(3) var<uniform> params: VisibilityParams;

fn sphere_visible(bounds: vec4f) -> bool {
    let center = bounds.xyz;
    let radius = bounds.w;
    let relative = center - camera.position.xyz;
    let view_x = dot(relative, camera.right.xyz);
    let view_y = dot(relative, camera.up.xyz);
    let view_z = dot(relative, camera.forward.xyz);
    let focal = camera.proj.x;
    let aspect = camera.proj.y;
    let near = camera.proj.z;
    let far = camera.proj.w;

    if (view_z + radius <= near) {
        return false;
    }
    if (view_z - radius >= far) {
        return false;
    }

    let depth = max(view_z, near);
    let half_width = depth * aspect / focal;
    let half_height = depth / focal;
    if (view_x < -half_width - radius || view_x > half_width + radius) {
        return false;
    }
    if (view_y < -half_height - radius || view_y > half_height + radius) {
        return false;
    }
    return true;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let mesh_index = gid.x;
    if (mesh_index >= params.mesh_count) {
        return;
    }
    let record = mesh_records.values[mesh_index];
    if (sphere_visible(record.bounds)) {
        indirect_commands.values[mesh_index] = record.draw;
    } else {
        indirect_commands.values[mesh_index] = vec4u(0u, 1u, 0u, 0u);
    }
}
"""


_HUD_FONT_FALLBACK: dict[str, tuple[str, ...]] = {
    " ": ("000", "000", "000", "000", "000"),
    "?": ("111", "001", "010", "000", "010"),
    ".": ("000", "000", "000", "000", "010"),
    ",": ("000", "000", "000", "010", "100"),
    ":": ("000", "010", "000", "010", "000"),
    "-": ("000", "000", "111", "000", "000"),
    "_": ("000", "000", "000", "000", "111"),
    "/": ("001", "001", "010", "100", "100"),
    "(": ("010", "100", "100", "100", "010"),
    ")": ("010", "001", "001", "001", "010"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "100", "100"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"),
    "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"),
    "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "101", "101"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "101", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("010", "101", "101", "101", "010"),
    "P": ("110", "101", "110", "100", "100"),
    "Q": ("010", "101", "101", "111", "011"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"),
    "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "Z": ("111", "001", "010", "100", "111"),
}


def _find_hud_font_path() -> str | None:
    try:
        font_path = subprocess.check_output(
            ["fc-match", "-f", "%{file}\n", "Menlo:style=Regular"],
            text=True,
        ).strip()
        if font_path and os.path.exists(font_path):
            return font_path
    except Exception:
        pass
    for candidate in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _build_hud_font_from_freetype() -> dict[str, tuple[str, ...]]:
    font_path = _find_hud_font_path()
    if not font_path:
        raise RuntimeError("No usable HUD font file found")

    library_path = "/opt/homebrew/opt/freetype/lib/libfreetype.dylib"
    if not os.path.exists(library_path):
        raise RuntimeError("FreeType library not available")

    freetype = ctypes.CDLL(library_path)
    c_void_p = ctypes.c_void_p

    class FT_Generic(ctypes.Structure):
        _fields_ = [("data", c_void_p), ("finalizer", c_void_p)]

    class FT_Vector(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class FT_BBox(ctypes.Structure):
        _fields_ = [
            ("xMin", ctypes.c_long),
            ("yMin", ctypes.c_long),
            ("xMax", ctypes.c_long),
            ("yMax", ctypes.c_long),
        ]

    class FT_Bitmap(ctypes.Structure):
        _fields_ = [
            ("rows", ctypes.c_uint),
            ("width", ctypes.c_uint),
            ("pitch", ctypes.c_int),
            ("buffer", ctypes.POINTER(ctypes.c_ubyte)),
            ("num_grays", ctypes.c_ushort),
            ("pixel_mode", ctypes.c_ubyte),
            ("palette_mode", ctypes.c_ubyte),
            ("palette", c_void_p),
        ]

    class FT_Glyph_Metrics(ctypes.Structure):
        _fields_ = [
            ("width", ctypes.c_long),
            ("height", ctypes.c_long),
            ("horiBearingX", ctypes.c_long),
            ("horiBearingY", ctypes.c_long),
            ("horiAdvance", ctypes.c_long),
            ("vertBearingX", ctypes.c_long),
            ("vertBearingY", ctypes.c_long),
            ("vertAdvance", ctypes.c_long),
        ]

    class FT_Outline(ctypes.Structure):
        _fields_ = [
            ("n_contours", ctypes.c_ushort),
            ("n_points", ctypes.c_ushort),
            ("points", c_void_p),
            ("tags", c_void_p),
            ("contours", c_void_p),
            ("flags", ctypes.c_int),
        ]

    class FT_SizeRec(ctypes.Structure):
        pass

    class FT_FaceRec(ctypes.Structure):
        pass

    class FT_GlyphSlotRec(ctypes.Structure):
        pass

    FT_Library = c_void_p
    FT_Size = ctypes.POINTER(FT_SizeRec)
    FT_CharMap = c_void_p
    FT_Driver = c_void_p
    FT_Memory = c_void_p
    FT_Stream = c_void_p
    FT_ListRec = c_void_p
    FT_Face_Internal = c_void_p
    FT_Slot_Internal = c_void_p
    FT_GlyphSlot = ctypes.POINTER(FT_GlyphSlotRec)
    FT_Face = ctypes.POINTER(FT_FaceRec)

    class FT_Size_Metrics(ctypes.Structure):
        _fields_ = [
            ("x_ppem", ctypes.c_ushort),
            ("y_ppem", ctypes.c_ushort),
            ("x_scale", ctypes.c_long),
            ("y_scale", ctypes.c_long),
            ("ascender", ctypes.c_long),
            ("descender", ctypes.c_long),
            ("height", ctypes.c_long),
            ("max_advance", ctypes.c_long),
        ]

    FT_SizeRec._fields_ = [
        ("face", FT_Face),
        ("generic", FT_Generic),
        ("metrics", FT_Size_Metrics),
        ("internal", c_void_p),
    ]

    FT_GlyphSlotRec._fields_ = [
        ("library", FT_Library),
        ("face", FT_Face),
        ("next", FT_GlyphSlot),
        ("glyph_index", ctypes.c_uint),
        ("generic", FT_Generic),
        ("metrics", FT_Glyph_Metrics),
        ("linearHoriAdvance", ctypes.c_long),
        ("linearVertAdvance", ctypes.c_long),
        ("advance", FT_Vector),
        ("format", ctypes.c_uint),
        ("bitmap", FT_Bitmap),
        ("bitmap_left", ctypes.c_int),
        ("bitmap_top", ctypes.c_int),
        ("outline", FT_Outline),
        ("num_subglyphs", ctypes.c_uint),
        ("subglyphs", c_void_p),
        ("control_data", c_void_p),
        ("control_len", ctypes.c_long),
        ("lsb_delta", ctypes.c_long),
        ("rsb_delta", ctypes.c_long),
        ("other", c_void_p),
        ("internal", FT_Slot_Internal),
    ]

    FT_FaceRec._fields_ = [
        ("num_faces", ctypes.c_long),
        ("face_index", ctypes.c_long),
        ("face_flags", ctypes.c_long),
        ("style_flags", ctypes.c_long),
        ("num_glyphs", ctypes.c_long),
        ("family_name", ctypes.c_char_p),
        ("style_name", ctypes.c_char_p),
        ("num_fixed_sizes", ctypes.c_int),
        ("available_sizes", c_void_p),
        ("num_charmaps", ctypes.c_int),
        ("charmaps", c_void_p),
        ("generic", FT_Generic),
        ("bbox", FT_BBox),
        ("units_per_EM", ctypes.c_ushort),
        ("ascender", ctypes.c_short),
        ("descender", ctypes.c_short),
        ("height", ctypes.c_short),
        ("max_advance_width", ctypes.c_short),
        ("max_advance_height", ctypes.c_short),
        ("underline_position", ctypes.c_short),
        ("underline_thickness", ctypes.c_short),
        ("glyph", FT_GlyphSlot),
        ("size", FT_Size),
        ("charmap", FT_CharMap),
        ("driver", FT_Driver),
        ("memory", FT_Memory),
        ("stream", FT_Stream),
        ("sizes_list", FT_ListRec),
        ("autohint", FT_Generic),
        ("extensions", c_void_p),
        ("internal", FT_Face_Internal),
    ]

    freetype.FT_Init_FreeType.argtypes = [ctypes.POINTER(FT_Library)]
    freetype.FT_Init_FreeType.restype = ctypes.c_int
    freetype.FT_New_Face.argtypes = [FT_Library, ctypes.c_char_p, ctypes.c_long, ctypes.POINTER(FT_Face)]
    freetype.FT_New_Face.restype = ctypes.c_int
    freetype.FT_Done_Face.argtypes = [FT_Face]
    freetype.FT_Done_Face.restype = ctypes.c_int
    freetype.FT_Done_FreeType.argtypes = [FT_Library]
    freetype.FT_Done_FreeType.restype = ctypes.c_int
    freetype.FT_Set_Pixel_Sizes.argtypes = [FT_Face, ctypes.c_uint, ctypes.c_uint]
    freetype.FT_Set_Pixel_Sizes.restype = ctypes.c_int
    freetype.FT_Load_Char.argtypes = [FT_Face, ctypes.c_ulong, ctypes.c_int]
    freetype.FT_Load_Char.restype = ctypes.c_int

    FT_LOAD_RENDER = 0x4
    lib_obj = FT_Library()
    face = FT_Face()
    error = freetype.FT_Init_FreeType(ctypes.byref(lib_obj))
    if error != 0:
        raise RuntimeError(f"FT_Init_FreeType failed: {error}")
    try:
        error = freetype.FT_New_Face(lib_obj, font_path.encode("utf-8"), 0, ctypes.byref(face))
        if error != 0:
            raise RuntimeError(f"FT_New_Face failed: {error}")
        try:
            error = freetype.FT_Set_Pixel_Sizes(face, 0, 7)
            if error != 0:
                raise RuntimeError(f"FT_Set_Pixel_Sizes failed: {error}")
            size_metrics = face.contents.size.contents.metrics
            cell_width = max(1, int(round(size_metrics.max_advance / 64.0)) + 1)
            cell_height = max(1, int(round((size_metrics.ascender - size_metrics.descender) / 64.0)))
            baseline = int(round(size_metrics.ascender / 64.0))
            font: dict[str, tuple[str, ...]] = {}
            for code in range(32, 127):
                char = chr(code)
                error = freetype.FT_Load_Char(face, code, FT_LOAD_RENDER)
                if error != 0:
                    font[char] = _HUD_FONT_FALLBACK.get(char, _HUD_FONT_FALLBACK["?"])
                    continue
                slot = face.contents.glyph.contents
                bitmap = slot.bitmap
                rows = [["0"] * cell_width for _ in range(cell_height)]
                if bitmap.width > 0 and bitmap.rows > 0 and bool(bitmap.buffer):
                    buffer = ctypes.cast(bitmap.buffer, ctypes.POINTER(ctypes.c_ubyte))
                    row_offset = baseline - slot.bitmap_top
                    for y in range(bitmap.rows):
                        dest_y = row_offset + y
                        if not (0 <= dest_y < cell_height):
                            continue
                        for x in range(bitmap.width):
                            dest_x = slot.bitmap_left + x
                            if not (0 <= dest_x < cell_width):
                                continue
                            value = buffer[y * bitmap.pitch + x]
                            if value > 64:
                                rows[dest_y][dest_x] = "1"
                font[char] = tuple("".join(row) for row in rows)
            if "?" not in font:
                font["?"] = _HUD_FONT_FALLBACK["?"]
            return font
        finally:
            freetype.FT_Done_Face(face)
    finally:
        freetype.FT_Done_FreeType(lib_obj)


def _build_hud_font() -> dict[str, tuple[str, ...]]:
    try:
        return _build_hud_font_from_freetype()
    except Exception:
        return dict(_HUD_FONT_FALLBACK)


HUD_FONT = _build_hud_font()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def normalize3(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vector
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0.0:
        return 0.0, 0.0, 0.0
    return x / length, y / length, z / length


def dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def pack_camera_uniform(
    position: tuple[float, float, float],
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    forward: tuple[float, float, float],
    focal: float,
    aspect: float,
    near: float,
    far: float,
    light_dir: tuple[float, float, float],
) -> bytes:
    return struct.pack(
        "<20f",
        position[0],
        position[1],
        position[2],
        0.0,
        right[0],
        right[1],
        right[2],
        0.0,
        up[0],
        up[1],
        up[2],
        0.0,
        forward[0],
        forward[1],
        forward[2],
        0.0,
        focal,
        aspect,
        near,
        far,
    )
def pack_vertex(
    position: tuple[float, float, float],
    normal: tuple[float, float, float],
    color: tuple[float, float, float],
    alpha: float = 1.0,
) -> bytes:
    return struct.pack(
        "<4f4f4f",
        position[0],
        position[1],
        position[2],
        1.0,
        normal[0],
        normal[1],
        normal[2],
        0.0,
        color[0],
        color[1],
        color[2],
        alpha,
    )


def _hud_glyph_rows(char: str) -> tuple[str, ...]:
    return HUD_FONT.get(char, HUD_FONT["?"])


def _screen_to_ndc(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    return (x / width) * 2.0 - 1.0, 1.0 - (y / height) * 2.0


def forward_vector(yaw: float, pitch: float) -> tuple[float, float, float]:
    cp = math.cos(pitch)
    return math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp


def flat_forward_vector(yaw: float) -> tuple[float, float, float]:
    return math.sin(yaw), 0.0, math.cos(yaw)


def right_vector(yaw: float) -> tuple[float, float, float]:
    return -math.cos(yaw), 0.0, math.sin(yaw)


@dataclass
class Camera:
    position: list[float]
    yaw: float
    pitch: float
    move_speed: float = 34.0
    look_speed: float = 0.0035
    sprint_multiplier: float = 2.25

    def clamp_pitch(self) -> None:
        self.pitch = clamp(self.pitch, -1.45, 1.45)


@dataclass
class ChunkMesh:
    chunk_x: int
    chunk_z: int
    vertex_count: int
    vertex_buffer: wgpu.GPUBuffer
    max_height: int
    vertex_offset: int = 0
    created_at: float = 0.0
    allocation_id: int | None = None
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    binding_offset: int = 0
    first_vertex: int = 0

    def __post_init__(self) -> None:
        half_chunk = CHUNK_SIZE * 0.5
        half_height = float(self.max_height) * 0.5
        self.bounds = (
            float(self.chunk_x * CHUNK_SIZE + half_chunk),
            float(half_height),
            float(self.chunk_z * CHUNK_SIZE + half_chunk),
            float(math.sqrt(half_chunk * half_chunk * 2.0 + half_height * half_height)),
        )
        self.binding_offset = int(self.vertex_offset % VERTEX_STRIDE)
        self.first_vertex = int((self.vertex_offset - self.binding_offset) // VERTEX_STRIDE)


@dataclass
class MeshOutputSlab:
    slab_id: int
    buffer: wgpu.GPUBuffer
    size_bytes: int
    free_ranges: list[tuple[int, int]]


@dataclass
class MeshBufferAllocation:
    allocation_id: int
    buffer: wgpu.GPUBuffer
    offset_bytes: int
    size_bytes: int
    slab_id: int | None = None
    refcount: int = 0


@dataclass
class ChunkRenderBatch:
    signature: tuple[tuple[int, int], ...]
    vertex_count: int
    vertex_buffer: wgpu.GPUBuffer


@dataclass
class ChunkDrawBatch:
    vertex_buffer: wgpu.GPUBuffer
    binding_offset: int
    vertex_count: int
    first_vertex: int
    bounds: tuple[float, float, float, float]
    chunk_count: int = 1


@dataclass
class AsyncVoxelMeshBatchResources:
    sample_size: int
    height_limit: int
    chunk_capacity: int
    column_capacity: int
    blocks_buffer: wgpu.GPUBuffer
    materials_buffer: wgpu.GPUBuffer
    coords_buffer: wgpu.GPUBuffer
    column_totals_buffer: wgpu.GPUBuffer
    chunk_totals_buffer: wgpu.GPUBuffer
    chunk_offsets_buffer: wgpu.GPUBuffer
    params_buffer: wgpu.GPUBuffer
    readback_buffer: wgpu.GPUBuffer
    coords_array: np.ndarray
    zero_counts_array: np.ndarray
    count_bind_group: object | None = None


@dataclass
class PendingChunkMeshBatch:
    chunk_coords: list[tuple[int, int]]
    chunk_count: int
    sample_size: int
    height_limit: int
    columns_per_side: int
    blocks_buffer: wgpu.GPUBuffer
    materials_buffer: wgpu.GPUBuffer
    coords_buffer: wgpu.GPUBuffer
    column_totals_buffer: wgpu.GPUBuffer
    chunk_totals_buffer: wgpu.GPUBuffer
    chunk_offsets_buffer: wgpu.GPUBuffer
    params_buffer: wgpu.GPUBuffer
    readback_buffer: wgpu.GPUBuffer
    resources: AsyncVoxelMeshBatchResources | None = None
    metadata_promise: object | None = None
    submitted_at: float = 0.0


class TerrainRenderer:
    def __init__(
        self,
        seed: int = 1337,
        use_gpu_terrain: bool | None = None,
        use_gpu_meshing: bool | None = None,
        terrain_batch_size: int = 32,
        mesh_batch_size: int | None = None,
    ) -> None:
        default_use_gpu = ENGINE_MODE.lower() == ENGINE_MODE_GPU
        self.use_gpu_terrain = default_use_gpu if use_gpu_terrain is None else bool(use_gpu_terrain)
        self.terrain_batch_size = max(1, int(terrain_batch_size))
        if mesh_batch_size is None:
            self.mesh_batch_size = max(1, min(int(self.terrain_batch_size), DEFAULT_MESH_BATCH_SIZE))
        else:
            self.mesh_batch_size = max(1, int(mesh_batch_size))
        if CHUNK_PREP_REQUEST_BUDGET_CAP > CHUNK_PREP_REQUEST_BATCH_SIZE:
            print(
                "Warning: chunk request budget exceeds chunk request batch size; "
                "near chunks may wait behind farther chunks inside a batch.",
                file=sys.stderr,
            )
        self.base_title = "Minechunk"
        self.engine_mode_label = ENGINE_MODE.upper()
        self.canvas = RenderCanvas(
            title=self.base_title,
            size=(1280, 800),
            update_mode="continuous",
            max_fps=SWAPCHAIN_MAX_FPS,
            vsync=SWAPCHAIN_USE_VSYNC,
        )
        request_adapter = getattr(wgpu.gpu, "request_adapter_sync", wgpu.gpu.request_adapter)
        self.adapter = request_adapter(canvas=self.canvas, power_preference="high-performance")
        if self.adapter is None:
            raise RuntimeError("No compatible GPU adapter was found.")
        request_device = getattr(self.adapter, "request_device_sync", self.adapter.request_device)
        supported_features = set(getattr(self.adapter, "features", []) or [])
        requested_features = []
        if {"timestamp-query", "timestamp-query-inside-passes"}.issubset(supported_features):
            requested_features = ["timestamp-query", "timestamp-query-inside-passes"]
        self.timestamp_query_supported = bool(requested_features)
        if requested_features:
            self.device = request_device(required_features=requested_features)
        else:
            self.device = request_device()
        self.context = self.canvas.get_wgpu_context()
        self.color_format = self.context.get_preferred_format(self.adapter)
        self.render_api_label = self._describe_render_api()
        self.context.configure(
            device=self.device,
            format=self.color_format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            alpha_mode="opaque",
        )

        self.world = VoxelWorld(
            seed,
            gpu_device=self.device,
            prefer_gpu_terrain=self.use_gpu_terrain,
            terrain_batch_size=self.terrain_batch_size,
        )

        self.camera = Camera(position=[0.0, 200.0, 0.0], yaw=math.pi, pitch=-1.20)
        self.keys_down: set[str] = set()
        self.dragging = False
        self.last_pointer: tuple[float, float] | None = None
        self.last_frame_time = time.perf_counter()
        self.depth_texture = None
        self.depth_view = None
        self.depth_size = (0, 0)
        # 512 blocks rounds up to a whole number of chunks.
        self.chunk_radius = max(1, math.ceil(DEFAULT_RENDER_DISTANCE_BLOCKS / CHUNK_SIZE))
        self.render_dimension_chunks = self.chunk_radius * 2 + 1
        self.chunk_cache: OrderedDict[tuple[int, int], ChunkMesh] = OrderedDict()
        self._visible_chunk_coords: list[tuple[int, int]] = []
        self._visible_chunk_coord_set: set[tuple[int, int]] = set()
        self._visible_chunk_origin: tuple[int, int] | None = None
        self._visible_displayed_coords: set[tuple[int, int]] = set()
        self._visible_missing_coords: set[tuple[int, int]] = set()
        self._chunk_request_queue: deque[tuple[int, int]] = deque()
        self._chunk_request_queue_origin: tuple[int, int] | None = None
        self._chunk_request_queue_yaw = 0.0
        self._chunk_request_queue_dirty = True
        self._pending_chunk_coords: set[tuple[int, int]] = set()
        self._transient_render_buffers: list[list[wgpu.GPUBuffer]] = []
        self._tile_render_batches: dict[tuple[int, int], ChunkRenderBatch] = {}
        self._mesh_buffer_refs: dict[int, int] = {}
        self._mesh_output_slabs: OrderedDict[int, MeshOutputSlab] = OrderedDict()
        self._mesh_allocations: dict[int, MeshBufferAllocation] = {}
        self._next_mesh_output_slab_id = 1
        self._next_mesh_allocation_id = 1
        self._mesh_output_binding_alignment = max(
            VERTEX_STRIDE,
            int(self._device_limit("min_storage_buffer_offset_alignment", 256)),
        )
        self._mesh_output_min_slab_bytes = max(
            MESH_OUTPUT_SLAB_MIN_BYTES,
            self._mesh_output_binding_alignment,
        )
        self.use_gpu_indirect_render = True
        self._mesh_draw_indirect_capacity = 0
        self._mesh_draw_indirect_buffer = None
        self._mesh_draw_indirect_array = np.empty((0, 4), dtype=np.uint32)
        self.use_gpu_built_visibility = True
        self._mesh_visibility_record_capacity = 0
        self._mesh_visibility_record_buffer = None
        self._mesh_visibility_record_array = np.empty(0, dtype=MESH_VISIBILITY_RECORD_DTYPE)
        self._mesh_visibility_params_buffer = None
        self.max_cached_chunks = MAX_CACHED_CHUNKS
        self._cache_capacity_warned = False
        self._current_move_speed = self.camera.move_speed
        self._chunk_prep_tokens = 0.0
        self.use_gpu_meshing = default_use_gpu if use_gpu_meshing is None else bool(use_gpu_meshing)
        self.mesh_backend_label = "GPU" if self.use_gpu_meshing else "CPU"
        self.voxel_mesh_scan_validate_every = 0
        self._voxel_mesh_scan_batches_processed = 0
        self._pending_voxel_mesh_results: deque = deque()
        self._pending_gpu_mesh_batches: deque[PendingChunkMeshBatch] = deque()
        self._gpu_mesh_deferred_buffer_cleanup: deque[tuple[int, list[wgpu.GPUBuffer]]] = deque()
        self._gpu_mesh_deferred_batch_resource_releases: deque[tuple[int, AsyncVoxelMeshBatchResources]] = deque()
        self._async_voxel_mesh_batch_pool: deque[AsyncVoxelMeshBatchResources] = deque()
        self._async_voxel_mesh_batch_pool_limit = 8
        self._gpu_mesh_async_finalize_budget = max(1, self.mesh_batch_size)
        self.profiling_enabled = False
        self.profiler: cProfile.Profile | None = None
        self.profile_window_start = 0.0
        self.profile_next_report = 0.0
        self.profile_window_cpu_ms = 0.0
        self.profile_window_frames = 0
        self.profile_window_frame_times: list[float] = []
        self.profile_hud_lines: list[str] = []
        self.profile_hud_vertex_bytes = b""
        self.profile_hud_vertex_count = 0
        self._hud_geometry_cache: OrderedDict[tuple[bool, int, int, tuple[str, ...]], tuple[bytes, int]] = OrderedDict()
        self.frame_breakdown_samples: dict[str, deque[float]] = {
            "world_update": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "visibility_lookup": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "chunk_stream": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "chunk_stream_bytes": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "chunk_displayed_added": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "camera_upload": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "swapchain_acquire": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "render_encode": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "command_finish": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "queue_submit": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "wall_frame": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "draw_calls": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "merged_chunks": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "visible_vertices": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "visible_chunk_targets": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "visible_chunks": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "pending_chunk_requests": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
            "voxel_mesh_backlog": deque(maxlen=FRAME_BREAKDOWN_SAMPLE_WINDOW),
        }
        self.frame_breakdown_lines: list[str] = []
        self.frame_breakdown_vertex_bytes = b""
        self.frame_breakdown_vertex_count = 0
        self._last_frame_draw_calls = 0
        self._last_frame_merged_batches = 0
        self._last_new_displayed_chunks = 0
        self._last_chunk_stream_drained = 0
        self._last_frame_visible_batches = 0
        self._last_displayed_chunk_coords: set[tuple[int, int]] = set()
        self._voxel_mesh_scratch_capacity = 0
        self._voxel_mesh_scratch_sample_size = 0
        self._voxel_mesh_scratch_height_limit = 0
        self._voxel_mesh_scratch_blocks_buffer = None
        self._voxel_mesh_scratch_materials_buffer = None
        self._voxel_mesh_scratch_coords_buffer = None
        self._voxel_mesh_scratch_column_totals_buffer = None
        self._voxel_mesh_scratch_chunk_totals_buffer = None
        self._voxel_mesh_scratch_chunk_offsets_buffer = None
        self._voxel_mesh_scratch_params_buffer = None
        self._voxel_mesh_scratch_batch_vertex_buffer = None
        self._voxel_mesh_scratch_blocks_array = None
        self._voxel_mesh_scratch_materials_array = None
        self._voxel_mesh_scratch_coords_array = None
        self._voxel_mesh_scratch_chunk_totals_array = None
        self._voxel_mesh_scratch_chunk_offsets_array = None
        self._voxel_mesh_scratch_chunk_metadata_readback_buffer = None

        self.camera_buffer = self.device.create_buffer(
            size=80,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._mesh_visibility_params_buffer = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._tile_merge_dummy_buffer = self.device.create_buffer(
            size=VERTEX_STRIDE,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        self.voxel_mesh_count_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 6,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 8,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self.voxel_mesh_scan_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 6,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 8,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self.voxel_mesh_emit_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 6,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 8,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self.voxel_surface_expand_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self.mesh_visibility_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
            ]
        )
        self.tile_merge_bind_group_layout = None
        self.tile_merge_pipeline = None
        if self._device_limit("max_storage_buffers_per_shader_stage", 0) >= MERGED_TILE_MAX_CHUNKS + 2:
            self.tile_merge_bind_group_layout = self.device.create_bind_group_layout(
                entries=[
                    {
                        "binding": index,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": "read-only-storage"},
                    }
                    for index in range(MERGED_TILE_MAX_CHUNKS)
                ]
                + [
                    {
                        "binding": MERGED_TILE_MAX_CHUNKS,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": "read-only-storage"},
                    },
                    {
                        "binding": MERGED_TILE_MAX_CHUNKS + 1,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": "uniform"},
                    },
                    {
                        "binding": MERGED_TILE_MAX_CHUNKS + 2,
                        "visibility": wgpu.ShaderStage.COMPUTE,
                        "buffer": {"type": "storage"},
                    },
                ]
            )
        self.render_bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                }
            ]
        )
        self.camera_bind_group = self.device.create_bind_group(
            layout=self.render_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": self.camera_buffer, "offset": 0, "size": 80},
                }
            ],
        )

        if self.tile_merge_bind_group_layout is not None:
            try:
                self.tile_merge_pipeline = self.device.create_compute_pipeline(
                    layout=self.device.create_pipeline_layout(bind_group_layouts=[self.tile_merge_bind_group_layout]),
                    compute={"module": self.device.create_shader_module(code=TILE_MERGE_SHADER), "entry_point": "combine_main"},
                )
            except Exception as exc:
                self.tile_merge_pipeline = None
                print(f"Warning: GPU tile merge pipeline could not be created ({exc!s}); using copy-based tile merges.", file=sys.stderr)

        try:
            self.mesh_visibility_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.mesh_visibility_bind_group_layout]),
                compute={"module": self.device.create_shader_module(code=GPU_VISIBILITY_SHADER), "entry_point": "main"},
            )
            self.voxel_mesh_count_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.voxel_mesh_count_bind_group_layout]),
                compute={"module": self.device.create_shader_module(code=VOXEL_MESH_BATCH_SHADER), "entry_point": "count_main"},
            )
            self.voxel_mesh_scan_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.voxel_mesh_scan_bind_group_layout]),
                compute={"module": self.device.create_shader_module(code=VOXEL_MESH_BATCH_SHADER), "entry_point": "scan_main"},
            )
            self.voxel_mesh_emit_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.voxel_mesh_emit_bind_group_layout]),
                compute={"module": self.device.create_shader_module(code=VOXEL_MESH_BATCH_SHADER), "entry_point": "emit_main"},
            )
            self.voxel_surface_expand_pipeline = self.device.create_compute_pipeline(
                layout=self.device.create_pipeline_layout(bind_group_layouts=[self.voxel_surface_expand_bind_group_layout]),
                compute={"module": self.device.create_shader_module(code=VOXEL_SURFACE_EXPAND_SHADER), "entry_point": "expand_main"},
            )
        except Exception as exc:
            self.mesh_visibility_pipeline = None
            self.tile_merge_pipeline = None
            self.voxel_mesh_count_pipeline = None
            self.voxel_mesh_scan_pipeline = None
            self.voxel_mesh_emit_pipeline = None
            self.voxel_surface_expand_pipeline = None
            self.use_gpu_meshing = False
            self.mesh_backend_label = "CPU"
            print(f"Warning: GPU meshing could not be created ({exc!s}); using CPU meshing.", file=sys.stderr)
        self.render_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[self.render_bind_group_layout]),
            vertex={
                "module": self.device.create_shader_module(code=RENDER_SHADER),
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": VERTEX_STRIDE,
                        "step_mode": "vertex",
                        "attributes": [
                            {"shader_location": 0, "offset": 0, "format": "float32x4"},
                            {"shader_location": 1, "offset": 16, "format": "float32x4"},
                            {"shader_location": 2, "offset": 32, "format": "float32x4"},
                        ],
                    }
                ],
            },
            fragment={
                "module": self.device.create_shader_module(code=RENDER_SHADER),
                "entry_point": "fs_main",
                "targets": [{"format": self.color_format}],
            },
            primitive={
                "topology": "triangle-list",
                "cull_mode": "none",
            },
            depth_stencil={
                "format": DEPTH_FORMAT,
                "depth_write_enabled": True,
                "depth_compare": "less",
            },
        )
        self.profile_hud_pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[]),
            vertex={
                "module": self.device.create_shader_module(code=HUD_SHADER),
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": VERTEX_STRIDE,
                        "step_mode": "vertex",
                        "attributes": [
                            {"shader_location": 0, "offset": 0, "format": "float32x4"},
                            {"shader_location": 1, "offset": 16, "format": "float32x4"},
                            {"shader_location": 2, "offset": 32, "format": "float32x4"},
                        ],
                    }
                ],
            },
            fragment={
                "module": self.device.create_shader_module(code=HUD_SHADER),
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": self.color_format,
                        "blend": {
                            "color": {
                                "operation": "add",
                                "src_factor": "src-alpha",
                                "dst_factor": "one-minus-src-alpha",
                            },
                            "alpha": {
                                "operation": "add",
                                "src_factor": "one",
                                "dst_factor": "one-minus-src-alpha",
                            },
                        },
                    }
                ],
            },
            primitive={
                "topology": "triangle-list",
                "cull_mode": "none",
            },
        )

        self.canvas.add_event_handler(self._handle_key_down, "key_down")
        self.canvas.add_event_handler(self._handle_key_up, "key_up")
        self.canvas.add_event_handler(self._handle_pointer_down, "pointer_down")
        self.canvas.add_event_handler(self._handle_pointer_move, "pointer_move")
        self.canvas.add_event_handler(self._handle_pointer_up, "pointer_up")
        self.canvas.add_event_handler(self._handle_resize, "resize")

        self._disable_profiling()
        self.canvas.request_draw(self.draw_frame)

    def run(self) -> None:
        loop.run()

    def _handle_resize(self, event) -> None:
        self.depth_size = (0, 0)
        self.depth_texture = None
        self.depth_view = None
        if self.profiling_enabled and self.profile_hud_lines:
            self.profile_hud_vertex_bytes, self.profile_hud_vertex_count = self._build_profile_hud_vertices(self.profile_hud_lines)
        if self.profiling_enabled and self.frame_breakdown_lines:
            self.frame_breakdown_vertex_bytes, self.frame_breakdown_vertex_count = self._build_frame_breakdown_hud_vertices(self.frame_breakdown_lines)

    def _normalize_key(self, event) -> str:
        key = str(event.get("key", "")).strip().lower()
        if key in {" ", "spacebar"}:
            return "space"
        if key in {"controlleft", "controlright", "ctrl"}:
            return "control"
        if key == "shiftleft":
            return "shiftleft"
        if key == "shiftright":
            return "shiftright"
        return key

    def _handle_key_down(self, event) -> None:
        key = self._normalize_key(event)
        is_new_press = key not in self.keys_down
        self.keys_down.add(key)
        if is_new_press and key == "f3":
            self._toggle_profiling()
        if is_new_press and key in {"r"}:
            self.regenerate_world()

    def _handle_key_up(self, event) -> None:
        self.keys_down.discard(self._normalize_key(event))

    def _handle_pointer_down(self, event) -> None:
        if int(event.get("button", 0)) == 1:
            self.dragging = True
            self.last_pointer = (float(event.get("x", 0.0)), float(event.get("y", 0.0)))

    def _handle_pointer_move(self, event) -> None:
        if not self.dragging:
            return
        x = float(event.get("x", 0.0))
        y = float(event.get("y", 0.0))
        if self.last_pointer is None:
            self.last_pointer = (x, y)
            return
        last_x, last_y = self.last_pointer
        dx = x - last_x
        dy = y - last_y
        self.last_pointer = (x, y)
        self.camera.yaw -= dx * self.camera.look_speed
        self.camera.pitch -= dy * self.camera.look_speed
        self.camera.clamp_pitch()

    def _handle_pointer_up(self, event) -> None:
        if int(event.get("button", 0)) == 1:
            self.dragging = False
            self.last_pointer = None

    def _key_active(self, *names: str) -> bool:
        for name in names:
            if name in self.keys_down:
                return True
        return False

    def _toggle_profiling(self) -> None:
        if self.profiling_enabled:
            self._disable_profiling()
        else:
            self._enable_profiling()

    def _enable_profiling(self) -> None:
        self.profiling_enabled = True
        self.profiler = cProfile.Profile()
        now = time.perf_counter()
        self.profile_window_start = now
        self.profile_next_report = now + PROFILE_REPORT_INTERVAL
        self.profile_window_cpu_ms = 0.0
        self.profile_window_frames = 0
        self.profile_window_frame_times = []
        for samples in self.frame_breakdown_samples.values():
            samples.clear()
        self.profile_hud_lines = [
            "PROFILE ON  AVG FPS --.-  CPU --.-MS",
            "FRAME P50 --.-MS  P95 --.-MS  P99 --.-MS",
            f"RENDER API  {self.render_api_label}",
            f"ENGINE MODE {self.engine_mode_label}",
            f"TERRAIN    {self.world.terrain_backend_label()}",
            f"MESH       {self.mesh_backend_label}",
            f"CHUNK DIMS {CHUNK_SIZE}x{WORLD_HEIGHT}x{CHUNK_SIZE}",
            f"BATCH SIZE {self.terrain_batch_size}",
            f"MESH DRAIN {self.mesh_batch_size}",
            f"PRESENT    FPS {SWAPCHAIN_MAX_FPS}  VSYNC {'ON' if SWAPCHAIN_USE_VSYNC else 'OFF'}",
            "MESH SLABS --  USED --.- MIB  FREE --.- MIB",
            "MESH BIGGEST GAP --.- MIB  ALLOCS --",
            f"CHUNK REQUEST BATCH SIZE {CHUNK_PREP_REQUEST_BATCH_SIZE}",
            "COLLECTING SAMPLES",
        ]
        self.profile_hud_vertex_bytes, self.profile_hud_vertex_count = self._build_profile_hud_vertices(self.profile_hud_lines)
        self.frame_breakdown_lines = [
            f"FRAME BREAKDOWN @ DIMENSION {self.render_dimension_chunks}x{self.render_dimension_chunks} CHUNKS",
            f"ENGINE MODE: {self.engine_mode_label}",
            "CPU FRAME ISSUE: --.- MS",
            "  WORLD UPDATE: --.- MS",
            "  VISIBILITY LOOKUP: --.- MS",
            "  CHUNK STREAM: --.- MS",
            "  CHUNK STREAM BW: --.- MIB/S",
            "  CAMERA UPLOAD: --.- MS",
            "  SWAPCHAIN ACQUIRE: --.- MS",
            "  RENDER ENCODE: --.- MS",
            "  COMMAND FINISH: --.- MS",
            "  QUEUE SUBMIT: --.- MS",
            f"CHUNK DIMS: {CHUNK_SIZE}x{WORLD_HEIGHT}x{CHUNK_SIZE}",
            f"BACKEND POLL SIZE: {self.terrain_batch_size}",
            f"MESH DRAIN SIZE: {self.mesh_batch_size}",
            f"PRESENT PACING: FPS {SWAPCHAIN_MAX_FPS}  VSYNC {'ON' if SWAPCHAIN_USE_VSYNC else 'OFF'}",
            f"CHUNK REQUEST BATCH SIZE: {CHUNK_PREP_REQUEST_BATCH_SIZE}",
            "MESH SLABS: --  USED --.- MIB  FREE --.- MIB",
            "MESH BIGGEST GAP: --.- MIB  ALLOCS --",
            "TOTAL DRAW VERTICES: --",
            "WALL FRAME: --.- MS",
            "CHUNK MEMORY: -- BYTES (--.- MIB)",
            "DRAW CALLS: --",
            "VISIBLE MERGED CHUNKS (VISIBLE ONLY): --",
        ]
        self.frame_breakdown_vertex_bytes, self.frame_breakdown_vertex_count = self._build_frame_breakdown_hud_vertices(self.frame_breakdown_lines)

    def _disable_profiling(self) -> None:
        self.profiling_enabled = False
        self.profiler = None
        self.profile_window_start = 0.0
        self.profile_next_report = 0.0
        self.profile_window_cpu_ms = 0.0
        self.profile_window_frames = 0
        self.profile_window_frame_times = []
        self.profile_hud_lines = []
        self.profile_hud_vertex_bytes = b""
        self.profile_hud_vertex_count = 0
        self.frame_breakdown_lines = []
        self.frame_breakdown_vertex_bytes = b""
        self.frame_breakdown_vertex_count = 0
        self._hud_geometry_cache.clear()
        for samples in self.frame_breakdown_samples.values():
            samples.clear()

    def _profile_begin_frame(self) -> float | None:
        if not self.profiling_enabled or self.profiler is None:
            return None
        self.profiler.enable()
        return time.perf_counter()

    def _profile_end_frame(self, started_at: float | None, frame_dt: float) -> None:
        if started_at is None or self.profiler is None:
            return
        self.profiler.disable()
        ended_at = time.perf_counter()
        self.profile_window_cpu_ms += (ended_at - started_at) * 1000.0
        self.profile_window_frames += 1
        self.profile_window_frame_times.append(frame_dt)
        if ended_at >= self.profile_next_report:
            self._refresh_profile_summary(ended_at)

    def _refresh_profile_summary(self, now: float) -> None:
        if self.profiler is None:
            return
        avg_cpu_ms = self.profile_window_cpu_ms / max(1, self.profile_window_frames)
        avg_fps = self._profile_average_fps()
        frame_p50_ms, frame_p95_ms, frame_p99_ms = self._profile_frame_time_percentiles()
        stats = Stats(self.profiler)
        entries = sorted(stats.stats.items(), key=lambda item: item[1][2], reverse=True)

        slab_count, slab_total_bytes, slab_used_bytes, slab_free_bytes, slab_largest_free_bytes, slab_alloc_count = self._mesh_output_allocator_stats()
        lines = [
            f"PROFILE ON  AVG FPS {avg_fps:5.1f}  CPU {avg_cpu_ms:5.1f}MS",
            f"FRAME P50 {frame_p50_ms:5.1f}MS  P95 {frame_p95_ms:5.1f}MS  P99 {frame_p99_ms:5.1f}MS",
            f"RENDER API  {self.render_api_label}",
            f"ENGINE MODE {self.engine_mode_label}",
            f"PRESENT     FPS {SWAPCHAIN_MAX_FPS}  VSYNC {'ON' if SWAPCHAIN_USE_VSYNC else 'OFF'}",
            f"MESH SLABS {slab_count:2d}  USED {slab_used_bytes / (1024.0 * 1024.0):4.1f} MIB  FREE {slab_free_bytes / (1024.0 * 1024.0):4.1f} MIB",
            f"MESH BIGGEST GAP {slab_largest_free_bytes / (1024.0 * 1024.0):4.1f} MIB  ALLOCS {slab_alloc_count:3d}",
        ]
        hotspot_label = self._profile_hotspot_label(entries)
        if hotspot_label:
            lines.append(hotspot_label.upper())
        for index, (key, stat) in enumerate(entries[:30], start=1):
            funcname = key[2].upper().replace("<", "").replace(">", "")
            funcname = funcname[:24]
            lines.append(f"{index} {funcname:<24} {stat[2] * 1000.0:5.1f}MS")
        self.profile_hud_lines = lines
        self.profile_hud_vertex_bytes, self.profile_hud_vertex_count = self._build_profile_hud_vertices(lines)

        self.profiler = cProfile.Profile()
        self.profile_window_start = now
        self.profile_next_report = now + PROFILE_REPORT_INTERVAL
        self.profile_window_cpu_ms = 0.0
        self.profile_window_frames = 0
        self.profile_window_frame_times = []

    def _profile_frame_time_percentiles(self) -> tuple[float, float, float]:
        if not self.profile_window_frame_times:
            return 0.0, 0.0, 0.0
        ordered = sorted(self.profile_window_frame_times)
        count = len(ordered)

        def pick(percentile: float) -> float:
            index = max(0, min(count - 1, math.ceil(percentile * count) - 1))
            return ordered[index] * 1000.0

        return pick(0.50), pick(0.95), pick(0.99)

    def _profile_average_fps(self) -> float:
        if not self.profile_window_frame_times:
            return 0.0
        avg_frame_time = sum(self.profile_window_frame_times) / len(self.profile_window_frame_times)
        return 1.0 / max(1e-6, avg_frame_time)

    def _record_frame_breakdown_sample(self, name: str, value: float) -> None:
        samples = self.frame_breakdown_samples.get(name)
        if samples is not None:
            samples.append(value)

    def _frame_breakdown_average(self, name: str) -> float:
        samples = self.frame_breakdown_samples.get(name)
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def _chunk_cache_memory_bytes(self) -> int:
        slab_sizes_by_buffer_id = {
            id(slab.buffer): slab.size_bytes
            for slab in self._mesh_output_slabs.values()
        }
        buffer_bytes: dict[int, int] = {}
        for mesh in self.chunk_cache.values():
            key = id(mesh.vertex_buffer)
            if key in slab_sizes_by_buffer_id:
                buffer_bytes[key] = slab_sizes_by_buffer_id[key]
            else:
                buffer_bytes[key] = max(
                    buffer_bytes.get(key, 0),
                    mesh.vertex_offset + mesh.vertex_count * VERTEX_STRIDE,
                )
        return sum(buffer_bytes.values())

    def _mesh_output_allocator_stats(self) -> tuple[int, int, int, int, int, int]:
        slab_count = len(self._mesh_output_slabs)
        total_bytes = 0
        free_bytes = 0
        largest_free_bytes = 0
        for slab in self._mesh_output_slabs.values():
            total_bytes += int(slab.size_bytes)
            for _, size in slab.free_ranges:
                size = int(size)
                free_bytes += size
                if size > largest_free_bytes:
                    largest_free_bytes = size
        used_bytes = max(0, total_bytes - free_bytes)
        allocation_count = len(self._mesh_allocations)
        return slab_count, total_bytes, used_bytes, free_bytes, largest_free_bytes, allocation_count

    def _refresh_frame_breakdown_summary(self) -> None:
        if not self.profiling_enabled:
            return
        avg_world_update = self._frame_breakdown_average("world_update")
        avg_visibility_lookup = self._frame_breakdown_average("visibility_lookup")
        avg_chunk_stream = self._frame_breakdown_average("chunk_stream")
        avg_chunk_stream_bytes = self._frame_breakdown_average("chunk_stream_bytes")
        avg_new_displayed_chunks = self._frame_breakdown_average("chunk_displayed_added")
        avg_camera_upload = self._frame_breakdown_average("camera_upload")
        avg_swapchain_acquire = self._frame_breakdown_average("swapchain_acquire")
        avg_render_encode = self._frame_breakdown_average("render_encode")
        avg_command_finish = self._frame_breakdown_average("command_finish")
        avg_queue_submit = self._frame_breakdown_average("queue_submit")
        avg_wall_frame = self._frame_breakdown_average("wall_frame")
        pending_chunk_requests = int(round(self._frame_breakdown_average("pending_chunk_requests")))
        visible_vertices = int(round(self._frame_breakdown_average("visible_vertices")))
        avg_issue_encode = (
            avg_world_update
            + avg_visibility_lookup
            + avg_chunk_stream
            + avg_camera_upload
            + avg_swapchain_acquire
            + avg_render_encode
            + avg_command_finish
            + avg_queue_submit
        )
        draw_calls = int(round(self._frame_breakdown_average("draw_calls")))
        merged_chunks = int(round(self._frame_breakdown_average("merged_chunks")))
        visible_chunk_targets = int(round(self._frame_breakdown_average("visible_chunk_targets")))
        visible_chunks = int(round(self._frame_breakdown_average("visible_chunks")))
        visible_but_not_ready = max(0, visible_chunk_targets - visible_chunks)
        chunk_memory_bytes = self._chunk_cache_memory_bytes()
        chunk_memory_mib = chunk_memory_bytes / (1024.0 * 1024.0)
        slab_count, slab_total_bytes, slab_used_bytes, slab_free_bytes, slab_largest_free_bytes, slab_alloc_count = self._mesh_output_allocator_stats()
        chunk_stream_bandwidth_mib_s = 0.0
        if avg_chunk_stream > 0.0:
            chunk_stream_bandwidth_mib_s = (avg_chunk_stream_bytes / (1024.0 * 1024.0)) / max(avg_chunk_stream / 1000.0, 1e-9)
        chunk_generation_per_s = 0.0
        if avg_wall_frame > 0.0:
            chunk_generation_per_s = avg_new_displayed_chunks / max(avg_wall_frame / 1000.0, 1e-9)

        lines = [
            f"FRAME BREAKDOWN @ DIMENSION {self.render_dimension_chunks}x{self.render_dimension_chunks} CHUNKS",
            f"ENGINE MODE: {self.engine_mode_label}",
            f"MOVE SPEED: {self._current_move_speed:5.1f} B/S",
            f"TERRAIN BACKEND: {self.world.terrain_backend_label()}",
            f"MESH BACKEND: {self.mesh_backend_label}",
            f"CHUNK DIMS: {CHUNK_SIZE}x{WORLD_HEIGHT}x{CHUNK_SIZE}",
            f"BACKEND POLL SIZE: {self.terrain_batch_size}",
            f"MESH DRAIN SIZE: {self.mesh_batch_size}",
            f"CHUNK REQUEST BATCH SIZE: {CHUNK_PREP_REQUEST_BATCH_SIZE}",
            f"MESH SLABS: {slab_count}  USED {slab_used_bytes / (1024.0 * 1024.0):4.1f} MIB  FREE {slab_free_bytes / (1024.0 * 1024.0):4.1f} MIB",
            f"MESH BIGGEST GAP: {slab_largest_free_bytes / (1024.0 * 1024.0):4.1f} MIB  ALLOCS {slab_alloc_count}",
            f"CPU FRAME ISSUE: {avg_issue_encode:5.1f} MS",
            f"  WORLD UPDATE: {avg_world_update:5.1f} MS",
            f"  VISIBILITY LOOKUP: {avg_visibility_lookup:5.1f} MS",
            f"  CHUNK STREAM: {avg_chunk_stream:5.1f} MS",
            f"  CHUNK STREAM BANDWIDTH: {chunk_stream_bandwidth_mib_s:5.1f} MIB/S",
            f"  NEW GENERATED CHUNKS / S: {chunk_generation_per_s:5.1f}",
            f"  CAMERA UPLOAD: {avg_camera_upload:5.1f} MS",
            f"  SWAPCHAIN ACQUIRE: {avg_swapchain_acquire:5.1f} MS",
            f"  RENDER ENCODE: {avg_render_encode:5.1f} MS",
            f"  COMMAND FINISH: {avg_command_finish:5.1f} MS",
            f"  QUEUE SUBMIT: {avg_queue_submit:5.1f} MS",
            f"WALL FRAME: {avg_wall_frame:5.1f} MS",
            f"CHUNK MEMORY: {chunk_memory_bytes:,} BYTES ({chunk_memory_mib:5.2f} MIB)",
            f"TOTAL DRAW VERTICES: {visible_vertices:,}",
            f"VISIBLE BUT NOT READY: {visible_but_not_ready}",
            f"PENDING CHUNK REQUESTS: {pending_chunk_requests}",
            f"DRAW CALLS: {draw_calls}",
            f"VISIBLE MERGED CHUNKS (VISIBLE ONLY): {merged_chunks}",
        ]
        self.frame_breakdown_lines = lines
        self.frame_breakdown_vertex_bytes, self.frame_breakdown_vertex_count = self._build_frame_breakdown_hud_vertices(lines)

    def _profile_hotspot_label(self, entries) -> str:
        if not entries:
            return ""

        hot_funcs = [entry[0][2] for entry in entries[:6]]
        if any(name in {"_submit_render", "get_current_texture", "_get_current_texture"} for name in hot_funcs):
            return "render submit bottleneck"
        if any(
            name in {
                "surface_profile_at",
                "chunk_surface_grids",
                "chunk_voxel_grid",
                "fill_chunk_surface_grids",
                "fill_chunk_voxel_grid",
                "count_chunk_voxel_vertices",
                "build_chunk_vertex_array",
                "build_chunk_vertex_array_from_voxels",
                "_build_chunk_vertex_bytes",
                "_gpu_make_chunk_mesh_from_voxels",
            }
            or "build_chunk_vertex" in name
            or "vertex_buffer" in name
            for name in hot_funcs
        ):
            return "chunk vertex bottleneck"
        if any(name in {"_prepare_chunks", "_ensure_chunk_mesh", "_make_chunk_mesh"} for name in hot_funcs):
            return "chunk meshing bottleneck"
        return ""

    def _describe_render_api(self) -> str:
        info = getattr(self.adapter, "info", None)
        summary = getattr(self.adapter, "summary", "")

        backend = ""
        adapter_type = ""
        description = ""

        if info is not None:
            getter = info.get if hasattr(info, "get") else None
            if getter is not None:
                backend = getter("backend_type", "") or getter("backend", "")
                adapter_type = getter("adapter_type", "") or getter("device_type", "")
                description = getter("description", "") or getter("device", "")
            else:
                backend = getattr(info, "backend_type", "") or getattr(info, "backend", "")
                adapter_type = getattr(info, "adapter_type", "") or getattr(info, "device_type", "")
                description = getattr(info, "description", "") or getattr(info, "device", "")

        if isinstance(summary, str) and summary.strip():
            if not backend and not adapter_type and not description:
                return summary.strip()

        parts = [part for part in (backend, adapter_type, description) if part]
        if parts:
            return " / ".join(parts)
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        return "unknown"

    def _device_limit(self, name: str, default: int) -> int:
        limits = getattr(self.device, "limits", None)
        if limits is None:
            return int(default)
        getter = limits.get if hasattr(limits, "get") else None
        if getter is not None:
            value = getter(name, default)
        else:
            value = getattr(limits, name, default)
        try:
            return int(value)
        except Exception:
            return int(default)

    def _align_up(self, value: int, alignment: int) -> int:
        if alignment <= 1:
            return int(value)
        return ((int(value) + alignment - 1) // alignment) * alignment

    def _mesh_output_slab_size_for_request(self, request_bytes: int) -> int:
        size_bytes = max(self._mesh_output_min_slab_bytes, int(request_bytes))
        slab_bytes = self._mesh_output_min_slab_bytes
        while slab_bytes < size_bytes:
            slab_bytes *= 2
        return slab_bytes

    def _create_mesh_output_slab(self, size_bytes: int) -> MeshOutputSlab:
        slab = MeshOutputSlab(
            slab_id=self._next_mesh_output_slab_id,
            buffer=self.device.create_buffer(
                size=max(1, int(size_bytes)),
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            ),
            size_bytes=int(size_bytes),
            free_ranges=[(0, int(size_bytes))],
        )
        self._next_mesh_output_slab_id += 1
        self._mesh_output_slabs[slab.slab_id] = slab
        return slab

    def _allocate_from_mesh_output_slab(
        self,
        slab: MeshOutputSlab,
        request_bytes: int,
    ) -> MeshBufferAllocation | None:
        needed_bytes = max(1, int(request_bytes))
        alignment = self._mesh_output_binding_alignment
        for index, (range_offset, range_size) in enumerate(slab.free_ranges):
            aligned_offset = self._align_up(range_offset, alignment)
            padding = aligned_offset - range_offset
            usable_size = range_size - padding
            if usable_size < needed_bytes:
                continue

            alloc_end = aligned_offset + needed_bytes
            new_ranges: list[tuple[int, int]] = []
            if padding > 0:
                new_ranges.append((range_offset, padding))
            tail_size = (range_offset + range_size) - alloc_end
            if tail_size > 0:
                new_ranges.append((alloc_end, tail_size))
            slab.free_ranges[index:index + 1] = new_ranges

            allocation = MeshBufferAllocation(
                allocation_id=self._next_mesh_allocation_id,
                buffer=slab.buffer,
                offset_bytes=aligned_offset,
                size_bytes=needed_bytes,
                slab_id=slab.slab_id,
                refcount=0,
            )
            self._next_mesh_allocation_id += 1
            self._mesh_allocations[allocation.allocation_id] = allocation
            return allocation
        return None

    def _allocate_mesh_output_range(self, request_bytes: int) -> MeshBufferAllocation:
        needed_bytes = max(1, int(request_bytes))
        needed_bytes = self._align_up(needed_bytes, self._mesh_output_binding_alignment)
        for slab in self._mesh_output_slabs.values():
            allocation = self._allocate_from_mesh_output_slab(slab, needed_bytes)
            if allocation is not None:
                return allocation

        slab = self._create_mesh_output_slab(self._mesh_output_slab_size_for_request(needed_bytes))
        allocation = self._allocate_from_mesh_output_slab(slab, needed_bytes)
        if allocation is None:
            raise RuntimeError("Failed to suballocate mesh output slab.")
        return allocation

    def _coalesce_mesh_output_free_ranges(self, free_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not free_ranges:
            return []
        merged: list[list[int]] = []
        for offset, size in sorted(free_ranges):
            if size <= 0:
                continue
            if not merged:
                merged.append([offset, size])
                continue
            last = merged[-1]
            last_end = last[0] + last[1]
            if offset <= last_end:
                last[1] = max(last_end, offset + size) - last[0]
            else:
                merged.append([offset, size])
        return [(offset, size) for offset, size in merged]

    def _free_mesh_output_range(self, slab_id: int, offset_bytes: int, size_bytes: int) -> None:
        slab = self._mesh_output_slabs.get(int(slab_id))
        if slab is None:
            return
        slab.free_ranges.append((int(offset_bytes), int(size_bytes)))
        slab.free_ranges = self._coalesce_mesh_output_free_ranges(slab.free_ranges)

    def _retain_chunk_mesh_storage(self, mesh: ChunkMesh) -> None:
        if mesh.allocation_id is None:
            self._retain_mesh_buffer(mesh.vertex_buffer)
            return
        allocation = self._mesh_allocations.get(mesh.allocation_id)
        if allocation is None:
            raise RuntimeError(f"Unknown mesh allocation id: {mesh.allocation_id}")
        allocation.refcount += 1

    def _release_chunk_mesh_storage(self, mesh: ChunkMesh) -> None:
        if mesh.allocation_id is None:
            self._release_mesh_buffer(mesh.vertex_buffer)
            return
        allocation = self._mesh_allocations.get(mesh.allocation_id)
        if allocation is None:
            return
        allocation.refcount -= 1
        if allocation.refcount > 0:
            return
        self._mesh_allocations.pop(mesh.allocation_id, None)
        if allocation.slab_id is None:
            allocation.buffer.destroy()
            return
        self._free_mesh_output_range(allocation.slab_id, allocation.offset_bytes, allocation.size_bytes)

    def _build_hud_vertices(
        self,
        lines: list[str],
        *,
        align_right: bool = False,
    ) -> tuple[bytes, int]:
        if not lines:
            return b"", 0

        screen_w, screen_h = self.canvas.get_physical_size()
        if screen_w <= 0 or screen_h <= 0:
            return b"", 0

        cache_key = (bool(align_right), int(screen_w), int(screen_h), tuple(lines))
        cached = self._hud_geometry_cache.get(cache_key)
        if cached is not None:
            self._hud_geometry_cache.move_to_end(cache_key)
            return cached

        scale = HUD_FONT_SCALE
        glyph_w = HUD_FONT_CHAR_WIDTH * scale
        glyph_h = HUD_FONT_CHAR_HEIGHT * scale
        advance_x = glyph_w + HUD_GLYPH_SPACING * scale
        line_step = glyph_h + HUD_LINE_SPACING
        max_chars = max(len(line) for line in lines)
        panel_w = HUD_PANEL_PADDING * 2 + max_chars * advance_x - (HUD_GLYPH_SPACING * scale if max_chars else 0)
        panel_h = HUD_PANEL_PADDING * 2 + len(lines) * line_step - HUD_LINE_SPACING

        vertices: list[bytes] = []

        def add_quad(px0: float, py0: float, px1: float, py1: float, color: tuple[float, float, float], alpha: float) -> None:
            x0, y0 = _screen_to_ndc(px0, py0, screen_w, screen_h)
            x1, y1 = _screen_to_ndc(px1, py1, screen_w, screen_h)
            vertices.append(pack_vertex((x0, y0, 0.0), (0.0, 0.0, 1.0), color, alpha))
            vertices.append(pack_vertex((x1, y0, 0.0), (0.0, 0.0, 1.0), color, alpha))
            vertices.append(pack_vertex((x1, y1, 0.0), (0.0, 0.0, 1.0), color, alpha))
            vertices.append(pack_vertex((x0, y0, 0.0), (0.0, 0.0, 1.0), color, alpha))
            vertices.append(pack_vertex((x1, y1, 0.0), (0.0, 0.0, 1.0), color, alpha))
            vertices.append(pack_vertex((x0, y1, 0.0), (0.0, 0.0, 1.0), color, alpha))

        panel_x = 12.0
        panel_y = 12.0
        if align_right:
            panel_x = max(12.0, float(screen_w) - panel_w - 12.0)
        add_quad(panel_x, panel_y, panel_x + panel_w, panel_y + panel_h, (0.05, 0.07, 0.09), 0.78)
        add_quad(panel_x, panel_y, panel_x + panel_w, panel_y + 4.0, (0.12, 0.72, 0.85), 0.85)

        for line_index, raw_line in enumerate(lines):
            line = raw_line.upper()
            cursor_x = panel_x + HUD_PANEL_PADDING
            cursor_y = panel_y + HUD_PANEL_PADDING + line_index * line_step
            text_color = (0.92, 0.97, 1.0) if line_index == 0 else (0.84, 0.90, 0.84)
            for char in line:
                glyph = _hud_glyph_rows(char)
                for row_index, row in enumerate(glyph):
                    for col_index, bit in enumerate(row):
                        if bit != "1":
                            continue
                        px0 = cursor_x + col_index * scale
                        py0 = cursor_y + row_index * scale
                        add_quad(px0, py0, px0 + scale, py0 + scale, text_color, 1.0)
                cursor_x += advance_x

        built = b"".join(vertices), len(vertices)
        self._hud_geometry_cache[cache_key] = built
        while len(self._hud_geometry_cache) > 8:
            self._hud_geometry_cache.popitem(last=False)
        return built

    def _build_profile_hud_vertices(self, lines: list[str]) -> tuple[bytes, int]:
        return self._build_hud_vertices(lines)

    def _build_frame_breakdown_hud_vertices(self, lines: list[str]) -> tuple[bytes, int]:
        return self._build_hud_vertices(lines, align_right=True)

    def _draw_hud_overlay(self, encoder, color_view, vertex_bytes: bytes, vertex_count: int) -> None:
        if not vertex_bytes:
            return
        hud_buffer = self.device.create_buffer_with_data(
            data=vertex_bytes,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        hud_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": color_view,
                    "resolve_target": None,
                    "load_op": wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        hud_pass.set_pipeline(self.profile_hud_pipeline)
        hud_pass.set_vertex_buffer(0, hud_buffer)
        hud_pass.draw(vertex_count, 1, 0, 0)
        hud_pass.end()

    def _draw_profile_hud(self, encoder, color_view) -> None:
        if not self.profiling_enabled:
            return
        self._draw_hud_overlay(encoder, color_view, self.profile_hud_vertex_bytes, self.profile_hud_vertex_count)

    def _draw_frame_breakdown_hud(self, encoder, color_view) -> None:
        if not self.profiling_enabled:
            return
        self._draw_hud_overlay(
            encoder,
            color_view,
            self.frame_breakdown_vertex_bytes,
            self.frame_breakdown_vertex_count,
        )

    def regenerate_world(self) -> None:
        for mesh in self.chunk_cache.values():
            self._release_chunk_mesh_storage(mesh)
        self.chunk_cache.clear()
        self._mesh_buffer_refs.clear()
        self._pending_chunk_coords.clear()
        self._chunk_request_queue.clear()
        self._chunk_request_queue_origin = None
        self._chunk_request_queue_dirty = True
        self._pending_voxel_mesh_results.clear()
        while self._pending_gpu_mesh_batches:
            pending = self._pending_gpu_mesh_batches.popleft()
            if pending.readback_buffer.map_state != "unmapped":
                try:
                    pending.readback_buffer.unmap()
                except Exception:
                    pass
            if pending.resources is not None:
                self._destroy_async_voxel_mesh_batch_resources(pending.resources)
                continue
            for buffer in (
                pending.blocks_buffer,
                pending.materials_buffer,
                pending.coords_buffer,
                pending.column_totals_buffer,
                pending.chunk_totals_buffer,
                pending.chunk_offsets_buffer,
                pending.params_buffer,
                pending.readback_buffer,
            ):
                try:
                    buffer.destroy()
                except Exception:
                    pass
        while self._gpu_mesh_deferred_buffer_cleanup:
            _, buffers = self._gpu_mesh_deferred_buffer_cleanup.popleft()
            for buffer in buffers:
                try:
                    buffer.destroy()
                except Exception:
                    pass
        while self._gpu_mesh_deferred_batch_resource_releases:
            _, resources = self._gpu_mesh_deferred_batch_resource_releases.popleft()
            self._destroy_async_voxel_mesh_batch_resources(resources)
        while self._async_voxel_mesh_batch_pool:
            self._destroy_async_voxel_mesh_batch_resources(self._async_voxel_mesh_batch_pool.popleft())
        self._clear_tile_render_batches()
        self._clear_transient_render_buffers()
        self._visible_chunk_origin = None
        self.world = VoxelWorld(
            int(time.time()) & 0x7FFFFFFF,
            gpu_device=self.device,
            prefer_gpu_terrain=self.use_gpu_terrain,
            terrain_batch_size=self.terrain_batch_size,
        )
        self.camera.position[:] = [0.0, 200.0, 0.0]
        self.camera.yaw = math.pi
        self.camera.pitch = -1.20
        self.camera.clamp_pitch()

    def _clear_transient_render_buffers(self) -> None:
        for buffer_group in self._transient_render_buffers:
            for buffer in buffer_group:
                buffer.destroy()
        self._transient_render_buffers.clear()

    def _retain_mesh_buffer(self, buffer: wgpu.GPUBuffer) -> None:
        key = id(buffer)
        self._mesh_buffer_refs[key] = self._mesh_buffer_refs.get(key, 0) + 1

    def _release_mesh_buffer(self, buffer: wgpu.GPUBuffer) -> None:
        key = id(buffer)
        refs = self._mesh_buffer_refs.get(key, 0)
        if refs <= 1:
            self._mesh_buffer_refs.pop(key, None)
            buffer.destroy()
        else:
            self._mesh_buffer_refs[key] = refs - 1

    def _store_chunk_mesh(self, mesh: ChunkMesh) -> None:
        key = (mesh.chunk_x, mesh.chunk_z)
        self._pending_chunk_coords.discard(key)
        existing = self.chunk_cache.pop(key, None)
        if existing is not None:
            self._release_chunk_mesh_storage(existing)
        self.chunk_cache[key] = mesh
        self._retain_chunk_mesh_storage(mesh)
        if key in self._visible_chunk_coord_set:
            self._visible_displayed_coords.add(key)
            self._visible_missing_coords.discard(key)
        while len(self.chunk_cache) > self.max_cached_chunks:
            old_key, old_mesh = self.chunk_cache.popitem(last=False)
            self._release_chunk_mesh_storage(old_mesh)
            if old_key in self._visible_chunk_coord_set:
                self._visible_displayed_coords.discard(old_key)
                if old_key not in self._pending_chunk_coords:
                    self._visible_missing_coords.add(old_key)
                    self._chunk_request_queue_dirty = True

    def _chunk_mesh_age(self, mesh: ChunkMesh) -> float:
        return max(0.0, time.perf_counter() - mesh.created_at)

    def _clear_tile_render_batches(self) -> None:
        for batch in self._tile_render_batches.values():
            batch.vertex_buffer.destroy()
        self._tile_render_batches.clear()

    def _merge_chunk_bounds(self, tile_meshes: list[ChunkMesh]) -> tuple[float, float, float, float]:
        if not tile_meshes:
            raise ValueError("min() arg is an empty sequence")

        bounds = tile_meshes[0].bounds
        bx = bounds[0]
        by = bounds[1]
        bz = bounds[2]
        br = bounds[3]

        min_x = bx - br
        max_x = bx + br
        min_y = by - br
        max_y = by + br
        min_z = bz - br
        max_z = bz + br

        for index in range(1, len(tile_meshes)):
            bounds = tile_meshes[index].bounds
            bx = bounds[0]
            by = bounds[1]
            bz = bounds[2]
            br = bounds[3]

            left = bx - br
            right = bx + br
            bottom = by - br
            top = by + br
            near = bz - br
            far = bz + br

            if left < min_x:
                min_x = left
            if right > max_x:
                max_x = right
            if bottom < min_y:
                min_y = bottom
            if top > max_y:
                max_y = top
            if near < min_z:
                min_z = near
            if far > max_z:
                max_z = far

        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        center_z = (min_z + max_z) * 0.5

        dx = max_x - center_x
        dy = max_y - center_y
        dz = max_z - center_z
        radius = math.sqrt(dx * dx + dy * dy + dz * dz)

        return center_x, center_y, center_z, radius

    def _merge_tile_meshes(self, tile_meshes: list[ChunkMesh], encoder) -> wgpu.GPUBuffer:
        total_vertices = sum(mesh.vertex_count for mesh in tile_meshes)
        total_vertex_bytes = total_vertices * VERTEX_STRIDE
        merged_buffer = self.device.create_buffer(
            size=max(1, total_vertex_bytes),
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        if (
            self.tile_merge_pipeline is None
            or self.tile_merge_bind_group_layout is None
            or len(tile_meshes) > MERGED_TILE_MAX_CHUNKS
        ):
            dest_offset = 0
            for mesh in tile_meshes:
                copy_size = mesh.vertex_count * VERTEX_STRIDE
                encoder.copy_buffer_to_buffer(
                    mesh.vertex_buffer,
                    mesh.vertex_offset,
                    merged_buffer,
                    dest_offset,
                    copy_size,
                )
                dest_offset += copy_size
            return merged_buffer

        metadata_array = np.zeros((MERGED_TILE_MAX_CHUNKS, 4), dtype=np.uint32)
        dst_first_vertex = 0
        for index, mesh in enumerate(tile_meshes):
            metadata_array[index, 0] = np.uint32(mesh.vertex_count)
            metadata_array[index, 1] = np.uint32(dst_first_vertex)
            dst_first_vertex += mesh.vertex_count

        metadata_buffer = self.device.create_buffer(
            size=MERGED_TILE_MAX_CHUNKS * 16,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        params_buffer = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(metadata_buffer, 0, memoryview(metadata_array))
        self.device.queue.write_buffer(
            params_buffer,
            0,
            struct.pack("<4I", int(len(tile_meshes)), int(total_vertices), 0, 0),
        )

        entries = []
        for index in range(MERGED_TILE_MAX_CHUNKS):
            if index < len(tile_meshes):
                mesh = tile_meshes[index]
                entries.append(
                    {
                        "binding": index,
                        "resource": {
                            "buffer": mesh.vertex_buffer,
                            "offset": mesh.vertex_offset,
                            "size": max(1, mesh.vertex_count * VERTEX_STRIDE),
                        },
                    }
                )
            else:
                entries.append({"binding": index, "resource": {"buffer": self._tile_merge_dummy_buffer}})
        entries.extend(
            [
                {"binding": MERGED_TILE_MAX_CHUNKS, "resource": {"buffer": metadata_buffer, "offset": 0, "size": MERGED_TILE_MAX_CHUNKS * 16}},
                {"binding": MERGED_TILE_MAX_CHUNKS + 1, "resource": {"buffer": params_buffer, "offset": 0, "size": 16}},
                {"binding": MERGED_TILE_MAX_CHUNKS + 2, "resource": {"buffer": merged_buffer, "offset": 0, "size": max(1, total_vertex_bytes)}},
            ]
        )
        bind_group = self.device.create_bind_group(
            layout=self.tile_merge_bind_group_layout,
            entries=entries,
        )
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.tile_merge_pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(max(1, (total_vertices + 63) // 64), 1, 1)
        compute_pass.end()
        self._schedule_gpu_buffer_cleanup([metadata_buffer, params_buffer], frames=3)
        return merged_buffer

    def _build_tile_draw_batches(
        self,
        meshes: list[ChunkMesh],
        encoder,
        *,
        age_gate: bool,
    ) -> tuple[list[ChunkDrawBatch], int, int, int]:
        tile_groups: dict[tuple[int, int], list[ChunkMesh]] = {}
        for mesh in meshes:
            if mesh.vertex_count <= 0:
                continue
            tile_groups.setdefault(self._tile_key(mesh.chunk_x, mesh.chunk_z), []).append(mesh)

        current_tile_keys = set(tile_groups.keys())
        stale_keys = [tile_key for tile_key in self._tile_render_batches if tile_key not in current_tile_keys]
        for tile_key in stale_keys:
            batch = self._tile_render_batches.pop(tile_key)
            self._transient_render_buffers.append([batch.vertex_buffer])

        draw_batches: list[ChunkDrawBatch] = []
        merged_chunk_count = 0
        visible_chunk_count = 0

        for tile_key in sorted(tile_groups):
            tile_meshes = tile_groups[tile_key]
            mature_meshes = [mesh for mesh in tile_meshes if self._chunk_mesh_age(mesh) >= MERGED_TILE_MIN_AGE_SECONDS]
            immature_meshes = [mesh for mesh in tile_meshes if self._chunk_mesh_age(mesh) < MERGED_TILE_MIN_AGE_SECONDS]

            if len(tile_meshes) == 1:
                mesh = tile_meshes[0]
                draw_batches.append(
                    ChunkDrawBatch(
                        vertex_buffer=mesh.vertex_buffer,
                        binding_offset=mesh.binding_offset,
                        vertex_count=mesh.vertex_count,
                        first_vertex=mesh.first_vertex,
                        bounds=mesh.bounds,
                    )
                )
                visible_chunk_count += 1
                continue

            existing = self._tile_render_batches.get(tile_key)
            if len(mature_meshes) < 2:
                if existing is not None:
                    self._tile_render_batches.pop(tile_key, None)
                    self._transient_render_buffers.append([existing.vertex_buffer])
                for mesh in immature_meshes:
                    draw_batches.append(
                        ChunkDrawBatch(
                            vertex_buffer=mesh.vertex_buffer,
                            binding_offset=mesh.binding_offset,
                            vertex_count=mesh.vertex_count,
                            first_vertex=mesh.first_vertex,
                            bounds=mesh.bounds,
                        )
                    )
                    visible_chunk_count += 1
                for mesh in mature_meshes:
                    draw_batches.append(
                        ChunkDrawBatch(
                            vertex_buffer=mesh.vertex_buffer,
                            binding_offset=mesh.binding_offset,
                            vertex_count=mesh.vertex_count,
                            first_vertex=mesh.first_vertex,
                            bounds=mesh.bounds,
                        )
                    )
                    visible_chunk_count += 1
                continue

            signature = tuple((mesh.chunk_x, mesh.chunk_z) for mesh in mature_meshes)
            batch_vertex_count = sum(mesh.vertex_count for mesh in mature_meshes)
            if (
                existing is None
                or existing.signature != signature
                or existing.vertex_count != batch_vertex_count
            ):
                old_buffer = existing.vertex_buffer if existing is not None else None
                merged_buffer = self._merge_tile_meshes(mature_meshes, encoder)
                self._tile_render_batches[tile_key] = ChunkRenderBatch(
                    signature=signature,
                    vertex_count=batch_vertex_count,
                    vertex_buffer=merged_buffer,
                )
                if old_buffer is not None:
                    self._transient_render_buffers.append([old_buffer])

            batch = self._tile_render_batches[tile_key]
            draw_batches.append(
                ChunkDrawBatch(
                    vertex_buffer=batch.vertex_buffer,
                    binding_offset=0,
                    vertex_count=batch.vertex_count,
                    first_vertex=0,
                    bounds=self._merge_chunk_bounds(mature_meshes),
                    chunk_count=len(mature_meshes),
                )
            )
            merged_chunk_count += len(mature_meshes)
            visible_chunk_count += len(mature_meshes)
            for mesh in immature_meshes:
                draw_batches.append(
                    ChunkDrawBatch(
                        vertex_buffer=mesh.vertex_buffer,
                        binding_offset=mesh.binding_offset,
                        vertex_count=mesh.vertex_count,
                        first_vertex=mesh.first_vertex,
                        bounds=mesh.bounds,
                    )
                )
                visible_chunk_count += 1

        while len(self._transient_render_buffers) > 3:
            old_buffers = self._transient_render_buffers.pop(0)
            for buffer in old_buffers:
                buffer.destroy()

        visible_vertex_count = sum(batch.vertex_count for batch in draw_batches)
        return draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count

    def _ensure_mesh_draw_indirect_scratch(self, command_capacity: int) -> None:
        needed = max(1, int(command_capacity))
        target_capacity = max(256, 1 << (needed - 1).bit_length())
        if (
            self._mesh_draw_indirect_capacity >= target_capacity
            and self._mesh_draw_indirect_buffer is not None
        ):
            return
        old_buffer = self._mesh_draw_indirect_buffer
        self._mesh_draw_indirect_capacity = target_capacity
        self._mesh_draw_indirect_buffer = self.device.create_buffer(
            size=max(INDIRECT_DRAW_COMMAND_STRIDE, target_capacity * INDIRECT_DRAW_COMMAND_STRIDE),
            usage=wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._mesh_draw_indirect_array = np.empty((target_capacity, 4), dtype=np.uint32)
        if old_buffer is not None:
            self._schedule_gpu_buffer_cleanup([old_buffer], frames=6)

    def _ensure_mesh_visibility_scratch(self, record_capacity: int) -> None:
        needed = max(1, int(record_capacity))
        target_capacity = max(256, 1 << (needed - 1).bit_length())
        if (
            self._mesh_visibility_record_capacity >= target_capacity
            and self._mesh_visibility_record_buffer is not None
            and self._mesh_draw_indirect_capacity >= target_capacity
            and self._mesh_draw_indirect_buffer is not None
        ):
            return
        old_buffer = self._mesh_visibility_record_buffer
        self._mesh_visibility_record_capacity = target_capacity
        self._mesh_visibility_record_buffer = self.device.create_buffer(
            size=max(MESH_VISIBILITY_RECORD_DTYPE.itemsize, target_capacity * MESH_VISIBILITY_RECORD_DTYPE.itemsize),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._mesh_visibility_record_array = np.empty(target_capacity, dtype=MESH_VISIBILITY_RECORD_DTYPE)
        self._ensure_mesh_draw_indirect_scratch(target_capacity)
        if old_buffer is not None:
            self._schedule_gpu_buffer_cleanup([old_buffer], frames=6)

    def _build_gpu_visibility_records(
        self,
        encoder,
    ) -> tuple[list[tuple[wgpu.GPUBuffer, int, int, int]], int, int, int, int]:
        visible_meshes = [mesh for _, _, mesh in self._visible_chunks() if mesh.vertex_count > 0]
        draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count = self._build_tile_draw_batches(
            visible_meshes,
            encoder,
            age_gate=True,
        )

        groups: OrderedDict[tuple[int, int], tuple[wgpu.GPUBuffer, int, list[ChunkDrawBatch]]] = OrderedDict()
        for batch in draw_batches:
            key = (id(batch.vertex_buffer), batch.binding_offset)
            if key not in groups:
                groups[key] = (batch.vertex_buffer, batch.binding_offset, [])
            groups[key][2].append(batch)

        command_count = len(draw_batches)
        self._ensure_mesh_visibility_scratch(command_count)
        metadata_buffer = self._mesh_visibility_record_buffer
        metadata_array = self._mesh_visibility_record_array
        params_buffer = self._mesh_visibility_params_buffer
        assert metadata_buffer is not None
        assert params_buffer is not None

        render_batches: list[tuple[wgpu.GPUBuffer, int, int, int]] = []
        command_index = 0
        for vertex_buffer, binding_offset, batches in groups.values():
            batch_start = command_index
            for batch in batches:
                metadata_array[command_index]["bounds"] = batch.bounds
                metadata_array[command_index]["draw"] = (
                    batch.vertex_count,
                    1,
                    batch.first_vertex,
                    0,
                )
                command_index += 1
            render_batches.append((vertex_buffer, binding_offset, batch_start, command_index - batch_start))

        if command_count > 0:
            self.device.queue.write_buffer(metadata_buffer, 0, memoryview(metadata_array[:command_count]))
            self.device.queue.write_buffer(params_buffer, 0, struct.pack("<4I", int(command_count), 0, 0, 0))

        return render_batches, command_count, merged_chunk_count, visible_chunk_count, visible_vertex_count

    def _visible_render_batches_indirect(
        self,
        encoder,
    ) -> tuple[list[tuple[wgpu.GPUBuffer, int, int, int]], float, int, int, int, int]:
        encode_start = time.perf_counter()
        visible_meshes = [mesh for _, _, mesh in self._visible_chunks() if mesh.vertex_count > 0]
        draw_batches, merged_chunk_count, visible_chunk_count, visible_vertex_count = self._build_tile_draw_batches(
            visible_meshes,
            encoder,
            age_gate=True,
        )

        groups: OrderedDict[tuple[int, int], tuple[wgpu.GPUBuffer, int, list[ChunkDrawBatch]]] = OrderedDict()
        for batch in draw_batches:
            key = (id(batch.vertex_buffer), batch.binding_offset)
            if key not in groups:
                groups[key] = (batch.vertex_buffer, batch.binding_offset, [])
            groups[key][2].append(batch)

        command_count = len(draw_batches)
        self._ensure_mesh_draw_indirect_scratch(command_count)
        indirect_buffer = self._mesh_draw_indirect_buffer
        indirect_array = self._mesh_draw_indirect_array
        assert indirect_buffer is not None

        render_batches: list[tuple[wgpu.GPUBuffer, int, int, int]] = []
        command_index = 0
        for vertex_buffer, binding_offset, batches in groups.values():
            batch_start = command_index
            for batch in batches:
                indirect_array[command_index, 0] = np.uint32(batch.vertex_count)
                indirect_array[command_index, 1] = np.uint32(1)
                indirect_array[command_index, 2] = np.uint32(batch.first_vertex)
                indirect_array[command_index, 3] = np.uint32(0)
                command_index += 1
            render_batches.append((vertex_buffer, binding_offset, batch_start, command_index - batch_start))

        if command_count > 0:
            self.device.queue.write_buffer(indirect_buffer, 0, memoryview(indirect_array[:command_count]))

        render_encode_ms = (time.perf_counter() - encode_start) * 1000.0
        return render_batches, render_encode_ms, command_count, merged_chunk_count, visible_chunk_count, visible_vertex_count

    def _update_camera(self, dt: float) -> None:
        sprinting = self._key_active("shift", "shiftleft", "shiftright")
        speed = SPRINT_FLY_SPEED if sprinting else self.camera.move_speed
        self._current_move_speed = float(speed)
        move = [0.0, 0.0, 0.0]

        forward = flat_forward_vector(self.camera.yaw)
        right = right_vector(self.camera.yaw)

        if self._key_active("w", "arrowup"):
            move[0] += forward[0]
            move[2] += forward[2]
        if self._key_active("s", "arrowdown"):
            move[0] -= forward[0]
            move[2] -= forward[2]
        if self._key_active("d", "arrowright"):
            move[0] += right[0]
            move[2] += right[2]
        if self._key_active("a", "arrowleft"):
            move[0] -= right[0]
            move[2] -= right[2]
        if self._key_active("x"):
            move[1] += 1.0
        if self._key_active("z"):
            move[1] -= 1.0

        length = math.sqrt(move[0] * move[0] + move[1] * move[1] + move[2] * move[2])
        if length > 0.0:
            scale = speed * dt / length
            self.camera.position[0] += move[0] * scale
            self.camera.position[1] += move[1] * scale
            self.camera.position[2] += move[2] * scale

        self.camera.position[1] = clamp(self.camera.position[1], 4.0, self.world.height + 48.0)

    def _ensure_depth_buffer(self) -> None:
        width, height = self.canvas.get_physical_size()
        if (width, height) == self.depth_size:
            return
        self.depth_size = (width, height)
        self.depth_texture = self.device.create_texture(
            size=(max(1, width), max(1, height), 1),
            format=DEPTH_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.depth_view = self.depth_texture.create_view()

    def _camera_forward(self) -> tuple[float, float, float]:
        return forward_vector(self.camera.yaw, self.camera.pitch)

    def _camera_basis(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        forward = normalize3(self._camera_forward())
        world_up = (0.0, 1.0, 0.0)
        right = cross3(forward, world_up)
        if right == (0.0, 0.0, 0.0):
            right = right_vector(self.camera.yaw)
        right = normalize3(right)
        up = normalize3(cross3(right, forward))
        return right, up, forward

    def _chunk_coords_in_view(self) -> list[tuple[int, int]]:
        chunk_x = int(self.camera.position[0] // CHUNK_SIZE)
        chunk_z = int(self.camera.position[2] // CHUNK_SIZE)
        coords: list[tuple[int, int]] = []
        for dz in range(-self.chunk_radius, self.chunk_radius + 1):
            for dx in range(-self.chunk_radius, self.chunk_radius + 1):
                coords.append((chunk_x + dx, chunk_z + dz))
        return coords

    def _refresh_visible_chunk_coords(self) -> None:
        self._visible_chunk_origin = (
            int(self.camera.position[0] // CHUNK_SIZE),
            int(self.camera.position[2] // CHUNK_SIZE),
        )
        self._visible_chunk_coords = self._chunk_coords_in_view()
        self._visible_chunk_coord_set = set(self._visible_chunk_coords)

    def _warn_if_visible_exceeds_cache(self) -> None:
        visible_count = len(self._visible_chunk_coords)
        if self._cache_capacity_warned or visible_count <= self.max_cached_chunks:
            return
        self._cache_capacity_warned = True
        print(
            f"Warning: visible chunk count ({visible_count}) exceeds cache capacity "
            f"({self.max_cached_chunks}). Expect missing chunks, evictions, or flashing.",
            file=sys.stderr,
        )

    @profile
    def _chunk_prep_priority(
        self,
        chunk_x: int,
        chunk_z: int,
        camera_chunk_x: int,
        camera_chunk_z: int,
        forward_x: float,
        forward_z: float,
        right_x: float,
        right_z: float,
    ) -> tuple[float, float, int, int]:
        dx = chunk_x - camera_chunk_x
        dz = chunk_z - camera_chunk_z
        forward_score = dx * forward_x + dz * forward_z
        lateral_score = abs(dx * right_x + dz * right_z)
        distance_sq = float(dx * dx + dz * dz)
        distance_score = distance_sq
        cone_distance_threshold = 1.0 / (1.0 + CHUNK_FORWARD_CONE_LATERAL_RATIO * CHUNK_FORWARD_CONE_LATERAL_RATIO)
        in_forward_cone = forward_score > 0.0 and forward_score * forward_score >= cone_distance_threshold * distance_sq
        cone_flag = 0 if in_forward_cone else 1
        if in_forward_cone:
            return (cone_flag, distance_score, lateral_score, -forward_score, abs(dz), abs(dx))
        return (cone_flag, distance_score, lateral_score, -forward_score, abs(dz), abs(dx))

    @profile
    def _chunk_in_forward_cone(
        self,
        chunk_x: int,
        chunk_z: int,
        camera_chunk_x: int,
        camera_chunk_z: int,
        forward_x: float,
        forward_z: float,
        right_x: float,
        right_z: float,
    ) -> bool:
        dx = chunk_x - camera_chunk_x
        dz = chunk_z - camera_chunk_z
        forward_score = dx * forward_x + dz * forward_z
        if forward_score <= 0.0:
            return False
        lateral_score = abs(dx * right_x + dz * right_z)
        return lateral_score <= forward_score * CHUNK_FORWARD_CONE_LATERAL_RATIO

    def _visible_chunks(self) -> list[tuple[int, int, ChunkMesh]]:
        visible: list[tuple[int, int, ChunkMesh]] = []
        if not self._visible_chunk_coords:
            self._refresh_visible_chunk_coords()
        for chunk_x, chunk_z in self._visible_chunk_coords:
            mesh = self.chunk_cache.get((chunk_x, chunk_z))
            if mesh is None:
                continue
            self.chunk_cache.move_to_end((chunk_x, chunk_z))
            visible.append((chunk_x, chunk_z, mesh))
        return visible

    def _tile_key(self, chunk_x: int, chunk_z: int) -> tuple[int, int]:
        return chunk_x // MERGED_TILE_SIZE_CHUNKS, chunk_z // MERGED_TILE_SIZE_CHUNKS

    def _visible_render_batches(
        self,
        encoder,
    ) -> tuple[list[tuple[wgpu.GPUBuffer, int, int]], float, int, int, int, int]:
        encode_start = time.perf_counter()
        visible_chunks = self._visible_chunks()

        tile_groups: dict[tuple[int, int], list[tuple[int, int, ChunkMesh]]] = {}
        for chunk_x, chunk_z, mesh in visible_chunks:
            tile_groups.setdefault(self._tile_key(chunk_x, chunk_z), []).append((chunk_x, chunk_z, mesh))

        render_batches: list[tuple[wgpu.GPUBuffer, int, int]] = []
        merged_chunk_count = 0

        current_tile_keys = set(tile_groups.keys())
        stale_keys = [tile_key for tile_key in self._tile_render_batches if tile_key not in current_tile_keys]
        for tile_key in stale_keys:
            batch = self._tile_render_batches.pop(tile_key)
            self._transient_render_buffers.append([batch.vertex_buffer])

        for tile_key in sorted(tile_groups):
            tile_chunks = tile_groups[tile_key]
            mature_chunks = [(chunk_x, chunk_z, mesh) for chunk_x, chunk_z, mesh in tile_chunks if self._chunk_mesh_age(mesh) >= MERGED_TILE_MIN_AGE_SECONDS]
            immature_chunks = [(chunk_x, chunk_z, mesh) for chunk_x, chunk_z, mesh in tile_chunks if self._chunk_mesh_age(mesh) < MERGED_TILE_MIN_AGE_SECONDS]

            if len(tile_chunks) == 1:
                _, _, mesh = tile_chunks[0]
                render_batches.append((mesh.vertex_buffer, mesh.vertex_count, mesh.vertex_offset))
                continue

            existing = self._tile_render_batches.get(tile_key)
            if len(mature_chunks) < 2:
                if existing is not None:
                    self._tile_render_batches.pop(tile_key, None)
                    self._transient_render_buffers.append([existing.vertex_buffer])
                for _, _, mesh in immature_chunks:
                    render_batches.append((mesh.vertex_buffer, mesh.vertex_count, mesh.vertex_offset))
                for _, _, mesh in mature_chunks:
                    render_batches.append((mesh.vertex_buffer, mesh.vertex_count, mesh.vertex_offset))
                continue

            merged_chunk_count += len(mature_chunks)
            batch_vertex_count = sum(mesh.vertex_count for _, _, mesh in mature_chunks)
            if (
                existing is None
                or existing.signature != tuple((chunk_x, chunk_z) for chunk_x, chunk_z, _ in mature_chunks)
                or existing.vertex_count != batch_vertex_count
            ):
                old_buffer = existing.vertex_buffer if existing is not None else None
                merged_buffer = self._merge_tile_meshes([mesh for _, _, mesh in mature_chunks], encoder)
                self._tile_render_batches[tile_key] = ChunkRenderBatch(
                    signature=tuple((chunk_x, chunk_z) for chunk_x, chunk_z, _ in mature_chunks),
                    vertex_count=batch_vertex_count,
                    vertex_buffer=merged_buffer,
                )
                if old_buffer is not None:
                    self._transient_render_buffers.append([old_buffer])

            batch = self._tile_render_batches[tile_key]
            render_batches.append((batch.vertex_buffer, batch.vertex_count, 0))
            for _, _, mesh in immature_chunks:
                render_batches.append((mesh.vertex_buffer, mesh.vertex_count, mesh.vertex_offset))

        while len(self._transient_render_buffers) > 3:
            old_buffers = self._transient_render_buffers.pop(0)
            for buffer in old_buffers:
                buffer.destroy()

        render_encode_ms = (time.perf_counter() - encode_start) * 1000.0
        visible_vertex_count = sum(vertex_count for _, vertex_count, _ in render_batches)
        return render_batches, render_encode_ms, len(render_batches), merged_chunk_count, len(visible_chunks), visible_vertex_count

    def _build_chunk_vertex_array(
        self,
        voxel_grid,
        material_grid,
        chunk_x: int,
        chunk_z: int,
    ) -> tuple[np.ndarray, int, int]:
        vertex_array, vertex_count = build_chunk_vertex_array_from_voxels(
            voxel_grid,
            material_grid,
            chunk_x,
            chunk_z,
            CHUNK_SIZE,
            WORLD_HEIGHT,
        )
        used_vertex_count = int(vertex_count)
        used_vertex_array = np.ascontiguousarray(vertex_array[:used_vertex_count])
        chunk_max_height = (
            int(voxel_grid.shape[0])
            if getattr(voxel_grid, "ndim", 0) == 3
            else int(np.max(voxel_grid))
        )
        return used_vertex_array, used_vertex_count, chunk_max_height

    def _cpu_make_chunk_mesh_batch_from_voxels(
        self,
        chunk_results: list[ChunkVoxelResult],
    ) -> list[ChunkMesh]:
        if not chunk_results:
            return []

        built_chunks: list[tuple[int, int, np.ndarray, int, int, int]] = []
        total_vertex_bytes = 0
        created_at = time.perf_counter()

        for result in chunk_results:
            chunk_x = int(result.chunk_x)
            chunk_z = int(result.chunk_z)
            vertex_array, vertex_count, chunk_max_height = self._build_chunk_vertex_array(
                result.blocks,
                result.materials,
                chunk_x,
                chunk_z,
            )
            vertex_bytes = int(vertex_count) * VERTEX_STRIDE
            built_chunks.append(
                (
                    chunk_x,
                    chunk_z,
                    vertex_array,
                    int(vertex_count),
                    int(vertex_bytes),
                    int(chunk_max_height),
                )
            )
            total_vertex_bytes += vertex_bytes

        batch_allocation = self._allocate_mesh_output_range(total_vertex_bytes)
        batch_buffer = batch_allocation.buffer
        batch_base_offset = batch_allocation.offset_bytes

        meshes: list[ChunkMesh] = []
        cursor_bytes = 0
        for chunk_x, chunk_z, vertex_array, vertex_count, vertex_bytes, chunk_max_height in built_chunks:
            vertex_offset = batch_base_offset + cursor_bytes
            if vertex_bytes > 0:
                vertex_view = memoryview(vertex_array.view(np.uint8).reshape(-1))
                self.device.queue.write_buffer(batch_buffer, vertex_offset, vertex_view)
            meshes.append(
                ChunkMesh(
                    chunk_x=chunk_x,
                    chunk_z=chunk_z,
                    vertex_count=vertex_count,
                    vertex_buffer=batch_buffer,
                    vertex_offset=vertex_offset,
                    max_height=chunk_max_height,
                    created_at=created_at,
                    allocation_id=batch_allocation.allocation_id,
                )
            )
            cursor_bytes += vertex_bytes

        return meshes


    def _cpu_make_chunk_mesh_from_voxels(
        self,
        chunk_x: int,
        chunk_z: int,
        voxel_grid,
        material_grid,
    ) -> ChunkMesh:
        meshes = self._cpu_make_chunk_mesh_batch_from_voxels(
            [
                ChunkVoxelResult(
                    chunk_x=int(chunk_x),
                    chunk_z=int(chunk_z),
                    blocks=np.ascontiguousarray(voxel_grid, dtype=np.uint32),
                    materials=np.ascontiguousarray(material_grid, dtype=np.uint32),
                    source="cpu",
                )
            ]
        )
        return meshes[0]

    def _ensure_voxel_mesh_batch_scratch(
        self,
        sample_size: int,
        height_limit: int,
        chunk_capacity: int | None = None,
    ) -> None:
        capacity = max(1, int(chunk_capacity if chunk_capacity is not None else self.mesh_batch_size))
        if (
            self._voxel_mesh_scratch_capacity >= capacity
            and self._voxel_mesh_scratch_sample_size == sample_size
            and self._voxel_mesh_scratch_height_limit == height_limit
            and self._voxel_mesh_scratch_blocks_buffer is not None
        ):
            return

        self._voxel_mesh_scratch_capacity = capacity
        self._voxel_mesh_scratch_sample_size = int(sample_size)
        self._voxel_mesh_scratch_height_limit = int(height_limit)
        max_chunk_count = capacity
        max_column_plane = max(1, (sample_size - 2) * (sample_size - 2))
        blocks_bytes = max_chunk_count * height_limit * sample_size * sample_size * 4
        coords_bytes = max_chunk_count * 2 * 4
        column_totals_bytes = max_chunk_count * max_column_plane * 4
        chunk_totals_bytes = max_chunk_count * 4

        self._voxel_mesh_scratch_blocks_buffer = self.device.create_buffer(
            size=max(1, blocks_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        self._voxel_mesh_scratch_materials_buffer = self.device.create_buffer(
            size=max(1, blocks_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        self._voxel_mesh_scratch_coords_buffer = self.device.create_buffer(
            size=max(1, coords_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        self._voxel_mesh_scratch_column_totals_buffer = self.device.create_buffer(
            size=max(1, column_totals_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        self._voxel_mesh_scratch_chunk_totals_buffer = self.device.create_buffer(
            size=max(1, chunk_totals_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        self._voxel_mesh_scratch_chunk_offsets_buffer = self.device.create_buffer(
            size=max(1, chunk_totals_bytes),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        self._voxel_mesh_scratch_chunk_metadata_readback_buffer = self.device.create_buffer(
            size=max(8, max_chunk_count * 8),
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        self._voxel_mesh_scratch_params_buffer = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._voxel_mesh_scratch_batch_vertex_buffer = self.device.create_buffer(
            size=VERTEX_STRIDE,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        self._voxel_mesh_scratch_blocks_array = np.empty(
            (max_chunk_count, height_limit, sample_size, sample_size),
            dtype=np.uint32,
        )
        self._voxel_mesh_scratch_materials_array = np.empty(
            (max_chunk_count, height_limit, sample_size, sample_size),
            dtype=np.uint32,
        )
        self._voxel_mesh_scratch_coords_array = np.empty((max_chunk_count, 2), dtype=np.int32)
        self._voxel_mesh_scratch_chunk_totals_array = np.empty(max_chunk_count, dtype=np.uint32)
        self._voxel_mesh_scratch_chunk_offsets_array = np.empty(max_chunk_count, dtype=np.uint32)

    def _create_async_voxel_mesh_batch_resources(
        self,
        sample_size: int,
        height_limit: int,
        chunk_count: int,
    ) -> AsyncVoxelMeshBatchResources:
        self._ensure_voxel_mesh_batch_scratch(sample_size, height_limit, 1)
        dummy_vertex_buffer = self._voxel_mesh_scratch_batch_vertex_buffer
        assert dummy_vertex_buffer is not None

        chunk_capacity = max(1, int(chunk_count))
        column_capacity = max(1, (sample_size - 2) * (sample_size - 2) * chunk_capacity)
        blocks_bytes = chunk_capacity * height_limit * sample_size * sample_size * 4
        coords_bytes = chunk_capacity * 2 * 4
        chunk_totals_bytes = chunk_capacity * 4
        resources = AsyncVoxelMeshBatchResources(
            sample_size=int(sample_size),
            height_limit=int(height_limit),
            chunk_capacity=chunk_capacity,
            column_capacity=column_capacity,
            blocks_buffer=self.device.create_buffer(
                size=max(1, blocks_bytes),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            ),
            materials_buffer=self.device.create_buffer(
                size=max(1, blocks_bytes),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            ),
            coords_buffer=self.device.create_buffer(
                size=max(1, coords_bytes),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            ),
            column_totals_buffer=self.device.create_buffer(
                size=max(1, column_capacity * 4),
                usage=wgpu.BufferUsage.STORAGE,
            ),
            chunk_totals_buffer=self.device.create_buffer(
                size=max(1, chunk_totals_bytes),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
            ),
            chunk_offsets_buffer=self.device.create_buffer(
                size=max(1, chunk_totals_bytes),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            ),
            params_buffer=self.device.create_buffer(
                size=16,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            ),
            readback_buffer=self.device.create_buffer(
                size=max(8, chunk_totals_bytes),
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            ),
            coords_array=np.empty((chunk_capacity, 2), dtype=np.int32),
            zero_counts_array=np.zeros(chunk_capacity, dtype=np.uint32),
        )
        resources.count_bind_group = self.device.create_bind_group(
            layout=self.voxel_mesh_count_bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": resources.blocks_buffer}},
                {"binding": 1, "resource": {"buffer": resources.materials_buffer}},
                {"binding": 2, "resource": {"buffer": resources.coords_buffer}},
                {"binding": 3, "resource": {"buffer": resources.column_totals_buffer}},
                {"binding": 4, "resource": {"buffer": resources.chunk_totals_buffer}},
                {"binding": 5, "resource": {"buffer": resources.chunk_offsets_buffer}},
                {"binding": 6, "resource": {"buffer": dummy_vertex_buffer}},
                {"binding": 8, "resource": {"buffer": resources.params_buffer, "offset": 0, "size": 16}},
            ],
        )
        return resources

    def _async_voxel_mesh_batch_resources_match(
        self,
        resources: AsyncVoxelMeshBatchResources,
        sample_size: int,
        height_limit: int,
        chunk_count: int,
    ) -> bool:
        return (
            resources.sample_size == int(sample_size)
            and resources.height_limit == int(height_limit)
            and resources.chunk_capacity >= max(1, int(chunk_count))
        )

    def _destroy_async_voxel_mesh_batch_resources(self, resources: AsyncVoxelMeshBatchResources) -> None:
        if resources.readback_buffer.map_state != "unmapped":
            try:
                resources.readback_buffer.unmap()
            except Exception:
                pass
        for buffer in (
            resources.blocks_buffer,
            resources.materials_buffer,
            resources.coords_buffer,
            resources.column_totals_buffer,
            resources.chunk_totals_buffer,
            resources.chunk_offsets_buffer,
            resources.params_buffer,
            resources.readback_buffer,
        ):
            try:
                buffer.destroy()
            except Exception:
                pass

    def _acquire_async_voxel_mesh_batch_resources(
        self,
        sample_size: int,
        height_limit: int,
        chunk_count: int,
    ) -> AsyncVoxelMeshBatchResources:
        target_chunk_count = max(1, int(chunk_count))
        pool_size = len(self._async_voxel_mesh_batch_pool)
        for _ in range(pool_size):
            resources = self._async_voxel_mesh_batch_pool.popleft()
            if self._async_voxel_mesh_batch_resources_match(resources, sample_size, height_limit, target_chunk_count):
                return resources
            self._async_voxel_mesh_batch_pool.append(resources)
        return self._create_async_voxel_mesh_batch_resources(sample_size, height_limit, target_chunk_count)

    def _release_async_voxel_mesh_batch_resources(self, resources: AsyncVoxelMeshBatchResources) -> None:
        if len(self._async_voxel_mesh_batch_pool) >= self._async_voxel_mesh_batch_pool_limit:
            self._destroy_async_voxel_mesh_batch_resources(resources)
            return
        self._async_voxel_mesh_batch_pool.append(resources)

    def _schedule_async_voxel_mesh_batch_resource_release(
        self,
        resources: AsyncVoxelMeshBatchResources,
        frames: int = 2,
    ) -> None:
        self._gpu_mesh_deferred_batch_resource_releases.append((max(1, int(frames)), resources))

    def _schedule_gpu_buffer_cleanup(self, buffers: list[wgpu.GPUBuffer], frames: int = 2) -> None:
        if buffers:
            self._gpu_mesh_deferred_buffer_cleanup.append((max(1, int(frames)), list(buffers)))

    def _process_gpu_buffer_cleanup(self) -> None:
        if self._gpu_mesh_deferred_buffer_cleanup:
            next_queue: deque[tuple[int, list[wgpu.GPUBuffer]]] = deque()
            while self._gpu_mesh_deferred_buffer_cleanup:
                frames_left, buffers = self._gpu_mesh_deferred_buffer_cleanup.popleft()
                frames_left -= 1
                if frames_left <= 0:
                    for buffer in buffers:
                        try:
                            buffer.destroy()
                        except Exception:
                            pass
                else:
                    next_queue.append((frames_left, buffers))
            self._gpu_mesh_deferred_buffer_cleanup = next_queue

        if self._gpu_mesh_deferred_batch_resource_releases:
            next_resource_queue: deque[tuple[int, AsyncVoxelMeshBatchResources]] = deque()
            while self._gpu_mesh_deferred_batch_resource_releases:
                frames_left, resources = self._gpu_mesh_deferred_batch_resource_releases.popleft()
                frames_left -= 1
                if frames_left <= 0:
                    self._release_async_voxel_mesh_batch_resources(resources)
                else:
                    next_resource_queue.append((frames_left, resources))
            self._gpu_mesh_deferred_batch_resource_releases = next_resource_queue

    def _enqueue_gpu_chunk_mesh_batch_from_gpu_buffers(
        self,
        chunk_coords: list[tuple[int, int]],
        resources: AsyncVoxelMeshBatchResources,
        sample_size: int,
        height_limit: int,
        *,
        params_already_uploaded: bool = False,
        encoder = None,
        submit: bool = True,
    ) -> None:
        if (
            self.voxel_mesh_count_pipeline is None
            or self.voxel_mesh_emit_pipeline is None
        ):
            raise RuntimeError("GPU meshing pipeline is unavailable.")
        if not chunk_coords:
            return

        chunk_coords_list = chunk_coords if isinstance(chunk_coords, list) else list(chunk_coords)
        chunk_count = len(chunk_coords_list)
        columns_per_side = sample_size - 2
        if not self._async_voxel_mesh_batch_resources_match(resources, sample_size, height_limit, chunk_count):
            raise RuntimeError("Async voxel mesh batch resources do not match requested batch size.")
        count_bind_group = resources.count_bind_group
        assert count_bind_group is not None

        coords_view = resources.coords_array[:chunk_count]
        coords_view[:] = np.asarray(chunk_coords_list, dtype=np.int32)
        zero_view = resources.zero_counts_array[:chunk_count]
        zero_view.fill(0)
        params_bytes = struct.pack(
            "<4I",
            int(sample_size),
            int(height_limit),
            int(chunk_count),
            int(CHUNK_SIZE),
        )

        self.device.queue.write_buffer(resources.coords_buffer, 0, memoryview(coords_view))
        self.device.queue.write_buffer(resources.chunk_totals_buffer, 0, memoryview(zero_view))
        if not params_already_uploaded:
            self.device.queue.write_buffer(resources.params_buffer, 0, params_bytes)

        if encoder is None:
            encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.voxel_mesh_count_pipeline)
        compute_pass.set_bind_group(0, count_bind_group)
        compute_pass.dispatch_workgroups(columns_per_side, columns_per_side, chunk_count)
        compute_pass.end()
        encoder.copy_buffer_to_buffer(resources.chunk_totals_buffer, 0, resources.readback_buffer, 0, chunk_count * 4)
        if submit:
            self.device.queue.submit([encoder.finish()])

        metadata_promise = resources.readback_buffer.map_async(wgpu.MapMode.READ, 0, chunk_count * 4)
        self._pending_gpu_mesh_batches.append(
            PendingChunkMeshBatch(
                chunk_coords=chunk_coords_list,
                chunk_count=chunk_count,
                sample_size=int(sample_size),
                height_limit=int(height_limit),
                columns_per_side=int(columns_per_side),
                blocks_buffer=resources.blocks_buffer,
                materials_buffer=resources.materials_buffer,
                coords_buffer=resources.coords_buffer,
                column_totals_buffer=resources.column_totals_buffer,
                chunk_totals_buffer=resources.chunk_totals_buffer,
                chunk_offsets_buffer=resources.chunk_offsets_buffer,
                params_buffer=resources.params_buffer,
                readback_buffer=resources.readback_buffer,
                resources=resources,
                metadata_promise=metadata_promise,
                submitted_at=time.perf_counter(),
            )
        )

    def _finalize_pending_gpu_mesh_batches(self, budget: int | None = None) -> int:
        if not self._pending_gpu_mesh_batches:
            return 0
        budget = max(1, int(self._gpu_mesh_async_finalize_budget if budget is None else budget))
        completed = 0
        remaining: deque[PendingChunkMeshBatch] = deque()
        while self._pending_gpu_mesh_batches:
            pending = self._pending_gpu_mesh_batches.popleft()
            if pending.readback_buffer.map_state != "mapped" or completed >= budget:
                remaining.append(pending)
                continue

            totals_nbytes = pending.chunk_count * 4
            try:
                metadata_view = pending.readback_buffer.read_mapped(0, totals_nbytes, copy=False)
                chunk_totals = np.frombuffer(metadata_view, dtype=np.uint32, count=pending.chunk_count).copy()
            finally:
                pending.readback_buffer.unmap()

            chunk_offsets = np.empty(pending.chunk_count, dtype=np.uint32)
            if pending.chunk_count > 0:
                np.cumsum(chunk_totals, dtype=np.uint32, out=chunk_offsets)
                chunk_offsets -= chunk_totals
            self.device.queue.write_buffer(pending.chunk_offsets_buffer, 0, memoryview(chunk_offsets))

            total_vertices = int(chunk_totals.sum(dtype=np.uint64))
            total_vertex_bytes = total_vertices * VERTEX_STRIDE
            batch_allocation = self._allocate_mesh_output_range(total_vertex_bytes)
            emit_bind_group = self.device.create_bind_group(
                layout=self.voxel_mesh_emit_bind_group_layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": pending.blocks_buffer}},
                    {"binding": 1, "resource": {"buffer": pending.materials_buffer}},
                    {"binding": 2, "resource": {"buffer": pending.coords_buffer}},
                    {"binding": 3, "resource": {"buffer": pending.column_totals_buffer}},
                    {"binding": 4, "resource": {"buffer": pending.chunk_totals_buffer}},
                    {"binding": 5, "resource": {"buffer": pending.chunk_offsets_buffer}},
                    {
                        "binding": 6,
                        "resource": {
                            "buffer": batch_allocation.buffer,
                            "offset": batch_allocation.offset_bytes,
                            "size": max(1, batch_allocation.size_bytes),
                        },
                    },
                    {"binding": 8, "resource": {"buffer": pending.params_buffer, "offset": 0, "size": 16}},
                ],
            )
            encoder = self.device.create_command_encoder()
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.voxel_mesh_emit_pipeline)
            compute_pass.set_bind_group(0, emit_bind_group)
            compute_pass.dispatch_workgroups(pending.columns_per_side, pending.columns_per_side, pending.chunk_count)
            compute_pass.end()
            self.device.queue.submit([encoder.finish()])

            created_at = time.perf_counter()
            for chunk_index, (chunk_x, chunk_z) in enumerate(pending.chunk_coords):
                self._store_chunk_mesh(
                    ChunkMesh(
                        chunk_x=int(chunk_x),
                        chunk_z=int(chunk_z),
                        vertex_count=int(chunk_totals[chunk_index]),
                        vertex_buffer=batch_allocation.buffer,
                        vertex_offset=batch_allocation.offset_bytes + int(chunk_offsets[chunk_index]) * VERTEX_STRIDE,
                        max_height=pending.height_limit,
                        created_at=created_at,
                        allocation_id=batch_allocation.allocation_id,
                    )
                )

            if pending.resources is not None:
                self._schedule_async_voxel_mesh_batch_resource_release(pending.resources, frames=2)
            else:
                self._schedule_gpu_buffer_cleanup(
                    [
                        pending.blocks_buffer,
                        pending.materials_buffer,
                        pending.coords_buffer,
                        pending.column_totals_buffer,
                        pending.chunk_totals_buffer,
                        pending.chunk_offsets_buffer,
                        pending.params_buffer,
                        pending.readback_buffer,
                    ],
                    frames=2,
                )
            completed += 1
        self._pending_gpu_mesh_batches = remaining
        return completed

    def _read_chunk_mesh_batch_metadata(
        self,
        chunk_totals_buffer,
        chunk_offsets_buffer,
        chunk_count: int,
        *,
        include_offsets: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if chunk_count <= 0:
            empty = np.empty(0, dtype=np.uint32)
            return empty, empty if include_offsets else None

        readback_buffer = self._voxel_mesh_scratch_chunk_metadata_readback_buffer
        assert readback_buffer is not None

        totals_nbytes = chunk_count * 4
        metadata_nbytes = totals_nbytes * (2 if include_offsets else 1)
        if readback_buffer.map_state != "unmapped":
            readback_buffer.unmap()

        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(chunk_totals_buffer, 0, readback_buffer, 0, totals_nbytes)
        if include_offsets:
            encoder.copy_buffer_to_buffer(chunk_offsets_buffer, 0, readback_buffer, totals_nbytes, totals_nbytes)
        self.device.queue.submit([encoder.finish()])

        # map_sync() is the unavoidable GPU->CPU wait in this path, so keep the
        # mapped range as small as possible on the hot path.
        readback_buffer.map_sync(wgpu.MapMode.READ, 0, metadata_nbytes)
        try:
            metadata_view = readback_buffer.read_mapped(0, metadata_nbytes, copy=False)
            totals = np.frombuffer(metadata_view, dtype=np.uint32, count=chunk_count).copy()
            if include_offsets:
                offsets = np.frombuffer(metadata_view, dtype=np.uint32, count=chunk_count, offset=totals_nbytes).copy()
            else:
                offsets = None
        finally:
            readback_buffer.unmap()

        return totals, offsets

    def _gpu_make_chunk_mesh_batch_from_voxels(
        self,
        chunk_results: list[ChunkVoxelResult],
        *,
        defer_finalize: bool = False,
    ) -> list[ChunkMesh]:
        if (
            self.voxel_mesh_count_pipeline is None
            or self.voxel_mesh_scan_pipeline is None
            or self.voxel_mesh_emit_pipeline is None
        ):
            raise RuntimeError("GPU meshing pipeline is unavailable.")
        if not chunk_results:
            return []

        chunk_count = len(chunk_results)
        sample_size = int(chunk_results[0].blocks.shape[1])
        height_limit = int(chunk_results[0].blocks.shape[0])
        columns_per_side = sample_size - 2
        column_plane = columns_per_side * columns_per_side
        self._ensure_voxel_mesh_batch_scratch(sample_size, height_limit, chunk_count)

        blocks = self._voxel_mesh_scratch_blocks_array
        materials = self._voxel_mesh_scratch_materials_array
        chunk_coords = self._voxel_mesh_scratch_coords_array
        chunk_totals = self._voxel_mesh_scratch_chunk_totals_array
        chunk_offsets = self._voxel_mesh_scratch_chunk_offsets_array

        blocks_buffer = self._voxel_mesh_scratch_blocks_buffer
        materials_buffer = self._voxel_mesh_scratch_materials_buffer
        coords_buffer = self._voxel_mesh_scratch_coords_buffer
        column_totals_buffer = self._voxel_mesh_scratch_column_totals_buffer
        chunk_totals_buffer = self._voxel_mesh_scratch_chunk_totals_buffer
        chunk_offsets_buffer = self._voxel_mesh_scratch_chunk_offsets_buffer
        batch_vertex_buffer = self._voxel_mesh_scratch_batch_vertex_buffer
        params_buffer = self._voxel_mesh_scratch_params_buffer

        assert blocks is not None
        assert materials is not None
        assert chunk_coords is not None
        assert chunk_totals is not None
        assert chunk_offsets is not None
        assert blocks_buffer is not None
        assert materials_buffer is not None
        assert coords_buffer is not None
        assert column_totals_buffer is not None
        assert chunk_totals_buffer is not None
        assert chunk_offsets_buffer is not None
        assert batch_vertex_buffer is not None
        assert params_buffer is not None

        chunk_coords_list: list[tuple[int, int]] = []
        for index, result in enumerate(chunk_results):
            chunk_x = int(result.chunk_x)
            chunk_z = int(result.chunk_z)
            blocks[index] = result.blocks
            materials[index] = result.materials
            chunk_coords[index, 0] = chunk_x
            chunk_coords[index, 1] = chunk_z
            chunk_coords_list.append((chunk_x, chunk_z))

        if defer_finalize:
            dedicated = self._create_dedicated_voxel_mesh_batch_buffers(sample_size, height_limit, chunk_count)
            blocks_buffer = dedicated["blocks"]
            materials_buffer = dedicated["materials"]
            self.device.queue.write_buffer(blocks_buffer, 0, memoryview(blocks[:chunk_count]))
            self.device.queue.write_buffer(materials_buffer, 0, memoryview(materials[:chunk_count]))
            self._enqueue_gpu_chunk_mesh_batch_from_gpu_buffers(
                chunk_coords_list,
                blocks_buffer,
                materials_buffer,
                sample_size,
                height_limit,
            )
            return []

        self.device.queue.write_buffer(blocks_buffer, 0, memoryview(blocks[:chunk_count]))
        self.device.queue.write_buffer(materials_buffer, 0, memoryview(materials[:chunk_count]))
        return self._gpu_make_chunk_mesh_batch_from_gpu_buffers(
            chunk_coords_list,
            blocks_buffer,
            materials_buffer,
            sample_size,
            height_limit,
        )

    def _gpu_make_chunk_mesh_batch_from_gpu_buffers(
        self,
        chunk_coords: list[tuple[int, int]],
        blocks_buffer,
        materials_buffer,
        sample_size: int,
        height_limit: int,
    ) -> list[ChunkMesh]:
        if (
            self.voxel_mesh_count_pipeline is None
            or self.voxel_mesh_scan_pipeline is None
            or self.voxel_mesh_emit_pipeline is None
        ):
            raise RuntimeError("GPU meshing pipeline is unavailable.")
        if not chunk_coords:
            return []

        chunk_count = len(chunk_coords)
        columns_per_side = sample_size - 2
        self._ensure_voxel_mesh_batch_scratch(sample_size, height_limit, chunk_count)

        chunk_coords_array = self._voxel_mesh_scratch_coords_array
        chunk_totals = self._voxel_mesh_scratch_chunk_totals_array
        chunk_offsets = self._voxel_mesh_scratch_chunk_offsets_array
        coords_buffer = self._voxel_mesh_scratch_coords_buffer
        column_totals_buffer = self._voxel_mesh_scratch_column_totals_buffer
        chunk_totals_buffer = self._voxel_mesh_scratch_chunk_totals_buffer
        chunk_offsets_buffer = self._voxel_mesh_scratch_chunk_offsets_buffer
        batch_vertex_buffer = self._voxel_mesh_scratch_batch_vertex_buffer
        params_buffer = self._voxel_mesh_scratch_params_buffer

        assert chunk_coords_array is not None
        assert chunk_totals is not None
        assert chunk_offsets is not None
        assert coords_buffer is not None
        assert column_totals_buffer is not None
        assert chunk_totals_buffer is not None
        assert chunk_offsets_buffer is not None
        assert batch_vertex_buffer is not None
        assert params_buffer is not None

        for index, (chunk_x, chunk_z) in enumerate(chunk_coords):
            chunk_coords_array[index, 0] = int(chunk_x)
            chunk_coords_array[index, 1] = int(chunk_z)

        self.device.queue.write_buffer(coords_buffer, 0, memoryview(chunk_coords_array[:chunk_count]))
        self.device.queue.write_buffer(
            params_buffer,
            0,
            struct.pack(
                "<4I",
                int(sample_size),
                int(height_limit),
                int(chunk_count),
                int(CHUNK_SIZE),
            ),
        )

        chunk_totals[:chunk_count].fill(0)
        self.device.queue.write_buffer(chunk_totals_buffer, 0, memoryview(chunk_totals[:chunk_count]))

        shared_entries = [
            {"binding": 0, "resource": {"buffer": blocks_buffer}},
            {"binding": 1, "resource": {"buffer": materials_buffer}},
            {"binding": 2, "resource": {"buffer": coords_buffer}},
            {"binding": 3, "resource": {"buffer": column_totals_buffer}},
            {"binding": 4, "resource": {"buffer": chunk_totals_buffer}},
            {"binding": 5, "resource": {"buffer": chunk_offsets_buffer}},
            {"binding": 6, "resource": {"buffer": batch_vertex_buffer}},
            {"binding": 8, "resource": {"buffer": params_buffer, "offset": 0, "size": 16}},
        ]
        count_bind_group = self.device.create_bind_group(
            layout=self.voxel_mesh_count_bind_group_layout,
            entries=shared_entries,
        )
        scan_bind_group = self.device.create_bind_group(
            layout=self.voxel_mesh_scan_bind_group_layout,
            entries=shared_entries,
        )
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.voxel_mesh_count_pipeline)
        compute_pass.set_bind_group(0, count_bind_group)
        compute_pass.dispatch_workgroups(columns_per_side, columns_per_side, chunk_count)
        compute_pass.end()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.voxel_mesh_scan_pipeline)
        compute_pass.set_bind_group(0, scan_bind_group)
        compute_pass.dispatch_workgroups(1, 1, chunk_count)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])

        validate_scan = (
            self.voxel_mesh_scan_validate_every > 0
            and (self._voxel_mesh_scan_batches_processed % self.voxel_mesh_scan_validate_every) == 0
        )
        self._voxel_mesh_scan_batches_processed += 1

        chunk_totals_readback, chunk_offsets_readback = self._read_chunk_mesh_batch_metadata(
            chunk_totals_buffer,
            chunk_offsets_buffer,
            chunk_count,
            include_offsets=validate_scan,
        )

        np.copyto(chunk_totals[:chunk_count], chunk_totals_readback)
        if chunk_count > 0:
            np.cumsum(chunk_totals_readback, dtype=np.uint32, out=chunk_offsets[:chunk_count])
            chunk_offsets[:chunk_count] -= chunk_totals_readback
            if validate_scan and chunk_offsets_readback is not None:
                if not np.array_equal(chunk_offsets_readback, chunk_offsets[:chunk_count]):
                    self.device.queue.write_buffer(chunk_offsets_buffer, 0, memoryview(chunk_offsets[:chunk_count]))
        else:
            chunk_offsets[:0] = 0

        total_vertices = int(chunk_totals_readback.sum(dtype=np.uint64))
        total_vertex_bytes = total_vertices * VERTEX_STRIDE
        batch_allocation = self._allocate_mesh_output_range(total_vertex_bytes)
        vertex_buffer = batch_allocation.buffer
        emit_bind_group = self.device.create_bind_group(
            layout=self.voxel_mesh_emit_bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": blocks_buffer}},
                {"binding": 1, "resource": {"buffer": materials_buffer}},
                {"binding": 2, "resource": {"buffer": coords_buffer}},
                {"binding": 3, "resource": {"buffer": column_totals_buffer}},
                {"binding": 4, "resource": {"buffer": chunk_totals_buffer}},
                {"binding": 5, "resource": {"buffer": chunk_offsets_buffer}},
                {
                    "binding": 6,
                    "resource": {
                        "buffer": vertex_buffer,
                        "offset": batch_allocation.offset_bytes,
                        "size": max(1, batch_allocation.size_bytes),
                    },
                },
                {"binding": 8, "resource": {"buffer": params_buffer, "offset": 0, "size": 16}},
            ],
        )
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.voxel_mesh_emit_pipeline)
        compute_pass.set_bind_group(0, emit_bind_group)
        compute_pass.dispatch_workgroups(columns_per_side, columns_per_side, chunk_count)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])

        meshes: list[ChunkMesh] = []
        for chunk_index, (chunk_x, chunk_z) in enumerate(chunk_coords):
            meshes.append(
                ChunkMesh(
                    chunk_x=int(chunk_x),
                    chunk_z=int(chunk_z),
                    vertex_count=int(chunk_totals[chunk_index]),
                    vertex_buffer=vertex_buffer,
                    vertex_offset=batch_allocation.offset_bytes + int(chunk_offsets[chunk_index]) * VERTEX_STRIDE,
                    max_height=height_limit,
                    created_at=time.perf_counter(),
                    allocation_id=batch_allocation.allocation_id,
                )
            )
        return meshes

    @profile
    def _gpu_make_chunk_mesh_batch_from_surface_gpu_batch(
        self,
        surface_batch: ChunkSurfaceGpuBatch,
        *,
        defer_finalize: bool = False,
    ) -> list[ChunkMesh]:
        if self.voxel_surface_expand_pipeline is None:
            raise RuntimeError("GPU surface expansion pipeline is unavailable.")
        if not surface_batch.chunks:
            return []

        chunk_coords = surface_batch.chunks if isinstance(surface_batch.chunks, list) else list(surface_batch.chunks)
        chunk_count = len(chunk_coords)
        sample_size = CHUNK_SAMPLE_SIZE
        height_limit = WORLD_HEIGHT
        params_bytes = struct.pack(
            "<4I",
            int(sample_size),
            int(height_limit),
            int(chunk_count),
            int(CHUNK_SIZE),
        )
        async_resources: AsyncVoxelMeshBatchResources | None = None
        if defer_finalize:
            async_resources = self._acquire_async_voxel_mesh_batch_resources(sample_size, height_limit, chunk_count)
            blocks_buffer = async_resources.blocks_buffer
            materials_buffer = async_resources.materials_buffer
            params_buffer = async_resources.params_buffer
        else:
            self._ensure_voxel_mesh_batch_scratch(sample_size, height_limit, chunk_count)
            blocks_buffer = self._voxel_mesh_scratch_blocks_buffer
            materials_buffer = self._voxel_mesh_scratch_materials_buffer
            params_buffer = self._voxel_mesh_scratch_params_buffer
        assert blocks_buffer is not None
        assert materials_buffer is not None
        assert params_buffer is not None

        self.device.queue.write_buffer(params_buffer, 0, params_bytes)
        expand_bind_group = self.device.create_bind_group(
            layout=self.voxel_surface_expand_bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": surface_batch.heights_buffer}},
                {"binding": 1, "resource": {"buffer": surface_batch.materials_buffer}},
                {"binding": 2, "resource": {"buffer": blocks_buffer}},
                {"binding": 3, "resource": {"buffer": materials_buffer}},
                {"binding": 4, "resource": {"buffer": params_buffer, "offset": 0, "size": 16}},
            ],
        )
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.voxel_surface_expand_pipeline)
        compute_pass.set_bind_group(0, expand_bind_group)
        workgroups = (sample_size + 7) // 8
        compute_pass.dispatch_workgroups(workgroups, workgroups, chunk_count * height_limit)
        compute_pass.end()

        if defer_finalize:
            assert async_resources is not None
            self._enqueue_gpu_chunk_mesh_batch_from_gpu_buffers(
                chunk_coords,
                async_resources,
                sample_size,
                height_limit,
                params_already_uploaded=True,
                encoder=encoder,
                submit=True,
            )
            return []

        self.device.queue.submit([encoder.finish()])

        return self._gpu_make_chunk_mesh_batch_from_gpu_buffers(
            chunk_coords,
            blocks_buffer,
            materials_buffer,
            sample_size,
            height_limit,
        )

    def _gpu_make_chunk_mesh_from_voxels(
        self,
        chunk_x: int,
        chunk_z: int,
        voxel_grid,
        material_grid,
    ) -> ChunkMesh:
        meshes = self._gpu_make_chunk_mesh_batch_from_voxels(
            [
                ChunkVoxelResult(
                    chunk_x=chunk_x,
                    chunk_z=chunk_z,
                    blocks=np.ascontiguousarray(voxel_grid, dtype=np.uint32),
                    materials=np.ascontiguousarray(material_grid, dtype=np.uint32),
                    source="gpu",
                )
            ]
            )
        if not meshes:
            empty_buffer = self.device.create_buffer(
                size=1,
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            )
            return ChunkMesh(chunk_x=chunk_x, chunk_z=chunk_z, vertex_count=0, vertex_buffer=empty_buffer, max_height=int(voxel_grid.shape[0]), vertex_offset=0, created_at=time.perf_counter())
        return meshes[0]

    def _make_chunk_mesh_from_voxels(
        self,
        chunk_x: int,
        chunk_z: int,
        voxel_grid,
        material_grid,
    ) -> ChunkMesh:
        if self.use_gpu_meshing:
            try:
                return self._gpu_make_chunk_mesh_from_voxels(chunk_x, chunk_z, voxel_grid, material_grid)
            except Exception as exc:
                self.use_gpu_meshing = False
                self.mesh_backend_label = "CPU"
                print(f"Warning: GPU meshing failed ({exc!s}); using CPU meshing.", file=sys.stderr)
        return self._cpu_make_chunk_mesh_from_voxels(chunk_x, chunk_z, voxel_grid, material_grid)

    def _make_chunk_mesh(self, chunk_x: int, chunk_z: int) -> ChunkMesh:
        voxel_grid, material_grid = self.world.chunk_voxel_grid(chunk_x, chunk_z)
        return self._make_chunk_mesh_from_voxels(chunk_x, chunk_z, voxel_grid, material_grid)

    def _accept_chunk_voxel_result(self, result) -> None:
        key = (int(result.chunk_x), int(result.chunk_z))
        self._pending_chunk_coords.discard(key)
        mesh = self._make_chunk_mesh_from_voxels(key[0], key[1], result.blocks, result.materials)
        self._store_chunk_mesh(mesh)

    def _ensure_chunk_mesh(self, chunk_x: int, chunk_z: int) -> ChunkMesh:
        key = (chunk_x, chunk_z)
        mesh = self.chunk_cache.get(key)
        if mesh is not None:
            self.chunk_cache.move_to_end(key)
            return mesh

        mesh = self._make_chunk_mesh(chunk_x, chunk_z)
        self._store_chunk_mesh(mesh)
        return mesh

    def _rebuild_visible_missing_tracking(self) -> None:
        displayed: set[tuple[int, int]] = set()
        missing: set[tuple[int, int]] = set()
        pending = self._pending_chunk_coords
        for coord in self._visible_chunk_coords:
            if coord in self.chunk_cache:
                displayed.add(coord)
            elif coord not in pending:
                missing.add(coord)
        self._visible_displayed_coords = displayed
        self._visible_missing_coords = missing
        self._chunk_request_queue.clear()
        self._chunk_request_queue_origin = None
        self._chunk_request_queue_dirty = True

    def _chunk_request_queue_needs_rebuild(self, current_origin: tuple[int, int]) -> bool:
        if self._chunk_request_queue_dirty:
            return True
        if self._chunk_request_queue_origin != current_origin:
            return True
        if self._visible_missing_coords and not self._chunk_request_queue:
            return True
        yaw_delta = abs(math.atan2(
            math.sin(self.camera.yaw - self._chunk_request_queue_yaw),
            math.cos(self.camera.yaw - self._chunk_request_queue_yaw),
        ))
        return yaw_delta >= CHUNK_PREP_REORDER_YAW_DELTA

    def _rebuild_chunk_request_queue(
        self,
        camera_chunk_x: int,
        camera_chunk_z: int,
    ) -> None:
        if not self._visible_missing_coords:
            self._chunk_request_queue.clear()
            self._chunk_request_queue_origin = (camera_chunk_x, camera_chunk_z)
            self._chunk_request_queue_yaw = self.camera.yaw
            self._chunk_request_queue_dirty = False
            return

        forward = flat_forward_vector(self.camera.yaw)
        right = right_vector(self.camera.yaw)
        ordered = sorted(
            self._visible_missing_coords,
            key=lambda coord: self._chunk_prep_priority(
                coord[0],
                coord[1],
                camera_chunk_x,
                camera_chunk_z,
                forward[0],
                forward[2],
                right[0],
                right[2],
            ),
        )
        self._chunk_request_queue = deque(ordered)
        self._chunk_request_queue_origin = (camera_chunk_x, camera_chunk_z)
        self._chunk_request_queue_yaw = self.camera.yaw
        self._chunk_request_queue_dirty = False

    def _refresh_visible_chunk_set(self) -> float:
        visibility_start = time.perf_counter()
        current_origin = (int(self.camera.position[0] // CHUNK_SIZE), int(self.camera.position[2] // CHUNK_SIZE))
        if self._visible_chunk_origin != current_origin or not self._visible_chunk_coords:
            self._visible_chunk_origin = current_origin
            self._visible_chunk_coords = self._chunk_coords_in_view()
            self._visible_chunk_coord_set = set(self._visible_chunk_coords)
            self._rebuild_visible_missing_tracking()
        self._warn_if_visible_exceeds_cache()
        return (time.perf_counter() - visibility_start) * 1000.0

    def _service_background_gpu_work(self) -> None:
        self._process_gpu_buffer_cleanup()
        if self.use_gpu_meshing:
            self._finalize_pending_gpu_mesh_batches()

    @profile
    def _prepare_chunks(self, dt: float) -> tuple[float, float]:
        visibility_lookup_ms = 0.0
        current_origin = (int(self.camera.position[0] // CHUNK_SIZE), int(self.camera.position[2] // CHUNK_SIZE))
        if self._visible_chunk_origin != current_origin or not self._visible_chunk_coords:
            visibility_lookup_ms = self._refresh_visible_chunk_set()

        prep_start = time.perf_counter()
        forward = flat_forward_vector(self.camera.yaw)
        right = right_vector(self.camera.yaw)
        using_gpu_terrain = self.world.terrain_backend_label() == "GPU"
        if self.use_gpu_meshing:
            ready_surface_batches = self.world.poll_ready_chunk_surface_gpu_batches()
            if ready_surface_batches:
                chunk_stream_added = sum(len(batch.chunks) for batch in ready_surface_batches)
                chunk_stream_drained = 0
                chunk_stream_bytes = 0.0
                for batch in ready_surface_batches:
                    chunk_stream_drained += len(batch.chunks)
                    chunk_stream_bytes += float(batch.cell_count * 8 * len(batch.chunks))
                    self._gpu_make_chunk_mesh_batch_from_surface_gpu_batch(batch, defer_finalize=True)
            elif not using_gpu_terrain:
                ready_results = self.world.poll_ready_chunk_voxel_batches()
                chunk_stream_bytes = 0.0
                chunk_stream_added = len(ready_results)
                chunk_stream_drained = 0
                drain_budget = max(1, self.mesh_batch_size)
                ready_results.sort(
                    key=lambda result: self._chunk_prep_priority(
                        int(result.chunk_x),
                        int(result.chunk_z),
                        current_origin[0],
                        current_origin[1],
                        forward[0],
                        forward[2],
                        right[0],
                        right[2],
                    )
                )
                for result in ready_results:
                    chunk_stream_bytes += float(result.blocks.nbytes + result.materials.nbytes)
                for result in reversed(ready_results):
                    self._pending_voxel_mesh_results.appendleft(result)
                while self._pending_voxel_mesh_results and drain_budget > 0:
                    batch_size = min(self.mesh_batch_size, len(self._pending_voxel_mesh_results))
                    batch_size = min(batch_size, drain_budget)
                    batch = [self._pending_voxel_mesh_results.popleft() for _ in range(batch_size)]
                    chunk_stream_drained += len(batch)
                    drain_budget -= len(batch)
                    for result in batch:
                        chunk_stream_bytes += float(result.blocks.nbytes + result.materials.nbytes)
                    for mesh in self._cpu_make_chunk_mesh_batch_from_voxels(batch):
                        self._store_chunk_mesh(mesh)
            else:
                chunk_stream_added = 0
                chunk_stream_drained = 0
                chunk_stream_bytes = 0.0
        else:
            ready_results = self.world.poll_ready_chunk_voxel_batches()
            chunk_stream_bytes = 0.0
            chunk_stream_added = len(ready_results)
            chunk_stream_drained = 0
            drain_budget = max(1, self.mesh_batch_size)
            ready_results.sort(
                key=lambda result: self._chunk_prep_priority(
                    int(result.chunk_x),
                    int(result.chunk_z),
                    current_origin[0],
                    current_origin[1],
                    forward[0],
                    forward[2],
                    right[0],
                    right[2],
                )
            )
            for result in reversed(ready_results):
                self._pending_voxel_mesh_results.appendleft(result)
            while self._pending_voxel_mesh_results and drain_budget > 0:
                batch_size = min(self.mesh_batch_size, len(self._pending_voxel_mesh_results))
                batch_size = min(batch_size, drain_budget)
                batch = [self._pending_voxel_mesh_results.popleft() for _ in range(batch_size)]
                chunk_stream_drained += len(batch)
                drain_budget -= len(batch)
                for result in batch:
                    chunk_stream_bytes += float(result.blocks.nbytes + result.materials.nbytes)
                    self._accept_chunk_voxel_result(result)

        self._chunk_prep_tokens = min(CHUNK_PREP_TOKEN_CAP, self._chunk_prep_tokens + CHUNK_PREP_RATE * dt)

        candidate_cap = max(1, min(self.mesh_batch_size, CHUNK_PREP_REQUEST_BUDGET_CAP))
        prep_budget = int(self._chunk_prep_tokens)
        missing_coords = self._visible_missing_coords
        missing_count = len(missing_coords)
        if prep_budget <= 0 and missing_count > 0:
            prep_budget = 1
        prep_budget = min(prep_budget, missing_count)
        prep_budget = min(prep_budget, candidate_cap)

        camera_chunk_x = current_origin[0]
        camera_chunk_z = current_origin[1]
        pending_chunk_coords = self._pending_chunk_coords
        displayed_chunk_coords = self._visible_displayed_coords

        if prep_budget > 0 and missing_count > 0 and self._chunk_request_queue_needs_rebuild(current_origin):
            self._rebuild_chunk_request_queue(camera_chunk_x, camera_chunk_z)

        request_coords: list[tuple[int, int]] = []
        request_queue = self._chunk_request_queue
        while request_queue and len(request_coords) < prep_budget:
            coord = request_queue.popleft()
            if coord in pending_chunk_coords or coord not in missing_coords:
                continue
            request_coords.append(coord)

        if not request_coords and prep_budget > 0 and missing_count > 0 and self._visible_missing_coords:
            self._rebuild_chunk_request_queue(camera_chunk_x, camera_chunk_z)
            request_queue = self._chunk_request_queue
            while request_queue and len(request_coords) < prep_budget:
                coord = request_queue.popleft()
                if coord in pending_chunk_coords or coord not in missing_coords:
                    continue
                request_coords.append(coord)

        if request_coords:
            batch_size = max(1, min(self.mesh_batch_size, CHUNK_PREP_REQUEST_BATCH_SIZE))
            request_batches = [
                request_coords[index : index + batch_size]
                for index in range(0, len(request_coords), batch_size)
            ]
            for batch in reversed(request_batches):
                self.world.request_chunk_voxel_batch(batch)
            for coord in request_coords:
                pending_chunk_coords.add(coord)
                missing_coords.discard(coord)
            self._chunk_prep_tokens -= float(len(request_coords))

        self._last_new_displayed_chunks = len(displayed_chunk_coords - self._last_displayed_chunk_coords)
        self._last_displayed_chunk_coords = set(displayed_chunk_coords)
        chunk_stream_ms = (time.perf_counter() - prep_start) * 1000.0
        self._last_chunk_stream_drained = chunk_stream_drained
        self._record_frame_breakdown_sample("chunk_stream_bytes", chunk_stream_bytes)
        return visibility_lookup_ms, chunk_stream_ms

    def _submit_render(self):
        self._ensure_depth_buffer()
        right, up, forward = self._camera_basis()
        camera_upload_start = time.perf_counter()
        self.device.queue.write_buffer(
            self.camera_buffer,
            0,
            pack_camera_uniform(
                tuple(self.camera.position),
                right,
                up,
                forward,
                1.0 / math.tan(math.radians(90.0) * 0.5),
                max(1.0, self.canvas.get_physical_size()[0] / max(1.0, float(self.canvas.get_physical_size()[1]))),
                0.1,
                1024.0,
                LIGHT_DIRECTION,
            ),
        )
        camera_upload_ms = (time.perf_counter() - camera_upload_start) * 1000.0

        encoder = self.device.create_command_encoder()
        use_gpu_visibility = bool(
            self.use_gpu_indirect_render
            and self.use_gpu_built_visibility
            and self.mesh_visibility_pipeline is not None
            and self._mesh_output_slabs
            and hasattr(wgpu.GPURenderCommandsMixin, "draw_indirect")
        )
        use_indirect = bool(
            self.use_gpu_indirect_render
            and self._mesh_allocations
            and hasattr(wgpu.GPURenderCommandsMixin, "draw_indirect")
        )
        if use_gpu_visibility:
            gpu_visibility_start = time.perf_counter()
            visible_batches, draw_calls, merged_batches, visible_chunks, visible_vertices = self._build_gpu_visibility_records(encoder)
            if draw_calls > 0:
                metadata_buffer = self._mesh_visibility_record_buffer
                indirect_buffer = self._mesh_draw_indirect_buffer
                params_buffer = self._mesh_visibility_params_buffer
                assert metadata_buffer is not None
                assert indirect_buffer is not None
                assert params_buffer is not None
                visibility_bind_group = self.device.create_bind_group(
                    layout=self.mesh_visibility_bind_group_layout,
                    entries=[
                        {"binding": 0, "resource": {"buffer": metadata_buffer}},
                        {"binding": 1, "resource": {"buffer": indirect_buffer}},
                        {"binding": 2, "resource": {"buffer": self.camera_buffer, "offset": 0, "size": 80}},
                        {"binding": 3, "resource": {"buffer": params_buffer, "offset": 0, "size": 16}},
                    ],
                )
                compute_pass = encoder.begin_compute_pass()
                compute_pass.set_pipeline(self.mesh_visibility_pipeline)
                compute_pass.set_bind_group(0, visibility_bind_group)
                compute_pass.dispatch_workgroups((draw_calls + GPU_VISIBILITY_WORKGROUP_SIZE - 1) // GPU_VISIBILITY_WORKGROUP_SIZE, 1, 1)
                compute_pass.end()
            render_encode_ms = (time.perf_counter() - gpu_visibility_start) * 1000.0
        elif use_indirect:
            visible_batches, render_encode_ms, draw_calls, merged_batches, visible_chunks, visible_vertices = self._visible_render_batches_indirect(encoder)
        else:
            visible_batches, render_encode_ms, draw_calls, merged_batches, visible_chunks, visible_vertices = self._visible_render_batches(encoder)

        # Drawable acquisition can block when GPU/display work backs up, so keep it
        # measured separately in the HUD instead of folding it into generic encode time.
        acquire_start = time.perf_counter()
        current_texture = self.context.get_current_texture()
        swapchain_acquire_ms = (time.perf_counter() - acquire_start) * 1000.0
        color_view = current_texture.create_view()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": color_view,
                    "resolve_target": None,
                    "clear_value": (0.60, 0.80, 0.98, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )
        render_pass.set_pipeline(self.render_pipeline)
        render_pass.set_bind_group(0, self.camera_bind_group)

        if use_gpu_visibility or use_indirect:
            indirect_buffer = self._mesh_draw_indirect_buffer
            assert indirect_buffer is not None
            for vertex_buffer, binding_offset, batch_start, batch_count in visible_batches:
                render_pass.set_vertex_buffer(0, vertex_buffer, binding_offset)
                if wgpu_native_multi_draw_indirect is not None and batch_count > 1:
                    wgpu_native_multi_draw_indirect(
                        render_pass,
                        indirect_buffer,
                        offset=batch_start * INDIRECT_DRAW_COMMAND_STRIDE,
                        count=batch_count,
                    )
                    continue
                for batch_index in range(batch_count):
                    indirect_offset = (batch_start + batch_index) * INDIRECT_DRAW_COMMAND_STRIDE
                    render_pass.draw_indirect(indirect_buffer, indirect_offset)
        else:
            for vertex_buffer, vertex_count, vertex_offset in visible_batches:
                render_pass.set_vertex_buffer(0, vertex_buffer, vertex_offset)
                render_pass.draw(vertex_count, 1, 0, 0)

        render_pass.end()
        stats = {
            "camera_upload_ms": camera_upload_ms,
            "visibility_lookup_ms": 0.0,
            "swapchain_acquire_ms": swapchain_acquire_ms,
            "render_encode_ms": render_encode_ms,
            "draw_calls": draw_calls,
            "merged_chunks": merged_batches,
            "visible_chunks": visible_chunks,
            "visible_vertices": visible_vertices,
        }
        return encoder, color_view, stats

    def draw_frame(self) -> None:
        frame_start = time.perf_counter()
        now = frame_start
        dt = min(0.05, now - self.last_frame_time)
        self.last_frame_time = now
        profile_started_at = self._profile_begin_frame()
        try:
            update_start = time.perf_counter()
            self._update_camera(dt)
            world_update_ms = (time.perf_counter() - update_start) * 1000.0

            visibility_lookup_ms = self._refresh_visible_chunk_set()
            encoder, color_view, render_stats = self._submit_render()
            visibility_lookup_ms += render_stats["visibility_lookup_ms"]
            camera_upload_ms = render_stats["camera_upload_ms"]
            swapchain_acquire_ms = render_stats["swapchain_acquire_ms"]
            render_encode_ms = render_stats["render_encode_ms"]
            draw_calls = int(render_stats["draw_calls"])
            merged_chunks = int(render_stats["merged_chunks"])
            visible_chunks = int(render_stats["visible_chunks"])
            visible_vertices = int(render_stats["visible_vertices"])

            self._draw_profile_hud(encoder, color_view)
            self._draw_frame_breakdown_hud(encoder, color_view)

            command_finish_start = time.perf_counter()
            command_buffer = encoder.finish()
            command_finish_ms = (time.perf_counter() - command_finish_start) * 1000.0

            queue_submit_start = time.perf_counter()
            self.device.queue.submit([command_buffer])
            queue_submit_ms = (time.perf_counter() - queue_submit_start) * 1000.0

            self._service_background_gpu_work()
            _, chunk_stream_ms = self._prepare_chunks(dt)

            wall_frame_ms = (time.perf_counter() - frame_start) * 1000.0

            self._record_frame_breakdown_sample("world_update", world_update_ms)
            self._record_frame_breakdown_sample("visibility_lookup", visibility_lookup_ms)
            self._record_frame_breakdown_sample("chunk_stream", chunk_stream_ms)
            self._record_frame_breakdown_sample("chunk_displayed_added", float(self._last_new_displayed_chunks))
            self._record_frame_breakdown_sample("camera_upload", camera_upload_ms)
            self._record_frame_breakdown_sample("swapchain_acquire", swapchain_acquire_ms)
            self._record_frame_breakdown_sample("render_encode", render_encode_ms)
            self._record_frame_breakdown_sample("command_finish", command_finish_ms)
            self._record_frame_breakdown_sample("queue_submit", queue_submit_ms)
            self._record_frame_breakdown_sample("wall_frame", wall_frame_ms)
            self._record_frame_breakdown_sample("draw_calls", float(draw_calls))
            self._record_frame_breakdown_sample("merged_chunks", float(merged_chunks))
            self._record_frame_breakdown_sample("visible_vertices", float(visible_vertices))
            self._record_frame_breakdown_sample("visible_chunk_targets", float(len(self._visible_chunk_coords)))
            self._record_frame_breakdown_sample("visible_chunks", float(visible_chunks))
            self._record_frame_breakdown_sample("pending_chunk_requests", float(len(self._pending_chunk_coords)))

            self._refresh_frame_breakdown_summary()
        except Exception:
            if profile_started_at is not None:
                self._profile_end_frame(profile_started_at, 0.0)
            raise
        self._profile_end_frame(profile_started_at, wall_frame_ms / 1000.0)
        self.canvas.request_draw(self.draw_frame)
