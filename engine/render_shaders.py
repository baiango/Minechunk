from __future__ import annotations

from . import render_constants as render_consts

INDIRECT_DRAW_COMMAND_STRIDE = render_consts.INDIRECT_DRAW_COMMAND_STRIDE
GPU_VISIBILITY_WORKGROUP_SIZE = render_consts.GPU_VISIBILITY_WORKGROUP_SIZE
MESH_OUTPUT_FREERANGE_SCAN_LIMIT = render_consts.MESH_OUTPUT_FREERANGE_SCAN_LIMIT
MESH_VISIBILITY_RECORD_DTYPE = render_consts.MESH_VISIBILITY_RECORD_DTYPE
VOXEL_SURFACE_EXPAND_BIND_GROUP_CACHE_LIMIT = render_consts.VOXEL_SURFACE_EXPAND_BIND_GROUP_CACHE_LIMIT

HUD_FONT_SCALE = render_consts.HUD_FONT_SCALE
HUD_FONT_CHAR_WIDTH = render_consts.HUD_FONT_CHAR_WIDTH
HUD_FONT_CHAR_HEIGHT = render_consts.HUD_FONT_CHAR_HEIGHT
HUD_PANEL_PADDING = render_consts.HUD_PANEL_PADDING
HUD_LINE_SPACING = render_consts.HUD_LINE_SPACING
HUD_GLYPH_SPACING = render_consts.HUD_GLYPH_SPACING
PROFILE_REPORT_INTERVAL = render_consts.PROFILE_REPORT_INTERVAL
FRAME_BREAKDOWN_SAMPLE_WINDOW = render_consts.FRAME_BREAKDOWN_SAMPLE_WINDOW
SWAPCHAIN_MAX_FPS = render_consts.SWAPCHAIN_MAX_FPS
SWAPCHAIN_USE_VSYNC = render_consts.SWAPCHAIN_USE_VSYNC
SPRINT_FLY_SPEED = render_consts.SPRINT_FLY_SPEED


def build_tile_merge_shader(merged_tile_max_chunks: int) -> str:
    source_bindings = "\n".join(
        f"@group(0) @binding({index}) var<storage, read> src_{index}: VertexBuffer;"
        for index in range(merged_tile_max_chunks)
    )
    source_cases = "\n".join(
        f"        case {index}u: {{ return src_{index}.values[local_vertex]; }}"
        for index in range(merged_tile_max_chunks)
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
@group(0) @binding({merged_tile_max_chunks}) var<storage, read> merge_meta: MergeMetaBuffer;
@group(0) @binding({merged_tile_max_chunks + 1}) var<uniform> merge_params: MergeParams;
@group(0) @binding({merged_tile_max_chunks + 2}) var<storage, read_write> merged_vertices: VertexBuffer;

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


TILE_MERGE_SHADER = build_tile_merge_shader(4 * 4)

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
