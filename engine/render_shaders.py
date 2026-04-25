from __future__ import annotations

from . import render_constants as render_consts

INDIRECT_DRAW_COMMAND_STRIDE = render_consts.INDIRECT_DRAW_COMMAND_STRIDE
GPU_VISIBILITY_WORKGROUP_SIZE = render_consts.GPU_VISIBILITY_WORKGROUP_SIZE
MESH_OUTPUT_FREERANGE_SCAN_LIMIT = render_consts.MESH_OUTPUT_FREERANGE_SCAN_LIMIT
MESH_VISIBILITY_RECORD_DTYPE = render_consts.MESH_VISIBILITY_RECORD_DTYPE
VOXEL_SURFACE_EXPAND_BIND_GROUP_CACHE_LIMIT = render_consts.VOXEL_SURFACE_EXPAND_BIND_GROUP_CACHE_LIMIT

HUD_FONT_SCALE = render_consts.HUD_FONT_SCALE
HUD_FONT_CHAR_WIDTH = render_consts.HUD_FONT_CHAR_WIDTH
RADIANCE_CASCADES_BASE_RAY_COUNT = render_consts.RADIANCE_CASCADES_BASE_RAY_COUNT
RADIANCE_CASCADES_RAY_COUNT_DECAY = render_consts.RADIANCE_CASCADES_RAY_COUNT_DECAY
RADIANCE_CASCADES_CASCADE_COUNT = render_consts.RADIANCE_CASCADES_CASCADE_COUNT
RADIANCE_CASCADES_STEPS_PER_CASCADE = render_consts.RADIANCE_CASCADES_STEPS_PER_CASCADE
RADIANCE_CASCADES_MERGE_OVERLAP = render_consts.RADIANCE_CASCADES_MERGE_OVERLAP
RADIANCE_CASCADES_MERGE_STRENGTH = render_consts.RADIANCE_CASCADES_MERGE_STRENGTH
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
    let y0 = f32(height) * cell_size;
    let east_y = f32(east_height) * cell_size;
    let south_y = f32(south_height) * cell_size;

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


GI_GBUFFER_SHADER = """
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
    @location(0) world_normal: vec3f,
    @location(1) color: vec3f,
    @location(2) view_z: f32,
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @location(1) gbuffer: vec4f,
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
        out.world_normal = normalize(input.normal.xyz);
        out.color = input.color.xyz;
        out.view_z = 0.0;
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
    out.world_normal = normalize(input.normal.xyz);
    out.color = input.color.xyz;
    out.view_z = view_z;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.color = vec4f(input.color, 1.0);
    out.gbuffer = vec4f(input.world_normal, input.view_z);
    return out;
}
"""


GI_CASCADE_SHADER = """
struct CameraUniform {
    position: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
    proj: vec4f,
}

struct GiParams {
    lighting_control: vec4f,
    merge_control: vec4f,
}

struct CascadeParams {
    cascade_meta: vec4f,
    interval_meta: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_tex: texture_2d<f32>;
@group(0) @binding(2) var prev_cascade_tex: texture_2d<f32>;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var<uniform> camera: CameraUniform;
@group(0) @binding(5) var<uniform> params: GiParams;
@group(0) @binding(6) var<uniform> cascade: CascadeParams;

const MAX_CASCADES: u32 = __MAX_CASCADES__u;
const MAX_RAYS: u32 = __MAX_RAYS__u;
const MAX_STEPS: u32 = __MAX_STEPS__u;
const BASE_RAY_COUNT: f32 = __BASE_RAY_COUNT__;
const RAY_COUNT_DECAY: f32 = __RAY_COUNT_DECAY__;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0),
    );
    var out: VertexOutput;
    let pos = positions[vertex_index];
    out.position = vec4f(pos, 0.0, 1.0);
    out.uv = vec2f(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn hash12(p: vec2f) -> f32 {
    let h = dot(p, vec2f(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn view_pos_from_uv_depth(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let focal = camera.proj.x;
    let aspect = camera.proj.y;
    return vec3f(ndc.x * aspect * depth / focal, ndc.y * depth / focal, depth);
}

fn world_pos_from_view_pos(view_pos: vec3f) -> vec3f {
    return camera.position.xyz
        + camera.right.xyz * view_pos.x
        + camera.up.xyz * view_pos.y
        + camera.forward.xyz * view_pos.z;
}

fn view_pos_from_world_pos(world_pos: vec3f) -> vec3f {
    let to_point = world_pos - camera.position.xyz;
    return vec3f(
        dot(to_point, camera.right.xyz),
        dot(to_point, camera.up.xyz),
        dot(to_point, camera.forward.xyz),
    );
}

fn uv_from_view_pos(view_pos: vec3f) -> vec2f {
    let focal = camera.proj.x;
    let aspect = camera.proj.y;
    let inv_z = 1.0 / max(view_pos.z, 0.0001);
    let ndc_x = view_pos.x * focal / aspect * inv_z;
    let ndc_y = view_pos.y * focal * inv_z;
    return vec2f(ndc_x * 0.5 + 0.5, 0.5 - ndc_y * 0.5);
}

fn build_basis(n: vec3f) -> mat3x3f {
    let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.92);
    let tangent = normalize(cross(helper, n));
    let bitangent = normalize(cross(n, tangent));
    return mat3x3f(tangent, bitangent, n);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let g0 = textureSample(gbuffer_tex, linear_sampler, input.uv);
    let depth = g0.w;
    let normal_len = length(g0.xyz);
    if (depth <= 0.0 || normal_len <= 0.001) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    let normal = normalize(g0.xyz);
    let view_pos = view_pos_from_uv_depth(input.uv, depth);
    let world_pos = world_pos_from_view_pos(view_pos);
    let basis = build_basis(normal);

    let bias = max(params.lighting_control.y, 0.005);
    let sky_strength = max(params.lighting_control.z, 0.0);
    let hit_thickness = max(params.lighting_control.w, 0.02);
    let merge_overlap = max(params.merge_control.x, 0.0);
    let merge_strength = clamp(params.merge_control.y, 0.0, 1.0);
    let cascade_index = u32(cascade.cascade_meta.x);
    let has_prev = cascade.cascade_meta.y > 0.5;
    let total_cascades = max(u32(cascade.cascade_meta.z), 1u);
    let base_interval = max(cascade.interval_meta.x, 0.1);
    let interval_scale = max(cascade.interval_meta.y, 1.1);
    let steps_per_cascade = clamp(u32(cascade.interval_meta.z), 1u, MAX_STEPS);

    var start_dist = bias * 2.0;
    var interval_length = base_interval;
    for (var i: u32 = 0u; i < MAX_CASCADES; i = i + 1u) {
        if (i >= cascade_index) {
            break;
        }
        start_dist = start_dist + interval_length;
        interval_length = interval_length * interval_scale;
    }
    let end_dist = start_dist + interval_length;
    let overlap = interval_length * merge_overlap;
    let sample_start = max(bias * 2.0, start_dist - overlap);
    let sample_end = end_dist + overlap;
    let sample_length = max(sample_end - sample_start, 0.001);

    let sky_color = vec3f(0.60, 0.80, 0.98);
    let rand = hash12(input.uv * 4096.0);
    let spin = rand * 6.28318530718;

    var local_dirs = array<vec3f, 8>(
        normalize(vec3f( 0.85,  0.10, 1.00)),
        normalize(vec3f(-0.85,  0.10, 1.00)),
        normalize(vec3f( 0.10,  0.85, 1.00)),
        normalize(vec3f( 0.10, -0.85, 1.00)),
        normalize(vec3f( 0.55,  0.55, 1.00)),
        normalize(vec3f(-0.55,  0.55, 1.00)),
        normalize(vec3f( 0.55, -0.55, 1.00)),
        normalize(vec3f(-0.55, -0.55, 1.00)),
    );

    var local_rgb = vec3f(0.0, 0.0, 0.0);
    var local_opacity = 0.0;
    var total_dir_weight = 0.0;
    let origin = world_pos + normal * bias;
    let ray_budget = clamp(u32(BASE_RAY_COUNT * pow(RAY_COUNT_DECAY, f32(cascade_index))), 1u, MAX_RAYS);

    for (var ray_index: u32 = 0u; ray_index < MAX_RAYS; ray_index = ray_index + 1u) {
        if (ray_index >= ray_budget) {
            break;
        }
        let local = local_dirs[ray_index];
        let cascade_spin = spin + f32(cascade_index) * 1.61803398875;
        let ccs = cos(cascade_spin);
        let csn = sin(cascade_spin);
        let rotated_local = vec3f(
            local.x * ccs - local.y * csn,
            local.x * csn + local.y * ccs,
            local.z,
        );
        let ray_dir = normalize(basis * rotated_local);
        let n_dot_l = max(dot(normal, ray_dir), 0.0);
        if (n_dot_l <= 0.001) {
            continue;
        }
        total_dir_weight = total_dir_weight + n_dot_l;

        var hit = false;
        for (var step_index: u32 = 0u; step_index < MAX_STEPS; step_index = step_index + 1u) {
            if (step_index >= steps_per_cascade) {
                break;
            }
            let t = (f32(step_index) + 0.5) / f32(steps_per_cascade);
            let dist = sample_start + sample_length * t;
            let sample_world = origin + ray_dir * dist;
            let sample_view = view_pos_from_world_pos(sample_world);
            if (sample_view.z <= camera.proj.z || sample_view.z >= camera.proj.w) {
                continue;
            }
            let sample_uv = uv_from_view_pos(sample_view);
            if (sample_uv.x <= 0.001 || sample_uv.x >= 0.999 || sample_uv.y <= 0.001 || sample_uv.y >= 0.999) {
                if (!has_prev && cascade_index + 1u == total_cascades) {
                    local_rgb = local_rgb + sky_color * (n_dot_l * sky_strength);
                }
                break;
            }
            let g = textureSample(gbuffer_tex, linear_sampler, sample_uv);
            let hit_depth = g.w;
            if (hit_depth <= 0.0) {
                continue;
            }
            let delta = sample_view.z - hit_depth;
            if (delta >= -hit_thickness && delta <= hit_thickness * 2.5) {
                let hit_normal_len = length(g.xyz);
                if (hit_normal_len > 0.001) {
                    let hit_normal = normalize(g.xyz);
                    let facing = saturate(dot(hit_normal, -ray_dir));
                    let center_dist = abs(dist - (start_dist + end_dist) * 0.5);
                    let center_weight = 1.0 - saturate(center_dist / (interval_length * (0.5 + merge_overlap)));
                    let bounce = textureSample(scene_tex, linear_sampler, sample_uv).rgb * (0.20 + 0.80 * facing);
                    let weight = n_dot_l * (0.35 + 0.65 * center_weight);
                    local_rgb = local_rgb + bounce * weight;
                    local_opacity = local_opacity + weight;
                    hit = true;
                    break;
                }
            }
        }
        if (!hit && !has_prev && cascade_index + 1u == total_cascades) {
            local_rgb = local_rgb + sky_color * (n_dot_l * sky_strength);
        }
    }

    if (total_dir_weight > 0.0001) {
        local_rgb = local_rgb / total_dir_weight;
        local_opacity = saturate(local_opacity / total_dir_weight);
    } else {
        local_rgb = vec3f(0.0, 0.0, 0.0);
        local_opacity = 0.0;
    }

    let local_interval = vec4f(local_rgb, local_opacity);
    var prev_interval = vec4f(0.0, 0.0, 0.0, 0.0);
    if (has_prev) {
        prev_interval = textureSample(prev_cascade_tex, linear_sampler, input.uv);
    }
    let carried_rgb = mix(prev_interval.rgb, prev_interval.rgb * (1.0 - local_interval.a), merge_strength);
    let merged_rgb = local_interval.rgb + carried_rgb;
    let merged_alpha = local_interval.a + prev_interval.a * (1.0 - local_interval.a * merge_strength);
    return vec4f(merged_rgb, merged_alpha);
}
"""
GI_CASCADE_SHADER = (
    GI_CASCADE_SHADER
    .replace("__MAX_CASCADES__", str(max(1, int(RADIANCE_CASCADES_CASCADE_COUNT))))
    .replace("__MAX_RAYS__", str(max(1, int(RADIANCE_CASCADES_BASE_RAY_COUNT))))
    .replace("__MAX_STEPS__", str(max(4, int(RADIANCE_CASCADES_STEPS_PER_CASCADE))))
    .replace("__BASE_RAY_COUNT__", f"{float(RADIANCE_CASCADES_BASE_RAY_COUNT):.4f}")
    .replace("__RAY_COUNT_DECAY__", f"{float(RADIANCE_CASCADES_RAY_COUNT_DECAY):.4f}")
)

GI_COMPOSE_SHADER = """
struct CameraUniform {
    position: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
    proj: vec4f,
}

struct GiParams {
    lighting_control: vec4f,
    merge_control: vec4f,
}

struct WorldRcParams {
    volume_min: array<vec4f, 4>,
    volume_inv_extent: array<vec4f, 4>,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var gbuffer_tex: texture_2d<f32>;
@group(0) @binding(2) var rc0_tex: texture_3d<f32>;
@group(0) @binding(3) var rc1_tex: texture_3d<f32>;
@group(0) @binding(4) var rc2_tex: texture_3d<f32>;
@group(0) @binding(5) var rc3_tex: texture_3d<f32>;
@group(0) @binding(6) var rc0_vis_tex: texture_3d<f32>;
@group(0) @binding(7) var rc1_vis_tex: texture_3d<f32>;
@group(0) @binding(8) var rc2_vis_tex: texture_3d<f32>;
@group(0) @binding(9) var rc3_vis_tex: texture_3d<f32>;
@group(0) @binding(10) var linear_sampler: sampler;
@group(0) @binding(11) var<uniform> camera: CameraUniform;
@group(0) @binding(12) var<uniform> params: GiParams;
@group(0) @binding(13) var<uniform> world_rc: WorldRcParams;

const PROBE_VISIBILITY_BIAS: f32 = __PROBE_VISIBILITY_BIAS__;
const PROBE_VISIBILITY_VARIANCE_BIAS: f32 = __PROBE_VISIBILITY_VARIANCE_BIAS__;
const PROBE_VISIBILITY_SHARPNESS: f32 = __PROBE_VISIBILITY_SHARPNESS__;
const PROBE_MIN_HIT_FRACTION: f32 = __PROBE_MIN_HIT_FRACTION__;
const PROBE_BACKFACE_SOFTNESS: f32 = __PROBE_BACKFACE_SOFTNESS__;
const PROBE_CASCADE_BLEND_EDGE_START: f32 = __PROBE_CASCADE_BLEND_EDGE_START__;
const PROBE_CASCADE_BLEND_EDGE_END: f32 = __PROBE_CASCADE_BLEND_EDGE_END__;
const PROBE_CASCADE_BLEND_MIN_WEIGHT: f32 = __PROBE_CASCADE_BLEND_MIN_WEIGHT__;
const PROBE_CASCADE_BLEND_CONFIDENCE_SCALE: f32 = __PROBE_CASCADE_BLEND_CONFIDENCE_SCALE__;
const PROBE_CASCADE_BLEND_FAR_BIAS: f32 = __PROBE_CASCADE_BLEND_FAR_BIAS__;
const PROBE_CAVE_MIN_LIGHT: f32 = __PROBE_CAVE_MIN_LIGHT__;
const PROBE_CAVE_SKY_POWER: f32 = __PROBE_CAVE_SKY_POWER__;
const PROBE_CAVE_DARKENING: f32 = __PROBE_CAVE_DARKENING__;
const SKY_HORIZON: vec3f = vec3f(__SKY_HORIZON_R__, __SKY_HORIZON_G__, __SKY_HORIZON_B__);
const SKY_ZENITH: vec3f = vec3f(__SKY_ZENITH_R__, __SKY_ZENITH_G__, __SKY_ZENITH_B__);
const SKY_GROUND: vec3f = vec3f(__SKY_GROUND_R__, __SKY_GROUND_G__, __SKY_GROUND_B__);
const SKY_SUN_GLOW: vec3f = vec3f(__SKY_SUN_R__, __SKY_SUN_G__, __SKY_SUN_B__);
const SKY_SUN_DIR: vec3f = vec3f(__SKY_SUN_DIR_X__, __SKY_SUN_DIR_Y__, __SKY_SUN_DIR_Z__);

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0),
    );
    var out: VertexOutput;
    let pos = positions[vertex_index];
    out.position = vec4f(pos, 0.0, 1.0);
    out.uv = vec2f(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn view_pos_from_uv_depth(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let focal = camera.proj.x;
    let aspect = camera.proj.y;
    return vec3f(ndc.x * aspect * depth / focal, ndc.y * depth / focal, depth);
}

fn world_pos_from_view_pos(view_pos: vec3f) -> vec3f {
    return camera.position.xyz
        + camera.right.xyz * view_pos.x
        + camera.up.xyz * view_pos.y
        + camera.forward.xyz * view_pos.z;
}

fn sky_color_from_uv(uv: vec2f) -> vec3f {
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let focal = camera.proj.x;
    let aspect = camera.proj.y;
    let view_dir = normalize(
        camera.right.xyz * (ndc.x * aspect / focal)
        + camera.up.xyz * (ndc.y / focal)
        + camera.forward.xyz
    );
    let up_amount = saturate(view_dir.y);
    let down_amount = saturate(-view_dir.y);
    var sky = mix(SKY_HORIZON, SKY_ZENITH, pow(up_amount, 0.72));
    sky = mix(sky, SKY_GROUND, pow(down_amount, 0.60));
    let sun_align = max(dot(view_dir, normalize(SKY_SUN_DIR)), 0.0);
    let sun_disk = pow(sun_align, 420.0);
    let sun_haze = pow(sun_align, 18.0) * 0.18;
    return clamp(sky + SKY_SUN_GLOW * (sun_disk * 1.35 + sun_haze), vec3f(0.0), vec3f(1.0));
}

fn load_world_rc(index: u32, coord: vec3i) -> vec4f {
    switch (index) {
        case 0u: {
            return textureLoad(rc0_tex, coord, 0);
        }
        case 1u: {
            return textureLoad(rc1_tex, coord, 0);
        }
        case 2u: {
            return textureLoad(rc2_tex, coord, 0);
        }
        default: {
            return textureLoad(rc3_tex, coord, 0);
        }
    }
}

fn load_world_rc_vis(index: u32, coord: vec3i) -> vec4f {
    switch (index) {
        case 0u: {
            return textureLoad(rc0_vis_tex, coord, 0);
        }
        case 1u: {
            return textureLoad(rc1_vis_tex, coord, 0);
        }
        case 2u: {
            return textureLoad(rc2_vis_tex, coord, 0);
        }
        default: {
            return textureLoad(rc3_vis_tex, coord, 0);
        }
    }
}

fn cascade_edge_blend(uvw: vec3f) -> f32 {
    let edge_dist = min(
        min(min(uvw.x, 1.0 - uvw.x), min(uvw.y, 1.0 - uvw.y)),
        min(uvw.z, 1.0 - uvw.z),
    );
    return smoothstep(PROBE_CASCADE_BLEND_EDGE_START, PROBE_CASCADE_BLEND_EDGE_END, edge_dist);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let albedo = textureSample(scene_tex, linear_sampler, input.uv).rgb;
    let g0 = textureSample(gbuffer_tex, linear_sampler, input.uv);
    let depth = g0.w;
    let normal_len = length(g0.xyz);
    if (depth <= 0.0 || normal_len <= 0.001) {
        return vec4f(sky_color_from_uv(input.uv), 1.0);
    }

    let normal = normalize(g0.xyz);
    let bias = max(params.lighting_control.y, 0.0);
    let world_pos = world_pos_from_view_pos(view_pos_from_uv_depth(input.uv, depth)) + normal * bias;
    let strength = max(params.lighting_control.x, 0.0);
    let cascade_count = clamp(u32(params.merge_control.z), 1u, 4u);
    let resolution = max(i32(params.merge_control.w + 0.5), 2);
    let max_index = resolution - 1;

    var accum = vec3f(0.0, 0.0, 0.0);
    var accum_weight = 0.0;
    var sky_access_accum = 0.0;
    var sky_access_weight = 0.0;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        if (i >= cascade_count) {
            break;
        }
        let min_corner = world_rc.volume_min[i].xyz;
        let inv_extent = world_rc.volume_inv_extent[i].xyz;
        let uvw = (world_pos - min_corner) * inv_extent;
        if (uvw.x < 0.0 || uvw.y < 0.0 || uvw.z < 0.0 || uvw.x > 1.0 || uvw.y > 1.0 || uvw.z > 1.0) {
            continue;
        }

        let grid_f = uvw * f32(max_index);
        let base_x = i32(clamp(floor(grid_f.x), 0.0, f32(max_index - 1)));
        let base_y = i32(clamp(floor(grid_f.y), 0.0, f32(max_index - 1)));
        let base_z = i32(clamp(floor(grid_f.z), 0.0, f32(max_index - 1)));
        let frac = fract(grid_f);

        var cascade_accum = vec3f(0.0, 0.0, 0.0);
        var cascade_weight_sum = 0.0;
        var cascade_sky_access_sum = 0.0;
        var cascade_sky_access_weight = 0.0;
        for (var oz: u32 = 0u; oz < 2u; oz = oz + 1u) {
            let wz = select(1.0 - frac.z, frac.z, oz == 1u);
            for (var oy: u32 = 0u; oy < 2u; oy = oy + 1u) {
                let wy = select(1.0 - frac.y, frac.y, oy == 1u);
                for (var ox: u32 = 0u; ox < 2u; ox = ox + 1u) {
                    let wx = select(1.0 - frac.x, frac.x, ox == 1u);
                    let tri_weight = wx * wy * wz;
                    let coord = vec3i(base_x + i32(ox), base_y + i32(oy), base_z + i32(oz));
                    let probe = load_world_rc(i, coord);
                    let probe_vis = load_world_rc_vis(i, coord);
                    let probe_alpha = saturate(probe.a);
                    let hit_fraction = saturate(probe_vis.z);
                    let sky_access = saturate(probe_vis.w);
                    if (probe_alpha <= 0.0001 && hit_fraction <= 0.0001) {
                        continue;
                    }

                    let probe_uvw = vec3f(f32(coord.x), f32(coord.y), f32(coord.z)) / f32(max_index);
                    let probe_world = min_corner + probe_uvw / inv_extent;
                    let to_probe = probe_world - world_pos;
                    let probe_dist = length(to_probe);
                    if (probe_dist <= 0.0001) {
                        continue;
                    }
                    let probe_dir = to_probe / probe_dist;
                    let backface = saturate((dot(normal, probe_dir) + PROBE_BACKFACE_SOFTNESS) / (1.0 + PROBE_BACKFACE_SOFTNESS));
                    let mean_distance = probe_vis.x;
                    let mean_distance_sq = max(probe_vis.y, mean_distance * mean_distance);
                    let variance = max(mean_distance_sq - mean_distance * mean_distance, PROBE_VISIBILITY_VARIANCE_BIAS);
                    let biased_distance = max(0.0, probe_dist - PROBE_VISIBILITY_BIAS);
                    var visibility = 1.0;
                    if (biased_distance > mean_distance) {
                        let delta = biased_distance - mean_distance;
                        visibility = variance / (variance + PROBE_VISIBILITY_SHARPNESS * delta * delta);
                    }
                    let validity = mix(0.10, 1.0, smoothstep(PROBE_MIN_HIT_FRACTION, 1.0, hit_fraction));
                    let luminance = dot(probe.rgb, vec3f(0.2126, 0.7152, 0.0722));
                    let perceptual = smoothstep(0.01, 0.06, luminance);
                    let openness_weight = mix(0.03, 1.0, sky_access * sky_access);
                    let weight = tri_weight * backface * visibility * visibility * validity * validity * probe_alpha * perceptual * openness_weight;
                    let exposure_weight = tri_weight * max(0.08, backface) * visibility * validity * mix(0.20, 1.0, sky_access);
                    cascade_accum = cascade_accum + probe.rgb * weight;
                    cascade_weight_sum = cascade_weight_sum + weight;
                    cascade_sky_access_sum = cascade_sky_access_sum + sky_access * exposure_weight;
                    cascade_sky_access_weight = cascade_sky_access_weight + exposure_weight;
                }
            }
        }

        if (cascade_weight_sum <= 0.0001) {
            continue;
        }
        let cascade_rgb = cascade_accum / cascade_weight_sum;
        let cascade_alpha = saturate(cascade_weight_sum * PROBE_CASCADE_BLEND_CONFIDENCE_SCALE);
        let cascade_sky_access = select(0.0, cascade_sky_access_sum / cascade_sky_access_weight, cascade_sky_access_weight > 0.0001);
        let edge_blend = cascade_edge_blend(uvw);
        var boundary_blend = 1.0;
        if (i + 1u < cascade_count) {
            boundary_blend = mix(PROBE_CASCADE_BLEND_MIN_WEIGHT, 1.0, edge_blend);
        }
        let cascade_preference = mix(1.0, PROBE_CASCADE_BLEND_FAR_BIAS, f32(i) / max(1.0, f32(cascade_count - 1u)));
        let weight = max(0.0001, cascade_alpha * boundary_blend * cascade_preference);
        accum = accum + cascade_rgb * weight;
        accum_weight = accum_weight + weight;
        sky_access_accum = sky_access_accum + cascade_sky_access * weight;
        sky_access_weight = sky_access_weight + weight;
    }

    let raw_rc_light = select(vec3f(0.0, 0.0, 0.0), accum / accum_weight, accum_weight > 0.0001);
    let sampled_sky_access = select(0.0, sky_access_accum / sky_access_weight, sky_access_weight > 0.0001);
    let upward_surface = smoothstep(0.0, 0.85, normal.y);
    let cave_exposure = pow(saturate(sampled_sky_access), PROBE_CAVE_SKY_POWER);
    let closed_cave_floor = max(PROBE_CAVE_MIN_LIGHT, 1.0 - PROBE_CAVE_DARKENING);
    let cave_multiplier = mix(closed_cave_floor, 1.0, cave_exposure);
    let open_surface_boost = mix(0.92, 1.06, upward_surface * cave_exposure);
    let rc_light = raw_rc_light * cave_multiplier * open_surface_boost;
    let lit = clamp(albedo * rc_light * strength, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0));
    return vec4f(lit, 1.0);
}
"""
GI_COMPOSE_SHADER = (
    GI_COMPOSE_SHADER
    .replace("__PROBE_VISIBILITY_BIAS__", f"{float(render_consts.WORLDSPACE_RC_VISIBILITY_BIAS):.4f}")
    .replace("__PROBE_VISIBILITY_VARIANCE_BIAS__", f"{float(render_consts.WORLDSPACE_RC_VISIBILITY_VARIANCE_BIAS):.4f}")
    .replace("__PROBE_VISIBILITY_SHARPNESS__", f"{float(render_consts.WORLDSPACE_RC_VISIBILITY_SHARPNESS):.4f}")
    .replace("__PROBE_MIN_HIT_FRACTION__", f"{float(render_consts.WORLDSPACE_RC_MIN_HIT_FRACTION):.4f}")
    .replace("__PROBE_BACKFACE_SOFTNESS__", f"{float(render_consts.WORLDSPACE_RC_BACKFACE_SOFTNESS):.4f}")
    .replace("__PROBE_CASCADE_BLEND_EDGE_START__", f"{float(render_consts.WORLDSPACE_RC_CASCADE_BLEND_EDGE_START):.4f}")
    .replace("__PROBE_CASCADE_BLEND_EDGE_END__", f"{float(render_consts.WORLDSPACE_RC_CASCADE_BLEND_EDGE_END):.4f}")
    .replace("__PROBE_CASCADE_BLEND_MIN_WEIGHT__", f"{float(render_consts.WORLDSPACE_RC_CASCADE_BLEND_MIN_WEIGHT):.4f}")
    .replace("__PROBE_CASCADE_BLEND_CONFIDENCE_SCALE__", f"{float(render_consts.WORLDSPACE_RC_CASCADE_BLEND_CONFIDENCE_SCALE):.4f}")
    .replace("__PROBE_CASCADE_BLEND_FAR_BIAS__", f"{float(render_consts.WORLDSPACE_RC_CASCADE_BLEND_FAR_BIAS):.4f}")
    .replace("__PROBE_CAVE_MIN_LIGHT__", f"{float(render_consts.WORLDSPACE_RC_CAVE_MIN_LIGHT):.4f}")
    .replace("__PROBE_CAVE_SKY_POWER__", f"{float(render_consts.WORLDSPACE_RC_CAVE_SKY_POWER):.4f}")
    .replace("__PROBE_CAVE_DARKENING__", f"{float(render_consts.WORLDSPACE_RC_CAVE_DARKENING):.4f}")
    .replace("__SKY_HORIZON_R__", f"{float(render_consts.RADIANCE_CASCADES_SKY_HORIZON_RGB[0]):.4f}")
    .replace("__SKY_HORIZON_G__", f"{float(render_consts.RADIANCE_CASCADES_SKY_HORIZON_RGB[1]):.4f}")
    .replace("__SKY_HORIZON_B__", f"{float(render_consts.RADIANCE_CASCADES_SKY_HORIZON_RGB[2]):.4f}")
    .replace("__SKY_ZENITH_R__", f"{float(render_consts.RADIANCE_CASCADES_SKY_ZENITH_RGB[0]):.4f}")
    .replace("__SKY_ZENITH_G__", f"{float(render_consts.RADIANCE_CASCADES_SKY_ZENITH_RGB[1]):.4f}")
    .replace("__SKY_ZENITH_B__", f"{float(render_consts.RADIANCE_CASCADES_SKY_ZENITH_RGB[2]):.4f}")
    .replace("__SKY_GROUND_R__", f"{float(render_consts.RADIANCE_CASCADES_SKY_GROUND_RGB[0]):.4f}")
    .replace("__SKY_GROUND_G__", f"{float(render_consts.RADIANCE_CASCADES_SKY_GROUND_RGB[1]):.4f}")
    .replace("__SKY_GROUND_B__", f"{float(render_consts.RADIANCE_CASCADES_SKY_GROUND_RGB[2]):.4f}")
    .replace("__SKY_SUN_R__", f"{float(render_consts.RADIANCE_CASCADES_SKY_SUN_GLOW_RGB[0]):.4f}")
    .replace("__SKY_SUN_G__", f"{float(render_consts.RADIANCE_CASCADES_SKY_SUN_GLOW_RGB[1]):.4f}")
    .replace("__SKY_SUN_B__", f"{float(render_consts.RADIANCE_CASCADES_SKY_SUN_GLOW_RGB[2]):.4f}")
    .replace("__SKY_SUN_DIR_X__", f"{float(render_consts.LIGHT_DIRECTION[0]):.4f}")
    .replace("__SKY_SUN_DIR_Y__", f"{float(render_consts.LIGHT_DIRECTION[1]):.4f}")
    .replace("__SKY_SUN_DIR_Z__", f"{float(render_consts.LIGHT_DIRECTION[2]):.4f}")
)

GI_POSTPROCESS_SHADER = GI_COMPOSE_SHADER









FINAL_BLIT_SHADER = """
struct PresentParams {
    viewport_inv_px: vec2f,
    reserved: vec2f,
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> params: PresentParams;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0),
    )[vertex_index];
    out.position = vec4f(pos, 0.0, 1.0);
    out.uv = vec2f(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let _inv_px = params.viewport_inv_px;
    let color = textureSample(src_tex, src_sampler, input.uv).rgb;
    return vec4f(color, 1.0);
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
    world_scale_and_pad: vec4f,
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

fn emit_triangle_at(base: u32, a: vec3f, b: vec3f, c: vec3f, normal: vec3f, color_a: vec3f, color_b: vec3f, color_c: vec3f) {
    emit_vertex(a, normal, color_a, base + 0u);
    emit_vertex(b, normal, color_b, base + 1u);
    emit_vertex(c, normal, color_c, base + 2u);
}

fn emit_quad_at(base: u32, p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, normal: vec3f, color0: vec3f, color1: vec3f, color2: vec3f, color3: vec3f) {
    emit_triangle_at(base + 0u, p0, p1, p2, normal, color0, color1, color2);
    emit_triangle_at(base + 3u, p0, p2, p3, normal, color0, color2, color3);
}

fn solid_at(chunk_base: u32, sample_size: u32, plane: u32, local_x: u32, local_z: u32, sample_y: i32, height_limit: u32) -> bool {
    if (sample_y < 0 || sample_y >= i32(height_limit)) {
        return false;
    }
    let idx = chunk_base + u32(sample_y) * plane + local_z * sample_size + local_x;
    return blocks.values[idx] != 0u;
}

fn ambient_occlusion_factor(side1: bool, side2: bool, corner: bool) -> f32 {
    var occlusion = 0u;
    if (side1 && side2) {
        occlusion = 3u;
    } else {
        occlusion = select(0u, 1u, side1) + select(0u, 1u, side2) + select(0u, 1u, corner);
    }
    if (occlusion == 0u) {
        return 1.0;
    }
    if (occlusion == 1u) {
        return 0.82;
    }
    if (occlusion == 2u) {
        return 0.68;
    }
    return 0.54;
}

fn ao_y_plane(chunk_base: u32, sample_size: u32, plane: u32, local_x: u32, local_z: u32, sample_y: i32, dx: i32, dz: i32, height_limit: u32) -> f32 {
    let side1 = solid_at(chunk_base, sample_size, plane, u32(i32(local_x) + dx), local_z, sample_y, height_limit);
    let side2 = solid_at(chunk_base, sample_size, plane, local_x, u32(i32(local_z) + dz), sample_y, height_limit);
    let corner = solid_at(chunk_base, sample_size, plane, u32(i32(local_x) + dx), u32(i32(local_z) + dz), sample_y, height_limit);
    return ambient_occlusion_factor(side1, side2, corner);
}

fn ao_x_plane(chunk_base: u32, sample_size: u32, plane: u32, sample_x: u32, local_z: u32, y: i32, dy: i32, dz: i32, height_limit: u32) -> f32 {
    let side1 = solid_at(chunk_base, sample_size, plane, sample_x, local_z, y + dy, height_limit);
    let side2 = solid_at(chunk_base, sample_size, plane, sample_x, u32(i32(local_z) + dz), y, height_limit);
    let corner = solid_at(chunk_base, sample_size, plane, sample_x, u32(i32(local_z) + dz), y + dy, height_limit);
    return ambient_occlusion_factor(side1, side2, corner);
}

fn ao_z_plane(chunk_base: u32, sample_size: u32, plane: u32, local_x: u32, sample_z: u32, y: i32, dx: i32, dy: i32, height_limit: u32) -> f32 {
    let side1 = solid_at(chunk_base, sample_size, plane, u32(i32(local_x) + dx), sample_z, y, height_limit);
    let side2 = solid_at(chunk_base, sample_size, plane, local_x, sample_z, y + dy, height_limit);
    let corner = solid_at(chunk_base, sample_size, plane, u32(i32(local_x) + dx), sample_z, y + dy, height_limit);
    return ambient_occlusion_factor(side1, side2, corner);
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

    let block_scale = params.world_scale_and_pad.x;
    let x0 = origin_x + f32(local_x - 1u) * block_scale;
    let x1 = x0 + block_scale;
    let z0 = origin_z + f32(local_z - 1u) * block_scale;
    let z1 = z0 + block_scale;
    let y0 = f32(y) * block_scale;
    let y1 = y0 + block_scale;

    let material = materials.values[cell_index];
    let top = face_color(material, y, 1.0);
    let east = face_color(material, y, 0.80);
    let west = face_color(material, y, 0.64);
    let south = face_color(material, y, 0.72);
    let north = face_color(material, y, 0.60);
    let bottom = face_color(material, y, 0.50);
    let yi = i32(y);
    var face_base = base;

    if (y == height_limit - 1u || blocks.values[cell_index + plane] == 0u) {
        let ao0 = ao_y_plane(chunk_base, sample_size, plane, local_x, local_z, yi + 1, -1, -1, height_limit);
        let ao1 = ao_y_plane(chunk_base, sample_size, plane, local_x, local_z, yi + 1, 1, -1, height_limit);
        let ao2 = ao_y_plane(chunk_base, sample_size, plane, local_x, local_z, yi + 1, 1, 1, height_limit);
        let ao3 = ao_y_plane(chunk_base, sample_size, plane, local_x, local_z, yi + 1, -1, 1, height_limit);
        emit_quad_at(
            face_base,
            vec3f(x0, y1, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y1, z1),
            vec3f(x0, y1, z1),
            vec3f(0.0, 1.0, 0.0),
            top * ao0,
            top * ao1,
            top * ao2,
            top * ao3,
        );
        face_base = face_base + 6u;
    }
    if (y == 0u || blocks.values[cell_index - plane] == 0u) {
        let ao0 = ao_y_plane(chunk_base, sample_size, plane, local_x, local_z, yi - 1, -1, -1, height_limit);
        let ao1 = ao_y_plane(chunk_base, sample_size, plane, local_x, local_z, yi - 1, -1, 1, height_limit);
        let ao2 = ao_y_plane(chunk_base, sample_size, plane, local_x, local_z, yi - 1, 1, 1, height_limit);
        let ao3 = ao_y_plane(chunk_base, sample_size, plane, local_x, local_z, yi - 1, 1, -1, height_limit);
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z0),
            vec3f(x0, y0, z1),
            vec3f(x1, y0, z1),
            vec3f(x1, y0, z0),
            vec3f(0.0, -1.0, 0.0),
            bottom * ao0,
            bottom * ao1,
            bottom * ao2,
            bottom * ao3,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index + 1u] == 0u) {
        let ao0 = ao_x_plane(chunk_base, sample_size, plane, local_x + 1u, local_z, yi, -1, -1, height_limit);
        let ao1 = ao_x_plane(chunk_base, sample_size, plane, local_x + 1u, local_z, yi, 1, -1, height_limit);
        let ao2 = ao_x_plane(chunk_base, sample_size, plane, local_x + 1u, local_z, yi, 1, 1, height_limit);
        let ao3 = ao_x_plane(chunk_base, sample_size, plane, local_x + 1u, local_z, yi, -1, 1, height_limit);
        emit_quad_at(
            face_base,
            vec3f(x1, y0, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y1, z1),
            vec3f(x1, y0, z1),
            vec3f(1.0, 0.0, 0.0),
            east * ao0,
            east * ao1,
            east * ao2,
            east * ao3,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index - 1u] == 0u) {
        let ao0 = ao_x_plane(chunk_base, sample_size, plane, local_x - 1u, local_z, yi, -1, -1, height_limit);
        let ao1 = ao_x_plane(chunk_base, sample_size, plane, local_x - 1u, local_z, yi, -1, 1, height_limit);
        let ao2 = ao_x_plane(chunk_base, sample_size, plane, local_x - 1u, local_z, yi, 1, 1, height_limit);
        let ao3 = ao_x_plane(chunk_base, sample_size, plane, local_x - 1u, local_z, yi, 1, -1, height_limit);
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z0),
            vec3f(x0, y0, z1),
            vec3f(x0, y1, z1),
            vec3f(x0, y1, z0),
            vec3f(-1.0, 0.0, 0.0),
            west * ao0,
            west * ao1,
            west * ao2,
            west * ao3,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index + sample_size] == 0u) {
        let ao0 = ao_z_plane(chunk_base, sample_size, plane, local_x, local_z + 1u, yi, -1, -1, height_limit);
        let ao1 = ao_z_plane(chunk_base, sample_size, plane, local_x, local_z + 1u, yi, 1, -1, height_limit);
        let ao2 = ao_z_plane(chunk_base, sample_size, plane, local_x, local_z + 1u, yi, 1, 1, height_limit);
        let ao3 = ao_z_plane(chunk_base, sample_size, plane, local_x, local_z + 1u, yi, -1, 1, height_limit);
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z1),
            vec3f(x1, y0, z1),
            vec3f(x1, y1, z1),
            vec3f(x0, y1, z1),
            vec3f(0.0, 0.0, 1.0),
            south * ao0,
            south * ao1,
            south * ao2,
            south * ao3,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index - sample_size] == 0u) {
        let ao0 = ao_z_plane(chunk_base, sample_size, plane, local_x, local_z - 1u, yi, -1, -1, height_limit);
        let ao1 = ao_z_plane(chunk_base, sample_size, plane, local_x, local_z - 1u, yi, -1, 1, height_limit);
        let ao2 = ao_z_plane(chunk_base, sample_size, plane, local_x, local_z - 1u, yi, 1, 1, height_limit);
        let ao3 = ao_z_plane(chunk_base, sample_size, plane, local_x, local_z - 1u, yi, 1, -1, height_limit);
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z0),
            vec3f(x0, y1, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y0, z0),
            vec3f(0.0, 0.0, -1.0),
            north * ao0,
            north * ao1,
            north * ao2,
            north * ao3,
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
    let chunk_world_size = f32(params.counts_and_flags.w) * params.world_scale_and_pad.x;
    let origin_x = f32(origin.x) * chunk_world_size;
    let origin_z = f32(origin.y) * chunk_world_size;
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
