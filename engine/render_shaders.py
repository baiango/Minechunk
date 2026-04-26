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

    let upward_surface = smoothstep(0.0, 0.85, normal.y);

    // RC is a positive lighting field, but direct daylight should not be
    // injected as a hard per-probe floor. Use a smooth sky-visibility baseline
    // for open upward terrain, then add probe-traced indirect radiance on top.
    var rc_indirect = vec3f(0.0, 0.0, 0.0);
    var sky_signal_accum = 0.0;
    var sky_signal_weight = 0.0;
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
                    let open_probe_gate = smoothstep(0.06, 0.55, sky_access);
                    let open_surface_probe_gate = mix(1.0, open_probe_gate, upward_surface);
                    let structural_weight = tri_weight * backface * visibility * visibility * validity * validity * probe_alpha * open_surface_probe_gate;
                    let exposure_weight = tri_weight * max(0.16, backface) * visibility * validity * probe_alpha * open_surface_probe_gate;

                    cascade_accum = cascade_accum + probe.rgb * structural_weight;
                    cascade_weight_sum = cascade_weight_sum + structural_weight;
                    cascade_sky_access_sum = cascade_sky_access_sum + sky_access * exposure_weight;
                    cascade_sky_access_weight = cascade_sky_access_weight + exposure_weight;
                }
            }
        }

        if (cascade_weight_sum <= 0.0001) {
            continue;
        }
        let cascade_rgb_raw = max(cascade_accum / cascade_weight_sum, vec3f(0.0, 0.0, 0.0));
        let cascade_alpha = saturate(cascade_weight_sum * PROBE_CASCADE_BLEND_CONFIDENCE_SCALE);
        let cascade_sky_access = select(0.0, cascade_sky_access_sum / cascade_sky_access_weight, cascade_sky_access_weight > 0.0001);
        let edge_blend = cascade_edge_blend(uvw);
        var boundary_blend = 1.0;
        if (i + 1u < cascade_count) {
            boundary_blend = mix(PROBE_CASCADE_BLEND_MIN_WEIGHT, 1.0, edge_blend);
        }
        let cascade_preference = mix(1.0, PROBE_CASCADE_BLEND_FAR_BIAS, f32(i) / max(1.0, f32(cascade_count - 1u)));
        let confidence = clamp(cascade_alpha * boundary_blend * cascade_preference, 0.0, 1.0);

        // Probe radiance is indirect/additive only. Dark cascades add zero; they
        // never reduce light by participating in a normalized average.
        var cascade_add_weight = 1.0;
        if (i == 1u) {
            cascade_add_weight = 0.55;
        } else if (i == 2u) {
            cascade_add_weight = 0.30;
        } else if (i >= 3u) {
            cascade_add_weight = 0.18;
        }
        rc_indirect = rc_indirect + cascade_rgb_raw * confidence * cascade_add_weight;

        // Sky visibility is a smooth daylight mask, not visible probe radiance.
        // Weight C1 most strongly so the open-sky baseline is broad instead of
        // shaped like individual C0 cells. Hard-occluded caves keep this at zero.
        var sky_cascade_weight = 0.42;
        if (i == 1u) {
            sky_cascade_weight = 1.0;
        } else if (i == 2u) {
            sky_cascade_weight = 0.35;
        } else if (i >= 3u) {
            sky_cascade_weight = 0.15;
        }
        let sky_signal = smoothstep(0.18, 0.72, cascade_sky_access);
        sky_signal_accum = sky_signal_accum + sky_signal * confidence * sky_cascade_weight;
        sky_signal_weight = sky_signal_weight + confidence * sky_cascade_weight;
    }

    let smoothed_sky_signal = smoothstep(0.06, 0.84, sky_signal_accum / max(sky_signal_weight, 0.0001));
    let hemisphere = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    let rc_coverage = saturate(sky_signal_weight * 0.95);
    let medium_surface = smoothstep(24.0, 160.0, depth);
    let distant_surface = smoothstep(96.0, 384.0, depth);
    let terrain_daylight_floor = medium_surface * mix(0.06, 0.18, hemisphere) * (0.40 + 0.60 * (1.0 - smoothed_sky_signal));
    let distant_open_fallback = (1.0 - rc_coverage) * distant_surface * mix(0.14, 0.70, hemisphere);
    let open_sky = max(smoothed_sky_signal, max(terrain_daylight_floor, distant_open_fallback));
    let sky_baseline = vec3f(0.38, 0.40, 0.38) * open_sky * mix(0.34, 1.0, hemisphere);
    let rc_indirect_soft = rc_indirect / (vec3f(1.0, 1.0, 1.0) + rc_indirect);
    let rc_light = max(sky_baseline + rc_indirect_soft * 0.36, vec3f(0.0, 0.0, 0.0));

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




WORLDSPACE_RC_UPDATE_PARAMS_FLOATS = 24
WORLDSPACE_RC_UPDATE_PARAMS_BYTES = WORLDSPACE_RC_UPDATE_PARAMS_FLOATS * 4

WORLDSPACE_RC_TRACE_SHADER = """
struct RcUpdateParams {
    min_corner_and_extent: vec4f,
    meta0: vec4f,
    meta1: vec4f,
    light0: vec4f,
    light_dir: vec4f,
    controls: vec4f,
}

@group(0) @binding(0) var dst_volume: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1) var dst_visibility: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> rc: RcUpdateParams;

const AIR: u32 = 0u;
const BEDROCK: u32 = 1u;
const STONE: u32 = 2u;
const DIRT: u32 = 3u;
const GRASS: u32 = 4u;
const SAND: u32 = 5u;
const SNOW: u32 = 6u;

const TERRAIN_FREQUENCY_SCALE: f32 = 0.30000000;
const CAVE_FREQUENCY_SCALE: f32 = 0.50000000;
const SURFACE_BREACH_FREQUENCY_SCALE: f32 = 1.00000000;
const CAVE_BEDROCK_CLEARANCE: i32 = 3;
const GOLDEN_ANGLE: f32 = 2.39996322972865332;

const SKY_VISIBILITY_STEPS: u32 = __SKY_VISIBILITY_STEPS__u;
const SKY_VISIBILITY_STEP_BLOCKS: u32 = __SKY_VISIBILITY_STEP_BLOCKS__u;
const SKY_VISIBILITY_SIDE_WEIGHT: f32 = __SKY_VISIBILITY_SIDE_WEIGHT__;
const SKY_VISIBILITY_APERTURE_RADIUS_BLOCKS: i32 = __SKY_VISIBILITY_APERTURE_RADIUS_BLOCKS__;
const SKY_VISIBILITY_APERTURE_POWER: f32 = __SKY_VISIBILITY_APERTURE_POWER__;
const SKY_VISIBILITY_MIN_APERTURE: f32 = __SKY_VISIBILITY_MIN_APERTURE__;

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn seed_u32() -> u32 {
    return u32(max(rc.meta1.x, 0.0) + 0.5);
}

fn world_height_limit() -> u32 {
    return max(1u, u32(max(rc.meta1.y, 1.0) + 0.5));
}

fn trace_base_steps() -> i32 {
    return max(4, i32(rc.meta1.z + 0.5));
}

fn trace_base_directions() -> u32 {
    return max(6u, u32(rc.meta1.w + 0.5));
}

fn mix_u32(value: u32) -> u32 {
    var x = value;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn hash2(ix: i32, iy: i32, seed: u32) -> f32 {
    var h = bitcast<u32>(ix) * 0x9e3779b9u;
    h = h ^ (bitcast<u32>(iy) * 0x85ebca6bu);
    h = h ^ (seed * 0xc2b2ae35u);
    h = mix_u32(h);
    return f32(h & 0x00ffffffu) / 16777215.0;
}

fn hash3(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    var h = bitcast<u32>(ix) * 0x9e3779b9u;
    h = h ^ (bitcast<u32>(iy) * 0x85ebca6bu);
    h = h ^ (bitcast<u32>(iz) * 0xc2b2ae35u);
    h = h ^ (seed * 0x27d4eb2fu);
    h = mix_u32(h);
    return f32(h & 0x00ffffffu) / 16777215.0;
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn value_noise_2d(x: f32, y: f32, seed: u32, frequency: f32) -> f32 {
    let px = x * frequency;
    let py = y * frequency;
    let x0 = floor(px);
    let y0 = floor(py);
    let xf = px - x0;
    let yf = py - y0;

    let ix0 = i32(x0);
    let iy0 = i32(y0);
    let ix1 = ix0 + 1;
    let iy1 = iy0 + 1;

    let v00 = hash2(ix0, iy0, seed);
    let v10 = hash2(ix1, iy0, seed);
    let v01 = hash2(ix0, iy1, seed);
    let v11 = hash2(ix1, iy1, seed);

    let u = fade(xf);
    let v = fade(yf);
    let nx0 = lerp_f32(v00, v10, u);
    let nx1 = lerp_f32(v01, v11, u);
    return lerp_f32(nx0, nx1, v) * 2.0 - 1.0;
}

fn value_noise_3d(x: f32, y: f32, z: f32, seed: u32, frequency: f32) -> f32 {
    let px = x * frequency;
    let py = y * frequency;
    let pz = z * frequency;

    let x0 = floor(px);
    let y0 = floor(py);
    let z0 = floor(pz);
    let xf = px - x0;
    let yf = py - y0;
    let zf = pz - z0;

    let ix0 = i32(x0);
    let iy0 = i32(y0);
    let iz0 = i32(z0);
    let ix1 = ix0 + 1;
    let iy1 = iy0 + 1;
    let iz1 = iz0 + 1;

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let c000 = hash3(ix0, iy0, iz0, seed);
    let c100 = hash3(ix1, iy0, iz0, seed);
    let c010 = hash3(ix0, iy1, iz0, seed);
    let c110 = hash3(ix1, iy1, iz0, seed);
    let c001 = hash3(ix0, iy0, iz1, seed);
    let c101 = hash3(ix1, iy0, iz1, seed);
    let c011 = hash3(ix0, iy1, iz1, seed);
    let c111 = hash3(ix1, iy1, iz1, seed);

    let x00 = lerp_f32(c000, c100, u);
    let x10 = lerp_f32(c010, c110, u);
    let x01 = lerp_f32(c001, c101, u);
    let x11 = lerp_f32(c011, c111, u);
    let y0v = lerp_f32(x00, x10, v);
    let y1v = lerp_f32(x01, x11, v);
    return lerp_f32(y0v, y1v, w) * 2.0 - 1.0;
}

fn surface_profile_at(world_x: i32, world_z: i32) -> vec2u {
    let seed = seed_u32();
    let height_limit = world_height_limit();
    let sample_x = f32(world_x);
    let sample_z = f32(world_z);

    let broad = value_noise_2d(sample_x, sample_z, seed + 11u, 0.0009765625 * TERRAIN_FREQUENCY_SCALE);
    let ridge = value_noise_2d(sample_x, sample_z, seed + 23u, 0.00390625 * TERRAIN_FREQUENCY_SCALE);
    let detail = value_noise_2d(sample_x, sample_z, seed + 47u, 0.010416667 * TERRAIN_FREQUENCY_SCALE);
    let micro = value_noise_2d(sample_x, sample_z, seed + 71u, 0.020833334 * TERRAIN_FREQUENCY_SCALE);
    let nano = value_noise_2d(sample_x, sample_z, seed + 97u, 0.041666668 * TERRAIN_FREQUENCY_SCALE);

    let upper_bound = height_limit - 1u;
    let upper_bound_f = f32(upper_bound);
    let normalized_height = 24.0 + broad * 11.0 + ridge * 8.0 + detail * 4.5 + micro * 1.75 + nano * 0.75;
    let height_scale = select(1.0, upper_bound_f / 50.0, upper_bound > 0u);
    var height_f = normalized_height * height_scale;
    height_f = clamp(height_f, 4.0, upper_bound_f);
    let height_i = u32(height_f);

    let sand_threshold = max(4u, u32(f32(height_limit) * 0.18));
    let stone_threshold = max(sand_threshold + 6u, u32(f32(height_limit) * 0.58));
    let snow_threshold = max(stone_threshold + 6u, u32(f32(height_limit) * 0.82));

    var material = GRASS;
    if (height_i >= snow_threshold) {
        material = SNOW;
    } else if (height_i <= sand_threshold) {
        material = SAND;
    } else if (height_i >= stone_threshold && (detail + micro * 0.5 + nano * 0.35) > 0.10) {
        material = STONE;
    }
    return vec2u(height_i, material);
}

fn should_carve_cave(world_x: i32, world_y: i32, world_z: i32, surface_height: u32) -> bool {
    if (world_y <= CAVE_BEDROCK_CLEARANCE) {
        return false;
    }
    if (world_y >= i32(world_height_limit()) - 2) {
        return false;
    }

    let depth_below_surface = f32(i32(surface_height) - world_y);
    let normalized_y = f32(world_y) / f32(max(1u, world_height_limit() - 1u));
    var vertical_band = 1.0 - abs(normalized_y - 0.45) * 1.6;
    vertical_band = saturate(vertical_band);
    if (vertical_band <= 0.0) {
        return false;
    }

    let seed = seed_u32();
    let xf = f32(world_x);
    let yf = f32(world_y);
    let zf = f32(world_z);
    let cave_primary = value_noise_3d(xf, yf * 0.85, zf, seed + 101u, 0.018 * CAVE_FREQUENCY_SCALE);
    let cave_detail = value_noise_3d(xf, yf * 1.15, zf, seed + 149u, 0.041666668 * CAVE_FREQUENCY_SCALE);
    let cave_shape = value_noise_3d(xf, yf * 0.35, zf, seed + 173u, 0.009765625 * CAVE_FREQUENCY_SCALE);
    let density = cave_primary * 0.70 + cave_detail * 0.25 - cave_shape * 0.10;

    let depth_bonus = min(depth_below_surface * 0.004, 0.12);
    var shallow_bonus = 0.0;
    if (depth_below_surface <= 6.0) {
        shallow_bonus = (6.0 - depth_below_surface) * (0.12 / 6.0);
    }

    let threshold = 0.62 - vertical_band * 0.08 - depth_bonus - shallow_bonus;
    if (density > threshold) {
        return true;
    }

    if (depth_below_surface <= 2.0) {
        let breach_primary = value_noise_2d(xf, zf, seed + 211u, 0.020833334 * SURFACE_BREACH_FREQUENCY_SCALE);
        let breach_detail = value_noise_3d(xf, yf, zf, seed + 233u, 0.03125 * CAVE_FREQUENCY_SCALE);
        let breach_density = breach_primary * 0.65 + breach_detail * 0.35;
        let breach_threshold = 0.78 - vertical_band * 0.06;
        return breach_density > breach_threshold;
    }

    return false;
}

fn material_at_block(world_x: i32, world_y: i32, world_z: i32) -> u32 {
    if (world_y < 0) {
        return BEDROCK;
    }
    if (world_y >= i32(world_height_limit())) {
        return AIR;
    }

    let profile = surface_profile_at(world_x, world_z);
    let surface_height = profile.x;
    let surface_material = profile.y;
    if (world_y >= i32(surface_height)) {
        return AIR;
    }
    if (should_carve_cave(world_x, world_y, world_z, surface_height)) {
        return AIR;
    }
    if (world_y == 0) {
        return BEDROCK;
    }
    if (world_y < i32(surface_height) - 4) {
        return STONE;
    }
    if (world_y < i32(surface_height) - 1) {
        return DIRT;
    }
    return surface_material;
}

fn solid_at_block(world_x: i32, world_y: i32, world_z: i32) -> bool {
    return material_at_block(world_x, world_y, world_z) != AIR;
}

fn material_rgb(material: u32) -> vec3f {
    if (material == BEDROCK) {
        return vec3f(0.24, 0.22, 0.20);
    }
    if (material == STONE) {
        return vec3f(0.42, 0.40, 0.38);
    }
    if (material == DIRT) {
        return vec3f(0.47, 0.31, 0.18);
    }
    if (material == GRASS) {
        return vec3f(0.31, 0.68, 0.24);
    }
    if (material == SAND) {
        return vec3f(0.78, 0.71, 0.49);
    }
    if (material == SNOW) {
        return vec3f(0.95, 0.97, 0.98);
    }
    return vec3f(0.60, 0.80, 0.98);
}

fn estimate_hit_normal(bx: i32, by: i32, bz: i32, ray_dir: vec3f) -> vec3f {
    let sx0 = select(0.0, 1.0, solid_at_block(bx - 1, by, bz));
    let sx1 = select(0.0, 1.0, solid_at_block(bx + 1, by, bz));
    let sy0 = select(0.0, 1.0, solid_at_block(bx, by - 1, bz));
    let sy1 = select(0.0, 1.0, solid_at_block(bx, by + 1, bz));
    let sz0 = select(0.0, 1.0, solid_at_block(bx, by, bz - 1));
    let sz1 = select(0.0, 1.0, solid_at_block(bx, by, bz + 1));
    var n = vec3f(sx0 - sx1, sy0 - sy1, sz0 - sz1);
    let len_n = length(n);
    if (len_n <= 0.0001) {
        return normalize(-ray_dir);
    }
    return n / len_n;
}

fn column_sky_access(bx: i32, by: i32, bz: i32, sx: i32, sz: i32) -> f32 {
    let column_x = bx + sx;
    let column_z = bz + sz;
    let surface_height = i32(surface_profile_at(column_x, column_z).x);
    if (by >= surface_height - 1) {
        return 1.0;
    }

    // Hard occlusion: a sealed ceiling must contribute zero sky access.
    // The old code returned a fractional value for any open air above the probe,
    // even when the ray eventually hit a solid block before reaching the surface.
    // That made completely sealed caves glow from "partial" sky visibility.
    for (var step: u32 = 1u; step <= SKY_VISIBILITY_STEPS; step = step + 1u) {
        let sample_y = by + i32(step * SKY_VISIBILITY_STEP_BLOCKS);
        if (sample_y >= surface_height) {
            return 1.0;
        }
        if (solid_at_block(column_x, sample_y, column_z)) {
            return 0.0;
        }
    }

    // If the finite test did not prove a clear path to the terrain surface, keep
    // the probe closed. False-dark is better than false-sky for sealed caves.
    return 0.0;
}

fn sky_visibility_at_block(bx: i32, by: i32, bz: i32) -> f32 {
    if (by < 0) {
        return 0.0;
    }
    let center_access = column_sky_access(bx, by, bz, 0, 0);
    if (center_access <= 0.0) {
        return 0.0;
    }
    if (center_access >= 0.999 || SKY_VISIBILITY_SIDE_WEIGHT <= 0.0001) {
        return center_access;
    }

    let r = SKY_VISIBILITY_APERTURE_RADIUS_BLOCKS;
    var side_access =
        column_sky_access(bx, by, bz, r, 0) +
        column_sky_access(bx, by, bz, -r, 0) +
        column_sky_access(bx, by, bz, 0, r) +
        column_sky_access(bx, by, bz, 0, -r);
    var side_count = 4.0;
    if (SKY_VISIBILITY_SIDE_WEIGHT >= 0.5) {
        side_access = side_access +
            column_sky_access(bx, by, bz, r, r) +
            column_sky_access(bx, by, bz, r, -r) +
            column_sky_access(bx, by, bz, -r, r) +
            column_sky_access(bx, by, bz, -r, -r);
        side_count = 8.0;
    }
    side_access = side_access / side_count;
    let aperture = max(SKY_VISIBILITY_MIN_APERTURE, pow(max(side_access, 0.0), SKY_VISIBILITY_APERTURE_POWER));
    return center_access * aperture;
}

fn direction_for_base_index(index: u32, base_count: u32) -> vec3f {
    let y = 1.0 - (2.0 * (f32(index) + 0.5) / f32(max(1u, base_count)));
    let radius = sqrt(max(0.0, 1.0 - y * y));
    let theta = GOLDEN_ANGLE * f32(index);
    return vec3f(cos(theta) * radius, y, sin(theta) * radius);
}

fn effective_ray_count(cascade_index: u32, origin_sky_visibility: f32) -> u32 {
    let base_count = trace_base_directions();
    var ray_target_count = base_count;
    if (cascade_index == 1u) {
        ray_target_count = max(18u, (base_count * 7u + 7u) / 8u);
    } else if (cascade_index == 2u) {
        ray_target_count = max(12u, (base_count * 3u + 3u) / 4u);
    } else if (cascade_index >= 3u) {
        ray_target_count = max(8u, (base_count + 1u) / 2u);
    }

    if (origin_sky_visibility <= 0.04 && cascade_index >= 1u) {
        let min_dark_count = select(8u, 12u, cascade_index == 1u);
        ray_target_count = max(min_dark_count, (ray_target_count * 3u + 3u) / 4u);
    }
    return clamp(ray_target_count, 1u, base_count);
}

fn effective_step_count(cascade_index: u32, origin_sky_visibility: f32) -> u32 {
    let base_steps = trace_base_steps();
    var steps = base_steps;
    if (origin_sky_visibility <= 0.04) {
        if (cascade_index == 0u) {
            steps = max(8, base_steps - 6);
        } else if (cascade_index == 1u) {
            steps = max(6, base_steps - 8);
        } else if (cascade_index == 2u) {
            steps = max(4, base_steps - 10);
        } else {
            steps = max(4, base_steps - 11);
        }
    } else if (origin_sky_visibility <= 0.20) {
        if (cascade_index == 0u) {
            steps = max(10, base_steps - 4);
        } else if (cascade_index == 1u) {
            steps = max(7, base_steps - 7);
        } else if (cascade_index == 2u) {
            steps = max(5, base_steps - 9);
        } else {
            steps = max(4, base_steps - 10);
        }
    } else {
        if (cascade_index == 0u) {
            steps = base_steps;
        } else if (cascade_index == 1u) {
            steps = max(8, base_steps - 5);
        } else if (cascade_index == 2u) {
            steps = max(5, base_steps - 8);
        } else {
            steps = max(4, base_steps - 10);
        }
    }
    return u32(max(1, steps));
}

fn block_coord_from_world(world_pos: vec3f, block_size: f32) -> vec3i {
    return vec3i(
        i32(floor(world_pos.x / block_size)),
        i32(floor(world_pos.y / block_size)),
        i32(floor(world_pos.z / block_size)),
    );
}

@compute @workgroup_size(4, 4, 4)
fn trace_main(@builtin(global_invocation_id) gid: vec3u) {
    let resolution = max(1u, u32(rc.meta0.x + 0.5));
    if (gid.x >= resolution || gid.y >= resolution || gid.z >= resolution) {
        return;
    }

    let coord = vec3i(i32(gid.x), i32(gid.y), i32(gid.z));
    let min_corner = rc.min_corner_and_extent.xyz;
    let full_extent = max(rc.min_corner_and_extent.w, 0.0001);
    let max_distance = max(rc.meta0.z, 0.0001);
    let block_size = max(rc.meta0.w, 0.000001);
    let cascade_index = u32(rc.meta0.y + 0.5);
    let grid_den = f32(max(1u, resolution - 1u));
    let grid_pos = vec3f(f32(gid.x), f32(gid.y), f32(gid.z)) / grid_den;
    var origin = min_corner + grid_pos * full_extent;
    var origin_block = block_coord_from_world(origin, block_size);
    var origin_was_solid = false;

    if (solid_at_block(origin_block.x, origin_block.y, origin_block.z)) {
        origin_was_solid = true;
        let dirs = array<vec3i, 6>(
            vec3i(0, 1, 0),
            vec3i(1, 0, 0),
            vec3i(-1, 0, 0),
            vec3i(0, 0, 1),
            vec3i(0, 0, -1),
            vec3i(0, -1, 0),
        );
        let max_push = max(1u, u32(rc.controls.x + 0.5));
        var found = false;
        var found_block = origin_block;
        for (var radius: u32 = 1u; radius <= max_push; radius = radius + 1u) {
            if (found) {
                break;
            }
            for (var i: u32 = 0u; i < 6u; i = i + 1u) {
                let d = dirs[i];
                let offset = vec3i(d.x * i32(radius), d.y * i32(radius), d.z * i32(radius));
                let candidate = origin_block + offset;
                if (!solid_at_block(candidate.x, candidate.y, candidate.z)) {
                    found = true;
                    found_block = candidate;
                    break;
                }
            }
        }
        if (!found) {
            textureStore(dst_volume, coord, vec4f(0.0, 0.0, 0.0, 0.0));
            textureStore(dst_visibility, coord, vec4f(max_distance, max_distance * max_distance, 0.0, 0.0));
            return;
        }
        let offset_blocks = vec3f(
            f32(found_block.x - origin_block.x),
            f32(found_block.y - origin_block.y),
            f32(found_block.z - origin_block.z),
        );
        origin = origin + offset_blocks * block_size;
        origin_block = found_block;
    }

    let origin_sky_visibility = sky_visibility_at_block(origin_block.x, origin_block.y, origin_block.z);
    let ray_count = effective_ray_count(cascade_index, origin_sky_visibility);
    let base_ray_count = trace_base_directions();
    let step_count = effective_step_count(cascade_index, origin_sky_visibility);
    let step_size = max(block_size, max_distance / f32(max(1u, step_count)));
    let sun_dir = normalize(rc.light_dir.xyz);
    let direct_sun_strength = max(rc.light0.y, 0.0);
    let indirect_floor = max(rc.light0.z, 0.0);
    // Open probes should represent neutral daylight energy, not pure blue sky.
    // A saturated blue sky term made underfilled moving C0 probes tint nearby
    // terrain blue. Mix skylight with warm sun/ground bounce before storing it.
    let sky_rgb = vec3f(1.00, 0.98, 0.88) * max(rc.light0.x, 0.0);
    let cheap_hit_shading = cascade_index >= 1u;
    let far_hit_sky_visibility = saturate(origin_sky_visibility);
    let direct_sun_scale = select(1.10, 0.62, cheap_hit_shading);

    var accum = vec3f(0.0, 0.0, 0.0);
    var hit_count = 0.0;
    var sky_count = 0.0;
    var hit_sky_visibility_accum = 0.0;
    var distance_accum = 0.0;
    var distance_sq_accum = 0.0;

    for (var ray_i: u32 = 0u; ray_i < ray_count; ray_i = ray_i + 1u) {
        var base_index = ray_i;
        if (ray_count < base_ray_count) {
            base_index = min(base_ray_count - 1u, u32((f32(ray_i) + 0.5) * f32(base_ray_count) / f32(ray_count)));
        }
        let dir = direction_for_base_index(base_index, base_ray_count);
        var dist = step_size * 0.5;
        var hit = false;
        var hit_distance = max_distance;

        loop {
            if (dist > max_distance) {
                break;
            }
            let sample_pos = origin + dir * dist;
            let b = block_coord_from_world(sample_pos, block_size);
            let material = material_at_block(b.x, b.y, b.z);
            if (material != AIR) {
                let color = material_rgb(material);
                var facing = 1.0;
                var sky_visibility = far_hit_sky_visibility;
                var open_hemi = pow(max(0.0, -dir.y), 1.5);
                var sun_term = max(0.0, -dot(dir, sun_dir));
                if (!cheap_hit_shading) {
                    let n = estimate_hit_normal(b.x, b.y, b.z, dir);
                    facing = saturate(-dot(n, dir));
                    sun_term = max(0.0, dot(n, sun_dir));
                    sky_visibility = sky_visibility_at_block(b.x, b.y, b.z);
                    open_hemi = pow(max(0.0, n.y), 1.5);
                }

                let sky_open = saturate(sky_visibility);
                let cave_gate = pow(sky_open, 1.35);
                let falloff = 1.0 - min(1.0, dist / max(max_distance, 0.0001));
                let occluded_bounce = (1.0 - cave_gate) * (indirect_floor + 0.018) * (0.65 + 0.35 * facing) * (0.55 + 0.45 * falloff);
                let ambient_sky = mix(0.055, (0.10 + 0.90 * open_hemi) * 0.30, cave_gate);
                let direct_sun = cave_gate * sun_term * direct_sun_strength * direct_sun_scale * 0.55;
                let range_term = 0.30 + 0.70 * falloff;
                let facing_term = 0.40 + 0.60 * facing;
                let bounce_scale = (indirect_floor + occluded_bounce + ambient_sky + direct_sun) * range_term * facing_term;
                accum = accum + color * bounce_scale;
                hit_sky_visibility_accum = hit_sky_visibility_accum + sky_visibility * (0.35 + 0.65 * facing);
                hit_count = hit_count + 1.0;
                hit = true;
                hit_distance = dist;
                break;
            }
            dist = dist + step_size;
        }

        if (!hit) {
            let sky_axis = max(0.0, dir.y);
            let sky_ray_weight = (0.04 + 0.96 * (sky_axis * sky_axis)) * pow(max(0.0, origin_sky_visibility), 1.40);
            // Keep sky misses mostly as a visibility signal, but allow a broader,
            // softer contribution so open terrain and cave mouths do not fall off
            // into black while avoiding the harsh cave-mouth hotspots.
            let sky_fill = mix(0.08, 0.13, saturate(origin_sky_visibility));
            accum = accum + sky_rgb * sky_ray_weight * sky_fill;
            sky_count = sky_count + sky_ray_weight;
        }

        distance_accum = distance_accum + hit_distance;
        distance_sq_accum = distance_sq_accum + hit_distance * hit_distance;
    }

    let inv_ray_count = 1.0 / f32(max(1u, ray_count));
    // A probe whose initial grid point was inside terrain is not necessarily an
    // invalid/leaky probe. Near open terrain, C0 grid vertices often begin just
    // under the surface and are pushed upward into real sky-visible air. Dimming
    // all pushed probes made newly-entered nearby chunks pulse dark. Keep pushed
    // probes dark only when the pushed position is still sky-occluded.
    let pushed_sky_gate = smoothstep(0.18, 0.72, origin_sky_visibility);
    let pushed_radiance_scale = mix(clamp(rc.controls.y, 0.0, 1.0), 1.0, pushed_sky_gate);
    let solid_radiance_scale = select(1.0, pushed_radiance_scale, origin_was_solid);
    let rgb = accum * inv_ray_count * solid_radiance_scale;
    let hit_fraction = hit_count * inv_ray_count;
    let sky_fraction = min(1.0, sky_count * inv_ray_count);
    let valid_fraction = saturate(hit_fraction + 0.75 * sky_fraction) * solid_radiance_scale;
    var observed_sky_access = 0.0;
    if (hit_count > 0.0) {
        observed_sky_access = hit_sky_visibility_accum / hit_count;
    }
    observed_sky_access = max(observed_sky_access, sky_fraction);
    let probe_sky_access = saturate(0.78 * origin_sky_visibility + 0.22 * observed_sky_access);
    let mean_distance = distance_accum * inv_ray_count;
    let mean_distance_sq = distance_sq_accum * inv_ray_count;

    textureStore(dst_volume, coord, vec4f(rgb, valid_fraction));
    textureStore(dst_visibility, coord, vec4f(mean_distance, mean_distance_sq, valid_fraction, probe_sky_access));
}
"""

WORLDSPACE_RC_FILTER_SHADER = """
struct RcUpdateParams {
    min_corner_and_extent: vec4f,
    meta0: vec4f,
    meta1: vec4f,
    light0: vec4f,
    light_dir: vec4f,
    controls: vec4f,
}

@group(0) @binding(0) var src_volume: texture_3d<f32>;
@group(0) @binding(1) var src_visibility: texture_3d<f32>;
@group(0) @binding(2) var dst_volume: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var dst_visibility: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<uniform> rc: RcUpdateParams;

fn in_bounds(c: vec3i, resolution: i32) -> bool {
    return c.x >= 0 && c.y >= 0 && c.z >= 0 && c.x < resolution && c.y < resolution && c.z < resolution;
}

@compute @workgroup_size(4, 4, 4)
fn filter_main(@builtin(global_invocation_id) gid: vec3u) {
    let resolution_u = max(1u, u32(rc.meta0.x + 0.5));
    if (gid.x >= resolution_u || gid.y >= resolution_u || gid.z >= resolution_u) {
        return;
    }

    let resolution = i32(resolution_u);
    let coord = vec3i(i32(gid.x), i32(gid.y), i32(gid.z));
    let cascade_index = u32(rc.meta0.y + 0.5);
    let far_filter = clamp(f32(cascade_index) * 0.45, 0.0, 1.0);

    let raw_volume = textureLoad(src_volume, coord, 0);
    let raw_visibility = textureLoad(src_visibility, coord, 0);
    let center_weight = mix(0.34, 0.28, far_filter);
    var filtered_volume = raw_volume * center_weight;
    var filtered_visibility = raw_visibility * center_weight;
    var weight = center_weight;

    // C0 keeps a tight 6-neighbor filter. Far cascades use a 3x3x3 bilateral
    // filter so they behave like broad irradiance fields instead of visible
    // probe-grid shadow squares. The sky-access bilateral term still blocks
    // open-sky probes from bleeding into sealed caves.
    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                let manhattan = abs(dx) + abs(dy) + abs(dz);
                var geometric_weight = 0.0;
                if (manhattan == 1) {
                    geometric_weight = mix(0.160, 0.135, far_filter);
                } else if (manhattan == 2) {
                    geometric_weight = mix(0.050, 0.070, far_filter);
                } else {
                    geometric_weight = mix(0.030, 0.045, far_filter);
                }
                if (geometric_weight <= 0.0001) {
                    continue;
                }

                let nc = coord + vec3i(dx, dy, dz);
                if (in_bounds(nc, resolution)) {
                    let neighbor_volume = textureLoad(src_volume, nc, 0);
                    let neighbor_visibility = textureLoad(src_visibility, nc, 0);

                    let sky_delta = abs(clamp(neighbor_visibility.w, 0.0, 1.0) - clamp(raw_visibility.w, 0.0, 1.0));
                    let sky_similarity = clamp(1.0 - smoothstep(mix(0.08, 0.12, far_filter), mix(0.55, 0.82, far_filter), sky_delta), 0.0, 1.0);
                    let neighbor_validity = clamp(max(neighbor_volume.a, neighbor_visibility.z), 0.0, 1.0);
                    let neighbor_weight = geometric_weight * sky_similarity * (0.25 + 0.75 * neighbor_validity);

                    filtered_volume = filtered_volume + neighbor_volume * neighbor_weight;
                    filtered_visibility = filtered_visibility + neighbor_visibility * neighbor_weight;
                    weight = weight + neighbor_weight;
                }
            }
        }
    }

    filtered_volume = filtered_volume / max(weight, 0.0001);
    filtered_visibility = filtered_visibility / max(weight, 0.0001);

    let sky_filter_power = max(0.25, rc.light0.w);
    let sky_filter_gate = pow(clamp(raw_visibility.w, 0.0, 1.0), sky_filter_power);
    var out_volume = raw_volume * (1.0 - sky_filter_gate) + filtered_volume * sky_filter_gate;
    var out_visibility = filtered_visibility;

    // Do not add a colored radiance floor here. Probe filtering should preserve
    // measured radiance; the compose pass handles open-surface confidence without
    // tinting the stored RC field blue.

    // Sky access is an occlusion classification, not radiance. Still, C0 cannot
    // stay completely raw or individual open probes become visible light blobs.
    // Use bilateral-filtered sky visibility for the daylight baseline while the
    // sky-similarity term above prevents open probes from flooding sealed caves.
    let sky_visibility_blend = mix(0.62, 0.86, far_filter);
    out_visibility.w = mix(raw_visibility.w, clamp(filtered_visibility.w, 0.0, 1.0), sky_visibility_blend);
    out_volume = vec4f(max(out_volume.rgb, vec3f(0.0, 0.0, 0.0)), clamp(out_volume.a, 0.0, 1.0));
    out_visibility = max(out_visibility, vec4f(0.0, 0.0, 0.0, 0.0));

    textureStore(dst_volume, coord, out_volume);
    textureStore(dst_visibility, coord, out_visibility);
}
"""

WORLDSPACE_RC_TRACE_SHADER = (
    WORLDSPACE_RC_TRACE_SHADER
    .replace("__SKY_VISIBILITY_STEPS__", str(max(1, int(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_STEPS))))
    .replace("__SKY_VISIBILITY_STEP_BLOCKS__", str(max(1, int(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_STEP_BLOCKS))))
    .replace("__SKY_VISIBILITY_SIDE_WEIGHT__", f"{float(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_SIDE_WEIGHT):.8f}")
    .replace("__SKY_VISIBILITY_APERTURE_RADIUS_BLOCKS__", str(max(1, int(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_APERTURE_RADIUS_BLOCKS))))
    .replace("__SKY_VISIBILITY_APERTURE_POWER__", f"{float(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_APERTURE_POWER):.8f}")
    .replace("__SKY_VISIBILITY_MIN_APERTURE__", f"{float(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_MIN_APERTURE):.8f}")
)








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
    local_height: u32,
    chunk_count: u32,
    chunk_size: u32,
    world_height: u32,
    seed: u32,
    _pad0: u32,
    _pad1: u32,
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

struct ChunkCoordBuffer {
    values: array<vec4i>,
}

@group(0) @binding(0) var<storage, read> surface_heights: HeightBuffer;
@group(0) @binding(1) var<storage, read> surface_materials: MaterialBuffer;
@group(0) @binding(2) var<storage, read_write> blocks: BlockBuffer;
@group(0) @binding(3) var<storage, read_write> voxel_materials: VoxelMaterialBuffer;
@group(0) @binding(4) var<uniform> params: ExpandParams;
@group(0) @binding(5) var<storage, read> chunk_coords: ChunkCoordBuffer;

fn mix_u32(value: u32) -> u32 {
    var x = value;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn hash2(ix: i32, iy: i32, seed: u32) -> f32 {
    var h = bitcast<u32>(ix) * 0x9e3779b9u;
    h = h ^ (bitcast<u32>(iy) * 0x85ebca6bu);
    h = h ^ (seed * 0xc2b2ae35u);
    h = mix_u32(h);
    return f32(h & 0x00ffffffu) / 16777215.0;
}

fn hash3(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    var h = bitcast<u32>(ix) * 0x9e3779b9u;
    h = h ^ (bitcast<u32>(iy) * 0x85ebca6bu);
    h = h ^ (bitcast<u32>(iz) * 0xc2b2ae35u);
    h = h ^ (seed * 0x27d4eb2fu);
    h = mix_u32(h);
    return f32(h & 0x00ffffffu) / 16777215.0;
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn value_noise_2d(x: f32, y: f32, seed: u32, frequency: f32) -> f32 {
    let px = x * frequency;
    let py = y * frequency;
    let x0 = floor(px);
    let y0 = floor(py);
    let xf = px - x0;
    let yf = py - y0;

    let ix0 = i32(x0);
    let iy0 = i32(y0);
    let ix1 = ix0 + 1;
    let iy1 = iy0 + 1;

    let v00 = hash2(ix0, iy0, seed);
    let v10 = hash2(ix1, iy0, seed);
    let v01 = hash2(ix0, iy1, seed);
    let v11 = hash2(ix1, iy1, seed);

    let u = fade(xf);
    let v = fade(yf);
    let nx0 = lerp(v00, v10, u);
    let nx1 = lerp(v01, v11, u);
    return lerp(nx0, nx1, v) * 2.0 - 1.0;
}

fn value_noise_3d(x: f32, y: f32, z: f32, seed: u32, frequency: f32) -> f32 {
    let px = x * frequency;
    let py = y * frequency;
    let pz = z * frequency;

    let x0 = floor(px);
    let y0 = floor(py);
    let z0 = floor(pz);
    let xf = px - x0;
    let yf = py - y0;
    let zf = pz - z0;

    let ix0 = i32(x0);
    let iy0 = i32(y0);
    let iz0 = i32(z0);
    let ix1 = ix0 + 1;
    let iy1 = iy0 + 1;
    let iz1 = iz0 + 1;

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let c000 = hash3(ix0, iy0, iz0, seed);
    let c100 = hash3(ix1, iy0, iz0, seed);
    let c010 = hash3(ix0, iy1, iz0, seed);
    let c110 = hash3(ix1, iy1, iz0, seed);
    let c001 = hash3(ix0, iy0, iz1, seed);
    let c101 = hash3(ix1, iy0, iz1, seed);
    let c011 = hash3(ix0, iy1, iz1, seed);
    let c111 = hash3(ix1, iy1, iz1, seed);

    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);
    let y0v = lerp(x00, x10, v);
    let y1v = lerp(x01, x11, v);
    return lerp(y0v, y1v, w) * 2.0 - 1.0;
}

fn clamp01(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn should_carve_cave(world_x: i32, world_y: i32, world_z: i32, surface_height: i32, seed: u32, world_height_limit: u32) -> bool {
    if (world_y <= 3) {
        return false;
    }
    if (world_y >= i32(world_height_limit) - 2) {
        return false;
    }

    let depth_below_surface = surface_height - world_y;
    let normalized_y = f32(world_y) / f32(max(1u, world_height_limit - 1u));
    let vertical_band = clamp01(1.0 - abs(normalized_y - 0.45) * 1.6);
    if (vertical_band <= 0.0) {
        return false;
    }

    let xf = f32(world_x);
    let yf = f32(world_y);
    let zf = f32(world_z);
    let cave_frequency_scale = 0.5;
    let cave_primary = value_noise_3d(xf, yf * 0.85, zf, seed + 101u, 0.018 * cave_frequency_scale);
    let cave_detail = value_noise_3d(xf, yf * 1.15, zf, seed + 149u, 0.041666668 * cave_frequency_scale);
    let cave_shape = value_noise_3d(xf, yf * 0.35, zf, seed + 173u, 0.009765625 * cave_frequency_scale);
    let density = cave_primary * 0.70 + cave_detail * 0.25 - cave_shape * 0.10;

    var depth_bonus = f32(depth_below_surface) * 0.004;
    if (depth_bonus > 0.12) {
        depth_bonus = 0.12;
    }

    var shallow_bonus = 0.0;
    if (depth_below_surface <= 6) {
        shallow_bonus = (6.0 - f32(depth_below_surface)) * (0.12 / 6.0);
    }

    let threshold = 0.62 - vertical_band * 0.08 - depth_bonus - shallow_bonus;
    if (density > threshold) {
        return true;
    }

    if (depth_below_surface <= 2) {
        let breach_primary = value_noise_2d(xf, zf, seed + 211u, 0.020833334);
        let breach_detail = value_noise_3d(xf, yf, zf, seed + 233u, 0.03125 * cave_frequency_scale);
        let breach_density = breach_primary * 0.65 + breach_detail * 0.35;
        let breach_threshold = 0.78 - vertical_band * 0.06;
        return breach_density > breach_threshold;
    }

    return false;
}

fn terrain_material_from_surface_profile(world_x: i32, world_y: i32, world_z: i32, surface_height: i32, surface_material: u32, seed: u32, world_height_limit: u32) -> u32 {
    if (world_y < 0 || world_y >= i32(world_height_limit)) {
        return 0u;
    }
    if (world_y >= surface_height) {
        return 0u;
    }
    if (should_carve_cave(world_x, world_y, world_z, surface_height, seed, world_height_limit)) {
        return 0u;
    }
    if (world_y == 0) {
        return 1u;
    }
    if (world_y < surface_height - 4) {
        return 2u;
    }
    if (world_y < surface_height - 1) {
        return 3u;
    }
    return surface_material;
}

@compute @workgroup_size(8, 8, 1)
fn expand_main(@builtin(global_invocation_id) gid: vec3u) {
    let sample_size = params.sample_size;
    let local_height = params.local_height;
    let chunk_count = params.chunk_count;
    let storage_height = local_height + 2u;
    let plane = sample_size * sample_size;
    let chunk_stride = storage_height * plane;
    if (gid.x >= sample_size || gid.y >= sample_size || gid.z >= chunk_count * storage_height) {
        return;
    }

    let chunk_index = gid.z / storage_height;
    let sample_y = gid.z - chunk_index * storage_height;
    let coord = chunk_coords.values[chunk_index];
    let world_y = coord.y * i32(params.chunk_size) + i32(sample_y) - 1;
    let world_x = coord.x * i32(params.chunk_size) - 1 + i32(gid.x);
    let world_z = coord.z * i32(params.chunk_size) - 1 + i32(gid.y);
    let cell_index = gid.y * sample_size + gid.x;
    let surface_index = chunk_index * plane + cell_index;
    let voxel_index = chunk_index * chunk_stride + sample_y * plane + cell_index;
    let surface_height = i32(surface_heights.values[surface_index]);
    let surface_material = surface_materials.values[surface_index];
    let material = terrain_material_from_surface_profile(
        world_x,
        world_y,
        world_z,
        surface_height,
        surface_material,
        params.seed,
        params.world_height,
    );

    blocks.values[voxel_index] = select(0u, 1u, material != 0u);
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
    values: array<vec4i>,
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
        let altitude = clamp((f32(height) - 360.0) / 1280.0, 0.0, 1.0);
        return mix(vec3f(0.18, 0.53, 0.18), vec3f(0.31, 0.68, 0.24), altitude);
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

fn solid_at_sample(chunk_base: u32, sample_size: u32, plane: u32, storage_height: u32, sample_x: u32, sample_z: u32, sample_y: i32) -> bool {
    if (sample_y < 0 || sample_y >= i32(storage_height)) {
        return false;
    }
    let idx = chunk_base + u32(sample_y) * plane + sample_z * sample_size + sample_x;
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

fn ao_y_plane(chunk_base: u32, sample_size: u32, plane: u32, storage_height: u32, local_x: u32, local_z: u32, sample_y: i32, dx: i32, dz: i32) -> f32 {
    let side1 = solid_at_sample(chunk_base, sample_size, plane, storage_height, u32(i32(local_x) + dx), local_z, sample_y);
    let side2 = solid_at_sample(chunk_base, sample_size, plane, storage_height, local_x, u32(i32(local_z) + dz), sample_y);
    let corner = solid_at_sample(chunk_base, sample_size, plane, storage_height, u32(i32(local_x) + dx), u32(i32(local_z) + dz), sample_y);
    return ambient_occlusion_factor(side1, side2, corner);
}

fn ao_x_plane(chunk_base: u32, sample_size: u32, plane: u32, storage_height: u32, sample_x: u32, local_z: u32, sample_y: i32, dy: i32, dz: i32) -> f32 {
    let side1 = solid_at_sample(chunk_base, sample_size, plane, storage_height, sample_x, local_z, sample_y + dy);
    let side2 = solid_at_sample(chunk_base, sample_size, plane, storage_height, sample_x, u32(i32(local_z) + dz), sample_y);
    let corner = solid_at_sample(chunk_base, sample_size, plane, storage_height, sample_x, u32(i32(local_z) + dz), sample_y + dy);
    return ambient_occlusion_factor(side1, side2, corner);
}

fn ao_z_plane(chunk_base: u32, sample_size: u32, plane: u32, storage_height: u32, local_x: u32, sample_z: u32, sample_y: i32, dx: i32, dy: i32) -> f32 {
    let side1 = solid_at_sample(chunk_base, sample_size, plane, storage_height, u32(i32(local_x) + dx), sample_z, sample_y);
    let side2 = solid_at_sample(chunk_base, sample_size, plane, storage_height, local_x, sample_z, sample_y + dy);
    let corner = solid_at_sample(chunk_base, sample_size, plane, storage_height, u32(i32(local_x) + dx), sample_z, sample_y + dy);
    return ambient_occlusion_factor(side1, side2, corner);
}

fn voxel_face_count(
    chunk_base: u32,
    sample_size: u32,
    plane: u32,
    storage_height: u32,
    local_x: u32,
    local_z: u32,
    y: u32,
) -> u32 {
    let sample_y = y + 1u;
    let cell_index = chunk_base + sample_y * plane + local_z * sample_size + local_x;
    if (blocks.values[cell_index] == 0u) {
        return 0u;
    }
    var count = 0u;
    if (blocks.values[cell_index + plane] == 0u) {
        count = count + 6u;
    }
    if (blocks.values[cell_index - plane] == 0u) {
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
    storage_height: u32,
    local_x: u32,
    local_z: u32,
    y: u32,
    origin: vec4i,
) {
    let sample_y = y + 1u;
    let cell_index = chunk_base + sample_y * plane + local_z * sample_size + local_x;
    if (blocks.values[cell_index] == 0u) {
        return;
    }

    let block_scale = params.world_scale_and_pad.x;
    let chunk_size = i32(params.counts_and_flags.w);
    let chunk_world_size = f32(params.counts_and_flags.w) * block_scale;
    let origin_x = f32(origin.x) * chunk_world_size;
    let origin_z = f32(origin.z) * chunk_world_size;
    let world_y0_i = origin.y * chunk_size + i32(y);
    let world_height = u32(max(0, world_y0_i));
    let x0 = origin_x + f32(local_x - 1u) * block_scale;
    let x1 = x0 + block_scale;
    let z0 = origin_z + f32(local_z - 1u) * block_scale;
    let z1 = z0 + block_scale;
    let y0 = f32(world_y0_i) * block_scale;
    let y1 = y0 + block_scale;

    let material = materials.values[cell_index];
    let top = face_color(material, world_height, 1.0);
    let east = face_color(material, world_height, 0.80);
    let west = face_color(material, world_height, 0.64);
    let south = face_color(material, world_height, 0.72);
    let north = face_color(material, world_height, 0.60);
    let bottom = face_color(material, world_height, 0.50);
    let syi = i32(sample_y);
    var face_base = base;

    if (blocks.values[cell_index + plane] == 0u) {
        let ao0 = ao_y_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z, syi + 1, -1, -1);
        let ao1 = ao_y_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z, syi + 1, 1, -1);
        let ao2 = ao_y_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z, syi + 1, 1, 1);
        let ao3 = ao_y_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z, syi + 1, -1, 1);
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
    if (blocks.values[cell_index - plane] == 0u) {
        let ao0 = ao_y_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z, syi - 1, -1, -1);
        let ao1 = ao_y_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z, syi - 1, -1, 1);
        let ao2 = ao_y_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z, syi - 1, 1, 1);
        let ao3 = ao_y_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z, syi - 1, 1, -1);
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
        let ao0 = ao_x_plane(chunk_base, sample_size, plane, storage_height, local_x + 1u, local_z, syi, -1, -1);
        let ao1 = ao_x_plane(chunk_base, sample_size, plane, storage_height, local_x + 1u, local_z, syi, 1, -1);
        let ao2 = ao_x_plane(chunk_base, sample_size, plane, storage_height, local_x + 1u, local_z, syi, 1, 1);
        let ao3 = ao_x_plane(chunk_base, sample_size, plane, storage_height, local_x + 1u, local_z, syi, -1, 1);
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
        let ao0 = ao_x_plane(chunk_base, sample_size, plane, storage_height, local_x - 1u, local_z, syi, -1, -1);
        let ao1 = ao_x_plane(chunk_base, sample_size, plane, storage_height, local_x - 1u, local_z, syi, -1, 1);
        let ao2 = ao_x_plane(chunk_base, sample_size, plane, storage_height, local_x - 1u, local_z, syi, 1, 1);
        let ao3 = ao_x_plane(chunk_base, sample_size, plane, storage_height, local_x - 1u, local_z, syi, 1, -1);
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
        let ao0 = ao_z_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z + 1u, syi, -1, -1);
        let ao1 = ao_z_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z + 1u, syi, 1, -1);
        let ao2 = ao_z_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z + 1u, syi, 1, 1);
        let ao3 = ao_z_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z + 1u, syi, -1, 1);
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
        let ao0 = ao_z_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z - 1u, syi, -1, -1);
        let ao1 = ao_z_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z - 1u, syi, -1, 1);
        let ao2 = ao_z_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z - 1u, syi, 1, 1);
        let ao3 = ao_z_plane(chunk_base, sample_size, plane, storage_height, local_x, local_z - 1u, syi, 1, -1);
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
    let storage_height = height_limit + 2u;
    let chunk_stride = storage_height * plane;
    let chunk_base = wid.z * chunk_stride;
    let local_x = wid.x + 1u;
    let local_z = wid.y + 1u;
    face_counts[y] = voxel_face_count(chunk_base, sample_size, plane, storage_height, local_x, local_z, y);
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
    let storage_height = height_limit + 2u;
    let chunk_stride = storage_height * plane;
    let chunk_index = wid.z;
    let chunk_base = chunk_index * chunk_stride;
    let local_x = wid.x + 1u;
    let local_z = wid.y + 1u;
    face_counts[y] = voxel_face_count(chunk_base, sample_size, plane, storage_height, local_x, local_z, y);
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
    emit_voxel_faces(column_base + prefix_offsets[y], chunk_base, sample_size, plane, storage_height, local_x, local_z, y, origin);
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
