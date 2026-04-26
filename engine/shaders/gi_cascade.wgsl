
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
