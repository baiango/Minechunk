
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
const PI: f32 = 3.14159265358979323846;
const RC_BASIS_GOLDEN_ANGLE: f32 = 2.39996322972865332;
const RC_DIRECTION_COUNT: i32 = 16;

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

fn screen_gbuffer_sky_at_offset(uv: vec2f, offset_px: vec2f, inv_dims: vec2f, dims_u: vec2u) -> f32 {
    let sample_uv = clamp(uv + offset_px * inv_dims, vec2f(0.0, 0.0), vec2f(0.9999, 0.9999));
    let dims_i = vec2i(dims_u);
    let xy = clamp(vec2i(sample_uv * vec2f(dims_u)), vec2i(0, 0), dims_i - vec2i(1, 1));
    let g = textureLoad(gbuffer_tex, xy, 0);
    let has_surface = g.w > 0.0 && length(g.xyz) > 0.001;
    return select(1.0, 0.0, has_surface);
}

fn screen_space_sky_exposure(uv: vec2f) -> f32 {
    let dims_u = textureDimensions(gbuffer_tex, 0u);
    let inv_dims = 1.0 / max(vec2f(dims_u), vec2f(1.0, 1.0));
    var weighted_sum = 0.0;
    var weight_total = 0.0;
    var support_count = 0.0;
    var strong = 0.0;

    // v12: the earlier version used a pure max() across many long-range taps.
    // A tiny on-screen sky hole could therefore light a broad area and looked
    // like ghosting, but it disappeared as soon as the hole moved off screen.
    // Require multi-tap support instead of letting one small visible hole win.
    let s0 = screen_gbuffer_sky_at_offset(uv, vec2f(0.0, -6.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s0 * 1.00; weight_total = weight_total + 1.00; support_count = support_count + select(0.0, 1.0, s0 > 0.001); strong = max(strong, s0 * 1.00);
    let s1 = screen_gbuffer_sky_at_offset(uv, vec2f(0.0, -16.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s1 * 0.95; weight_total = weight_total + 0.95; support_count = support_count + select(0.0, 1.0, s1 > 0.001); strong = max(strong, s1 * 0.95);
    let s2 = screen_gbuffer_sky_at_offset(uv, vec2f(-18.0, -18.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s2 * 0.82; weight_total = weight_total + 0.82; support_count = support_count + select(0.0, 1.0, s2 > 0.001); strong = max(strong, s2 * 0.82);
    let s3 = screen_gbuffer_sky_at_offset(uv, vec2f(18.0, -18.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s3 * 0.82; weight_total = weight_total + 0.82; support_count = support_count + select(0.0, 1.0, s3 > 0.001); strong = max(strong, s3 * 0.82);
    let s4 = screen_gbuffer_sky_at_offset(uv, vec2f(0.0, -36.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s4 * 0.72; weight_total = weight_total + 0.72; support_count = support_count + select(0.0, 1.0, s4 > 0.001); strong = max(strong, s4 * 0.72);
    let s5 = screen_gbuffer_sky_at_offset(uv, vec2f(-42.0, -42.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s5 * 0.58; weight_total = weight_total + 0.58; support_count = support_count + select(0.0, 1.0, s5 > 0.001); strong = max(strong, s5 * 0.58);
    let s6 = screen_gbuffer_sky_at_offset(uv, vec2f(42.0, -42.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s6 * 0.58; weight_total = weight_total + 0.58; support_count = support_count + select(0.0, 1.0, s6 > 0.001); strong = max(strong, s6 * 0.58);
    let s7 = screen_gbuffer_sky_at_offset(uv, vec2f(0.0, -72.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s7 * 0.42; weight_total = weight_total + 0.42; support_count = support_count + select(0.0, 1.0, s7 > 0.001); strong = max(strong, s7 * 0.42);
    let s8 = screen_gbuffer_sky_at_offset(uv, vec2f(0.0, -128.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s8 * 0.28; weight_total = weight_total + 0.28; support_count = support_count + select(0.0, 1.0, s8 > 0.001); strong = max(strong, s8 * 0.28);
    let s9 = screen_gbuffer_sky_at_offset(uv, vec2f(-24.0, 0.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s9 * 0.18; weight_total = weight_total + 0.18; support_count = support_count + select(0.0, 1.0, s9 > 0.001); strong = max(strong, s9 * 0.18);
    let s10 = screen_gbuffer_sky_at_offset(uv, vec2f(24.0, 0.0), inv_dims, dims_u);
    weighted_sum = weighted_sum + s10 * 0.18; weight_total = weight_total + 0.18; support_count = support_count + select(0.0, 1.0, s10 > 0.001); strong = max(strong, s10 * 0.18);

    let avg_exposure = weighted_sum / max(weight_total, 0.0001);
    let support_gate = smoothstep(1.5, 3.5, support_count);
    return saturate(max(avg_exposure * 1.10, strong * 0.55) * support_gate);
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

fn smootherstep01(v: f32) -> f32 {
    let t = clamp(v, 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn normalize_direction_descriptor(v: vec4f) -> vec4f {
    let len_v = length(v.xyz);
    let dir = select(vec3f(0.0, 1.0, 0.0), v.xyz / max(len_v, 0.0001), len_v > 0.0001);
    return vec4f(dir * saturate(len_v), saturate(v.w));
}

fn rc_active_direction_count(cascade_index: u32) -> i32 {
    // v9: C0=4 was too sparse for voxel hills and showed visible probe-scale
    // splotches. Keep the 16-slot atlas, but give the near cascades enough
    // angular buckets that open-sky/terrain samples do not collapse into four
    // unstable lobes.
    if (cascade_index == 0u) { return 8; }
    if (cascade_index == 1u) { return 12; }
    return RC_DIRECTION_COUNT;
}

fn rc_direction_basis(slot: i32, active_count: i32) -> vec3f {
    let count = clamp(active_count, 1, RC_DIRECTION_COUNT);
    let clamped_slot = clamp(slot, 0, count - 1);
    let n = f32(count);
    let i = f32(clamped_slot) + 0.5;
    let y = 1.0 - 2.0 * i / n;
    let r = sqrt(max(0.0, 1.0 - y * y));
    let theta = RC_BASIS_GOLDEN_ANGLE * i;
    return normalize(vec3f(cos(theta) * r, y, sin(theta) * r));
}

fn load_world_rc_dir(index: u32, slot: i32, coord: vec3i, resolution: i32) -> vec4f {
    let atlas_coord = vec3i(coord.x + slot * resolution, coord.y, coord.z);
    switch (index) {
        case 0u: { return textureLoad(rc0_tex, atlas_coord, 0); }
        case 1u: { return textureLoad(rc1_tex, atlas_coord, 0); }
        case 2u: { return textureLoad(rc2_tex, atlas_coord, 0); }
        default: { return textureLoad(rc3_tex, atlas_coord, 0); }
    }
}

fn safe_normalize3(v: vec3f, fallback: vec3f) -> vec3f {
    let len_v = length(v);
    return select(fallback, v / max(len_v, 0.0001), len_v > 0.0001);
}

fn eon_fon_albedo_approx(mu_in: f32, roughness: f32) -> f32 {
    let mu = clamp(mu_in, 0.001, 1.0);
    let r = clamp(roughness, 0.0, 1.0);
    let mucomp = 1.0 - mu;
    let mucomp2 = mucomp * mucomp;
    let gover_pi =
        0.0571085289 * mucomp
        - 0.332181442 * mucomp2
        + 0.491881867 * mucomp * mucomp2
        + 0.0714429953 * mucomp2 * mucomp2;
    let constant1_fon = 0.5 - 2.0 / (3.0 * PI);
    return (1.0 + r * gover_pi) / (1.0 + constant1_fon * r);
}

fn eon_rough_diffuse_response(n: vec3f, view_dir: vec3f, light_dir: vec3f, roughness: f32, albedo_luma: f32) -> f32 {
    let mu_i = saturate(dot(n, light_dir));
    let mu_o = saturate(dot(n, view_dir));
    if (mu_i <= 0.0001 || mu_o <= 0.0001) {
        let wrapped = saturate((dot(n, light_dir) + 0.26) / 1.26);
        return wrapped * 0.075 * roughness;
    }

    let r = clamp(roughness, 0.0, 1.0);
    let rho = clamp(albedo_luma, 0.035, 0.985);
    let constant1_fon = 0.5 - 2.0 / (3.0 * PI);
    let constant2_fon = 2.0 / 3.0 - 28.0 / (15.0 * PI);
    let af = 1.0 / (1.0 + constant1_fon * r);

    // FON/EON single-scatter angular term. The linked EON paper expresses this
    // in local coordinates; the scalar s term can be evaluated directly from
    // world-space directions and the normal.
    let s = dot(light_dir, view_dir) - mu_i * mu_o;
    let sovert_f = select(s, s / max(mu_i, mu_o), s > 0.0);
    let f_ss_over_rho = (af / PI) * max(0.0, 1.0 + r * sovert_f);

    let efi = eon_fon_albedo_approx(mu_i, r);
    let efo = eon_fon_albedo_approx(mu_o, r);
    let avg_ef = af * (1.0 + constant2_fon * r);

    // Energy compensation from the EON model. We divide by rho because this
    // compose pass applies the block albedo after RC lighting; this keeps the
    // angular compensation without double-tinting the material color.
    let rho_ms = (rho * rho) * avg_ef / max(0.0001, 1.0 - rho * (1.0 - avg_ef));
    let f_ms_over_rho = (rho_ms / max(rho, 0.0001))
        * max(0.000001, 1.0 - efo)
        * max(0.000001, 1.0 - efi)
        / (PI * max(0.000001, 1.0 - avg_ef));

    let response = PI * (f_ss_over_rho + f_ms_over_rho) * mu_i;
    let rough_horizon_wrap = saturate((dot(n, light_dir) + 0.24) / 1.24) * 0.055 * r;
    return clamp(max(response, rough_horizon_wrap), 0.0, 1.25);
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
    let view_dir = safe_normalize3(camera.position.xyz - world_pos, -camera.forward.xyz);
    let albedo_luma = clamp(dot(albedo, vec3f(0.2126, 0.7152, 0.0722)), 0.035, 0.985);

    // RC is a positive lighting field, but direct daylight should not be
    // injected as a hard per-probe floor. Use a smooth sky-visibility baseline
    // for open upward terrain, then add probe-traced indirect radiance on top.
    let rc_debug_mode = i32(params.merge_control.x + 0.5);
    var rc_indirect = vec3f(0.0, 0.0, 0.0);
    var sky_signal_accum = 0.0;
    var sky_signal_weight = 0.0;
    var rc_confidence_max = 0.0;
    var debug_cascade_contrib = vec3f(0.0, 0.0, 0.0);
    var debug_cascade_confidence = vec3f(0.0, 0.0, 0.0);
    var debug_volume_coverage = vec3f(0.0, 0.0, 0.0);
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
        var coverage_color = vec3f(1.0, 0.22, 0.08);
        if (i == 1u) {
            coverage_color = vec3f(0.20, 1.0, 0.22);
        } else if (i == 2u) {
            coverage_color = vec3f(0.18, 0.36, 1.0);
        } else if (i >= 3u) {
            coverage_color = vec3f(1.0, 0.28, 1.0);
        }
        debug_volume_coverage = max(debug_volume_coverage, coverage_color * 0.60);

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
                    let probe_vis = load_world_rc_vis(i, coord);
                    let sky_access = saturate(probe_vis.w);

                    let probe_uvw = vec3f(f32(coord.x), f32(coord.y), f32(coord.z)) / f32(max_index);
                    let probe_world = min_corner + probe_uvw / inv_extent;
                    let to_probe = probe_world - world_pos;
                    let probe_dist = length(to_probe);
                    if (probe_dist <= 0.0001) {
                        continue;
                    }
                    let probe_dir = to_probe / probe_dist;
                    let backface = saturate((dot(normal, probe_dir) + PROBE_BACKFACE_SOFTNESS) / (1.0 + PROBE_BACKFACE_SOFTNESS));
                    let geometric_weight = max(0.18, backface);
                    let open_probe_gate = smoothstep(0.02, 0.32, sky_access);
                    let open_surface_probe_gate = mix(1.0, mix(0.35, 1.0, open_probe_gate), upward_surface);
                    let terrain_roughness = 0.82;

                    var probe_alpha_max = 0.0;
                    var probe_directional_sum = vec3f(0.0, 0.0, 0.0);
                    var probe_directional_weight = 0.0;
                    let active_dir_count = rc_active_direction_count(i);
                    for (var dir_slot: i32 = 0; dir_slot < active_dir_count; dir_slot = dir_slot + 1) {
                        let dir_probe = load_world_rc_dir(i, dir_slot, coord, resolution);
                        let dir_alpha = saturate(dir_probe.a);
                        if (dir_alpha <= 0.0001) {
                            continue;
                        }
                        let incoming_dir = rc_direction_basis(dir_slot, active_dir_count);
                        let eon_response = eon_rough_diffuse_response(normal, view_dir, incoming_dir, terrain_roughness, albedo_luma);
                        let sky_hemi_response = sky_access * upward_surface * saturate(incoming_dir.y);
                        let receiver_response = max(eon_response, sky_hemi_response * 0.55);
                        let directional_response = clamp(mix(0.05, 1.12, receiver_response), 0.0, 1.12);
                        probe_directional_sum = probe_directional_sum + dir_probe.rgb * dir_alpha * directional_response;
                        probe_directional_weight = probe_directional_weight + dir_alpha;
                        probe_alpha_max = max(probe_alpha_max, dir_alpha);
                    }
                    if (probe_directional_weight <= 0.0001 || probe_alpha_max <= 0.0001) {
                        continue;
                    }

                    let validity = mix(0.10, 1.0, probe_alpha_max);
                    let structural_weight = tri_weight * geometric_weight * validity * validity * probe_alpha_max * open_surface_probe_gate;
                    let exposure_weight = tri_weight * max(0.20, geometric_weight) * validity * probe_alpha_max * open_surface_probe_gate;

                    cascade_accum = cascade_accum + probe_directional_sum * structural_weight;
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
        let transition_blend = smootherstep01(edge_blend);
        var boundary_blend = 1.0;
        if (i + 1u < cascade_count) {
            boundary_blend = mix(PROBE_CASCADE_BLEND_MIN_WEIGHT, 1.0, transition_blend);
        }
        let cascade_preference = mix(1.0, PROBE_CASCADE_BLEND_FAR_BIAS, f32(i) / max(1.0, f32(cascade_count - 1u)));
        let raw_confidence = clamp(cascade_alpha * boundary_blend * cascade_preference, 0.0, 1.0);
        let confidence = smoothstep(0.025, 0.92, raw_confidence);
        rc_confidence_max = max(rc_confidence_max, confidence * cascade_preference);

        // Probe radiance is indirect/additive only. Dark cascades add zero; they
        // never reduce light by participating in a normalized average.
        // v8: the earlier tuning intentionally crushed far cascades to avoid
        // additive overbright, but it made C1/C2/C3 visibly much darker than C0.
        // Rebalance the compose energy so farther cascades still look like the
        // same lighting field, just blurrier/lower-detail rather than darker.
        var cascade_add_weight = 1.0;
        if (i == 1u) {
            cascade_add_weight = 0.92;
        } else if (i == 2u) {
            cascade_add_weight = 0.84;
        } else if (i >= 3u) {
            cascade_add_weight = 0.76;
        }
        let far_energy_comp = mix(1.0, 1.18, f32(i) / max(1.0, f32(cascade_count - 1u)));
        let edge_handoff_damper = select(1.0, mix(0.88, 1.0, transition_blend), i + 1u < cascade_count);
        let cascade_contribution = cascade_rgb_raw * confidence * cascade_add_weight * far_energy_comp * edge_handoff_damper;
        rc_indirect = rc_indirect + cascade_contribution;

        var cascade_debug_color = vec3f(1.0, 0.22, 0.08);
        if (i == 1u) {
            cascade_debug_color = vec3f(0.20, 1.0, 0.22);
        } else if (i == 2u) {
            cascade_debug_color = vec3f(0.18, 0.36, 1.0);
        } else if (i >= 3u) {
            cascade_debug_color = vec3f(1.0, 0.28, 1.0);
        }
        let cascade_contribution_luma = dot(cascade_contribution, vec3f(0.2126, 0.7152, 0.0722));
        debug_cascade_contrib = debug_cascade_contrib + cascade_debug_color * cascade_contribution_luma;
        debug_cascade_confidence = debug_cascade_confidence + cascade_debug_color * confidence * 0.35;

        // Sky visibility is a smooth daylight mask, not visible probe radiance.
        // Weight C1 most strongly so the open-sky baseline is broad instead of
        // shaped like individual C0 cells. Hard-occluded caves keep this at zero.
        // Keep C0 from driving the broad daylight field by itself. C0 is close
        // to terrain and intentionally high-frequency; using C1/C2 as the main
        // sky baseline removes near-probe blotches while preserving C0 detail.
        var sky_cascade_weight = 0.34;
        if (i == 1u) {
            sky_cascade_weight = 1.0;
        } else if (i == 2u) {
            sky_cascade_weight = 0.82;
        } else if (i >= 3u) {
            sky_cascade_weight = 0.62;
        }
        let sky_signal = smoothstep(0.18, 0.72, cascade_sky_access);
        let sky_transition_weight = mix(0.72, 1.0, transition_blend);
        sky_signal_accum = sky_signal_accum + sky_signal * confidence * sky_cascade_weight * sky_transition_weight;
        sky_signal_weight = sky_signal_weight + confidence * sky_cascade_weight * sky_transition_weight;
    }

    let smoothed_sky_signal = smoothstep(0.06, 0.84, sky_signal_accum / max(sky_signal_weight, 0.0001));
    let hemisphere = clamp(normal.y * 0.5 + 0.5, 0.0, 1.0);
    let rc_coverage = saturate(sky_signal_weight * 0.95);
    let near_surface = 1.0 - smoothstep(8.0, 40.0, depth);
    let medium_surface = smoothstep(24.0, 160.0, depth);
    let distant_surface = smoothstep(96.0, 384.0, depth);
    let close_terrain_daylight_floor =
        near_surface
        * upward_surface
        * smoothstep(0.04, 0.32, max(smoothed_sky_signal, rc_coverage * 0.65))
        * mix(0.025, 0.085, hemisphere);
    let terrain_daylight_floor = max(
        close_terrain_daylight_floor,
        medium_surface * mix(0.06, 0.18, hemisphere) * (0.40 + 0.60 * (1.0 - smoothed_sky_signal))
    );
    let distant_open_fallback = (1.0 - rc_coverage) * distant_surface * mix(0.14, 0.70, hemisphere);
    let local_visible_surface = 1.0 - smoothstep(40.0, 360.0, depth);
    let screen_exterior_hint = screen_space_sky_exposure(input.uv);
    let rc_hole = 1.0 - smoothstep(0.025, 0.30, rc_confidence_max);
    let top_surface = smoothstep(0.42, 0.92, normal.y);
    let height_locality = 1.0 - smoothstep(38.0, 128.0, abs(world_pos.y - camera.position.y));

    let screen_exterior_floor =
        screen_exterior_hint
        * rc_hole
        * mix(0.06, 0.22, max(near_surface, local_visible_surface))
        * mix(0.55, 1.0, hemisphere);

    // Non-angle-dependent recovery for the exact failure in the diagnostics:
    // a visible upward terrain surface inside the RC volumes, but with zero
    // probe confidence / sky signal. Keep this a modest exterior-style floor
    // rather than a radiance term, so RC still owns actual bounce lighting.
    let local_top_confidence_floor =
        top_surface
        * local_visible_surface
        * rc_hole
        * mix(0.12, 0.34, max(height_locality, near_surface))
        * mix(0.65, 1.0, hemisphere);

    let base_open_sky = max(
        smoothed_sky_signal,
        max(max(terrain_daylight_floor, distant_open_fallback), max(screen_exterior_floor, local_top_confidence_floor))
    );
    let confidence_recovery_floor =
        top_surface
        * max(near_surface, height_locality * 0.70)
        * rc_hole
        * smoothstep(0.005, 0.08, base_open_sky)
        * mix(0.04, 0.14, hemisphere);
    let open_sky = max(base_open_sky, confidence_recovery_floor);
    let sky_baseline = vec3f(0.38, 0.40, 0.38) * open_sky * mix(0.34, 1.0, hemisphere);
    let rc_indirect_soft = rc_indirect / (vec3f(1.0, 1.0, 1.0) + rc_indirect);
    let rc_light = max(sky_baseline + rc_indirect_soft * 0.36, vec3f(0.0, 0.0, 0.0));

    if (rc_debug_mode == 1) {
        let dbg = debug_cascade_contrib / (vec3f(1.0, 1.0, 1.0) + debug_cascade_contrib);
        return vec4f(clamp(dbg * 2.4, vec3f(0.0), vec3f(1.0)), 1.0);
    }
    if (rc_debug_mode == 2) {
        let rc_open_debug = max(smoothed_sky_signal, max(terrain_daylight_floor, distant_open_fallback));
        return vec4f(vec3f(rc_open_debug), 1.0);
    }
    if (rc_debug_mode == 3) {
        return vec4f(normal * 0.5 + vec3f(0.5), 1.0);
    }
    if (rc_debug_mode == 4) {
        return vec4f(clamp(debug_cascade_confidence, vec3f(0.0), vec3f(1.0)), 1.0);
    }
    if (rc_debug_mode == 5) {
        return vec4f(clamp(rc_indirect_soft * 2.2, vec3f(0.0), vec3f(1.0)), 1.0);
    }
    if (rc_debug_mode == 6) {
        return vec4f(clamp(rc_light, vec3f(0.0), vec3f(1.0)), 1.0);
    }
    if (rc_debug_mode == 7) {
        return vec4f(clamp(debug_volume_coverage, vec3f(0.0), vec3f(1.0)), 1.0);
    }

    let lit = clamp(albedo * rc_light * strength, vec3f(0.0, 0.0, 0.0), vec3f(1.0, 1.0, 1.0));
    return vec4f(lit, 1.0);
}
