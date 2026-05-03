
struct PresentParams {
    viewport_inv_px: vec2f,
    reserved: vec2f,
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> params: PresentParams;
@group(0) @binding(3) var gbuffer_tex: texture_2d<f32>;

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

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn load_gbuffer_at(xy: vec2i, dims: vec2i) -> vec4f {
    let clamped = clamp(xy, vec2i(0, 0), dims - vec2i(1, 1));
    return textureLoad(gbuffer_tex, clamped, 0);
}

fn crease_shadow_sample(center_xy: vec2i, offset: vec2i, dims: vec2i, center_depth: f32, center_normal: vec3f) -> f32 {
    let g = load_gbuffer_at(center_xy + offset, dims);
    let sample_depth = g.w;
    let sample_normal_len = length(g.xyz);
    if (sample_depth <= 0.0 || sample_normal_len <= 0.001) {
        return 0.0;
    }

    let sample_normal = g.xyz / sample_normal_len;
    let depth_delta = center_depth - sample_depth;
    let depth_radius = max(0.055, center_depth * 0.018);
    let depth_bias = max(0.004, center_depth * 0.0012);
    let range_weight = 1.0 - smoothstep(depth_radius, depth_radius * 4.0, abs(depth_delta));
    let depth_occlusion = smoothstep(depth_bias, depth_radius, depth_delta);
    let normal_delta = 1.0 - saturate(dot(center_normal, sample_normal));
    let crease_occlusion = smoothstep(0.10, 0.70, normal_delta) * range_weight * 0.55;
    return max(depth_occlusion * range_weight, crease_occlusion);
}

fn screen_space_crease_shadow(uv: vec2f) -> f32 {
    let dims_u = textureDimensions(gbuffer_tex, 0u);
    let dims = vec2i(dims_u);
    let center_xy = clamp(vec2i(uv * vec2f(dims_u)), vec2i(0, 0), dims - vec2i(1, 1));
    let center_g = load_gbuffer_at(center_xy, dims);
    let center_depth = center_g.w;
    let center_normal_len = length(center_g.xyz);
    if (center_depth <= 0.0 || center_normal_len <= 0.001) {
        return 1.0;
    }

    let center_normal = center_g.xyz / center_normal_len;
    let offsets = array<vec2i, 12>(
        vec2i(1, 0),
        vec2i(-1, 0),
        vec2i(0, 1),
        vec2i(0, -1),
        vec2i(2, 1),
        vec2i(-2, 1),
        vec2i(2, -1),
        vec2i(-2, -1),
        vec2i(4, 0),
        vec2i(-4, 0),
        vec2i(0, 4),
        vec2i(0, -4),
    );
    var occlusion = 0.0;
    for (var i = 0u; i < 12u; i = i + 1u) {
        occlusion = occlusion + crease_shadow_sample(center_xy, offsets[i], dims, center_depth, center_normal);
    }
    let normalized = saturate(occlusion / 12.0);
    return clamp(1.0 - normalized * 1.35, 0.42, 1.0);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let _inv_px = params.viewport_inv_px;
    let color = textureSample(src_tex, src_sampler, input.uv).rgb;
    return vec4f(color * screen_space_crease_shadow(input.uv), 1.0);
}
