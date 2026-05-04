
struct RcUpdateParams {
    min_corner_and_extent: vec4f,
    meta0: vec4f,
    meta1: vec4f,
    light0: vec4f,
    light_dir: vec4f,
    controls: vec4f,
}

struct WorldRcParams {
    volume_min: array<vec4f, 4>,
    volume_inv_extent: array<vec4f, 4>,
}

@group(0) @binding(0) var dst_volume: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1) var dst_visibility: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> rc: RcUpdateParams;
@group(0) @binding(3) var rc0_src_tex: texture_3d<f32>;
@group(0) @binding(4) var rc1_src_tex: texture_3d<f32>;
@group(0) @binding(5) var rc2_src_tex: texture_3d<f32>;
@group(0) @binding(6) var rc3_src_tex: texture_3d<f32>;
@group(0) @binding(7) var rc0_src_vis_tex: texture_3d<f32>;
@group(0) @binding(8) var rc1_src_vis_tex: texture_3d<f32>;
@group(0) @binding(9) var rc2_src_vis_tex: texture_3d<f32>;
@group(0) @binding(10) var rc3_src_vis_tex: texture_3d<f32>;
@group(0) @binding(11) var<uniform> world_rc: WorldRcParams;

const AIR: u32 = 0u;
const BEDROCK: u32 = 1u;
const STONE: u32 = 2u;
const DIRT: u32 = 3u;
const GRASS: u32 = 4u;
const SAND: u32 = 5u;
const SNOW: u32 = 6u;

const TERRAIN_FREQUENCY_SCALE: f32 = 0.30000000;
const CAVE_FREQUENCY_SCALE: f32 = 1.00000000;
const CAVE_DETAIL_FREQUENCY_MULTIPLIER: f32 = 3.00000000;
const CAVE_DETAIL_WEIGHT: f32 = 0.18000000;
const CAVE_BEDROCK_CLEARANCE: i32 = 3;
const CAVE_ACTIVE_BAND_MIN: f32 = 0.58000000;
const CAVE_PRIMARY_THRESHOLD: f32 = 0.66000000;
const CAVE_VERTICAL_BONUS: f32 = 0.06000000;
const CAVE_DEPTH_BONUS_SCALE: f32 = 0.00150000;
const CAVE_DEPTH_BONUS_MAX: f32 = 0.06000000;
const GOLDEN_ANGLE: f32 = 2.39996322972865332;

const SKY_VISIBILITY_STEPS: u32 = __SKY_VISIBILITY_STEPS__u;
const SKY_VISIBILITY_STEP_BLOCKS: u32 = __SKY_VISIBILITY_STEP_BLOCKS__u;
const SKY_VISIBILITY_SIDE_WEIGHT: f32 = __SKY_VISIBILITY_SIDE_WEIGHT__;
const SKY_VISIBILITY_APERTURE_RADIUS_BLOCKS: i32 = __SKY_VISIBILITY_APERTURE_RADIUS_BLOCKS__;
const SKY_VISIBILITY_APERTURE_POWER: f32 = __SKY_VISIBILITY_APERTURE_POWER__;
const SKY_VISIBILITY_MIN_APERTURE: f32 = __SKY_VISIBILITY_MIN_APERTURE__;
const RC_DIRECTION_COUNT: i32 = 16;
const RC_TEMPORAL_ALPHA: f32 = __RC_TEMPORAL_ALPHA__;
const RC_BOUNCE_FEEDBACK_STRENGTH: f32 = __RC_BOUNCE_FEEDBACK_STRENGTH__;
const RC_FEEDBACK_MAX_LUMA: f32 = 0.42;
const RC_INTERVAL_MAX_LUMA: f32 = 1.15;
const RC_FAR_MERGE_MAX_LUMA: f32 = 0.90;
const RC_FAR_MERGE_RINGING_VARIANCE_SCALE: f32 = 7.5;
const RC_FAR_MERGE_MIN_STABILITY: f32 = 0.42;
const RC_MERGE_CONTINUITY_FEATHER: f32 = __RC_MERGE_CONTINUITY_FEATHER__;
const RC_DDA_MAX_VISITS: u32 = __RC_DDA_MAX_VISITS__u;
const RC_DDA_INF: f32 = 1.0e20;

fn saturate(v: f32) -> f32 {
    return clamp(v, 0.0, 1.0);
}

fn smootherstep01(v: f32) -> f32 {
    let t = clamp(v, 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn rc_luma(rgb: vec3f) -> f32 {
    return dot(max(rgb, vec3f(0.0, 0.0, 0.0)), vec3f(0.2126, 0.7152, 0.0722));
}

fn rc_clamp_luma(rgb: vec3f, max_luma: f32) -> vec3f {
    let safe_rgb = max(rgb, vec3f(0.0, 0.0, 0.0));
    let luma = rc_luma(safe_rgb);
    let scale = min(1.0, max_luma / max(luma, 0.0001));
    return safe_rgb * scale;
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
    let depth_below_surface_i = i32(surface_height) - world_y;
    if (depth_below_surface_i <= 0) {
        return false;
    }
    if (world_y >= i32(world_height_limit()) - 2) {
        return false;
    }

    let normalized_y = f32(world_y) / f32(max(1u, world_height_limit() - 1u));
    var vertical_band = 1.0;
    if (normalized_y > 0.45) {
        vertical_band = saturate(1.0 - (normalized_y - 0.45) * 1.6);
    }

    let seed = seed_u32();
    if (vertical_band <= CAVE_ACTIVE_BAND_MIN) {
        return false;
    }

    let xf = f32(world_x);
    let yf = f32(world_y);
    let zf = f32(world_z);
    let cave_frequency = 0.018 * CAVE_FREQUENCY_SCALE;
    let cave_primary = value_noise_3d(xf, yf * 0.85, zf, seed + 101u, cave_frequency);
    let cave_detail = value_noise_3d(xf, yf * 0.85, zf, seed + 157u, cave_frequency * CAVE_DETAIL_FREQUENCY_MULTIPLIER);
    let cave_value = cave_primary + cave_detail * CAVE_DETAIL_WEIGHT;
    let depth_below_surface = f32(depth_below_surface_i);
    let depth_bonus = min(depth_below_surface * CAVE_DEPTH_BONUS_SCALE, CAVE_DEPTH_BONUS_MAX);
    let threshold = CAVE_PRIMARY_THRESHOLD - vertical_band * CAVE_VERTICAL_BONUS - depth_bonus;
    return cave_value > threshold;
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

    // v7: keep the world-space RC sky classifier cheap again. The earlier
    // side/cone checks were called from many trace samples and tanked frame time,
    // while the diagnostics showed the remaining black hill problem was better
    // handled in compose as a visible-surface recovery path.
    return column_sky_access(bx, by, bz, 0, 0);
}

fn direction_for_base_index(index: u32, base_count: u32) -> vec3f {
    let y = 1.0 - (2.0 * (f32(index) + 0.5) / f32(max(1u, base_count)));
    let radius = sqrt(max(0.0, 1.0 - y * y));
    let theta = GOLDEN_ANGLE * f32(index);
    return vec3f(cos(theta) * radius, y, sin(theta) * radius);
}

fn normalize_direction_descriptor(v: vec4f) -> vec4f {
    let len_v = length(v.xyz);
    let dir = select(vec3f(0.0, 1.0, 0.0), v.xyz / max(len_v, 0.0001), len_v > 0.0001);
    return vec4f(dir * saturate(len_v), saturate(v.w));
}

fn rc_active_direction_count(cascade_index: u32) -> i32 {
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
    let theta = GOLDEN_ANGLE * i;
    return normalize(vec3f(cos(theta) * r, y, sin(theta) * r));
}

fn rc_direction_slot(dir: vec3f, active_count: i32) -> i32 {
    let count = clamp(active_count, 1, RC_DIRECTION_COUNT);
    var best_slot = 0;
    var best_dot = dot(dir, rc_direction_basis(0, count));
    for (var slot: i32 = 1; slot < count; slot = slot + 1) {
        let d = dot(dir, rc_direction_basis(slot, count));
        if (d > best_dot) {
            best_dot = d;
            best_slot = slot;
        }
    }
    return best_slot;
}

fn load_src_world_rc(index: u32, coord: vec3i) -> vec4f {
    switch (index) {
        case 0u: { return textureLoad(rc0_src_tex, coord, 0); }
        case 1u: { return textureLoad(rc1_src_tex, coord, 0); }
        case 2u: { return textureLoad(rc2_src_tex, coord, 0); }
        default: { return textureLoad(rc3_src_tex, coord, 0); }
    }
}

fn load_src_world_rc_vis(index: u32, coord: vec3i) -> vec4f {
    switch (index) {
        case 0u: { return textureLoad(rc0_src_vis_tex, coord, 0); }
        case 1u: { return textureLoad(rc1_src_vis_tex, coord, 0); }
        case 2u: { return textureLoad(rc2_src_vis_tex, coord, 0); }
        default: { return textureLoad(rc3_src_vis_tex, coord, 0); }
    }
}

fn sample_src_world_rc(index: u32, world_pos: vec3f, resolution: i32) -> vec4f {
    let min_corner = world_rc.volume_min[index].xyz;
    let inv_extent = world_rc.volume_inv_extent[index].xyz;
    let uvw = (world_pos - min_corner) * inv_extent;
    if (uvw.x < 0.0 || uvw.y < 0.0 || uvw.z < 0.0 || uvw.x > 1.0 || uvw.y > 1.0 || uvw.z > 1.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    let max_index = max(resolution - 1, 1);
    let grid_f = uvw * f32(max_index);
    let base_x = i32(clamp(floor(grid_f.x), 0.0, f32(max_index - 1)));
    let base_y = i32(clamp(floor(grid_f.y), 0.0, f32(max_index - 1)));
    let base_z = i32(clamp(floor(grid_f.z), 0.0, f32(max_index - 1)));
    let frac = fract(grid_f);
    var accum = vec4f(0.0);
    for (var oz: u32 = 0u; oz < 2u; oz = oz + 1u) {
        let wz = select(1.0 - frac.z, frac.z, oz == 1u);
        for (var oy: u32 = 0u; oy < 2u; oy = oy + 1u) {
            let wy = select(1.0 - frac.y, frac.y, oy == 1u);
            for (var ox: u32 = 0u; ox < 2u; ox = ox + 1u) {
                let wx = select(1.0 - frac.x, frac.x, ox == 1u);
                let coord = vec3i(base_x + i32(ox), base_y + i32(oy), base_z + i32(oz));
                accum = accum + load_src_world_rc(index, coord) * (wx * wy * wz);
            }
        }
    }
    return accum;
}

fn load_src_world_rc_dir(index: u32, slot: i32, coord: vec3i, resolution: i32) -> vec4f {
    return load_src_world_rc(index, vec3i(coord.x + slot * resolution, coord.y, coord.z));
}

fn sample_src_world_rc_dir(index: u32, world_pos: vec3f, resolution: i32, slot: i32) -> vec4f {
    let min_corner = world_rc.volume_min[index].xyz;
    let inv_extent = world_rc.volume_inv_extent[index].xyz;
    let uvw = (world_pos - min_corner) * inv_extent;
    if (uvw.x < 0.0 || uvw.y < 0.0 || uvw.z < 0.0 || uvw.x > 1.0 || uvw.y > 1.0 || uvw.z > 1.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    let max_index = max(resolution - 1, 1);
    let grid_f = uvw * f32(max_index);
    let base_x = i32(clamp(floor(grid_f.x), 0.0, f32(max_index - 1)));
    let base_y = i32(clamp(floor(grid_f.y), 0.0, f32(max_index - 1)));
    let base_z = i32(clamp(floor(grid_f.z), 0.0, f32(max_index - 1)));
    let frac = fract(grid_f);
    var accum = vec4f(0.0);
    for (var oz: u32 = 0u; oz < 2u; oz = oz + 1u) {
        let wz = select(1.0 - frac.z, frac.z, oz == 1u);
        for (var oy: u32 = 0u; oy < 2u; oy = oy + 1u) {
            let wy = select(1.0 - frac.y, frac.y, oy == 1u);
            for (var ox: u32 = 0u; ox < 2u; ox = ox + 1u) {
                let wx = select(1.0 - frac.x, frac.x, ox == 1u);
                let coord = vec3i(base_x + i32(ox), base_y + i32(oy), base_z + i32(oz));
                accum = accum + load_src_world_rc_dir(index, slot, coord, resolution) * (wx * wy * wz);
            }
        }
    }
    return accum;
}

fn sample_src_world_rc_angular(index: u32, world_pos: vec3f, resolution: i32, ray_dir: vec3f, active_count: i32) -> vec4f {
    // Forked angular reprojection: a current-cascade miss does not map to one
    // brittle far angular texel. Gather all active far intervals with a sharp
    // positive dot kernel, producing a smoother directional interval merge that
    // is closer to canonical RC radiance carry/forking behavior.
    let count = clamp(active_count, 1, RC_DIRECTION_COUNT);
    let dir_n = normalize(ray_dir);
    var rgb_accum = vec3f(0.0, 0.0, 0.0);
    var weight_accum = 0.0;
    var alpha_accum = 0.0;
    var angular_weight_accum = 0.0;
    for (var slot: i32 = 0; slot < RC_DIRECTION_COUNT; slot = slot + 1) {
        if (slot >= count) {
            continue;
        }
        let basis = rc_direction_basis(slot, count);
        let align = max(0.0, dot(dir_n, basis));
        let angular_weight = pow(align, 7.0);
        if (angular_weight <= 0.000001) {
            continue;
        }
        angular_weight_accum = angular_weight_accum + angular_weight;
        let probe = sample_src_world_rc_dir(index, world_pos, resolution, slot);
        let a = saturate(probe.a);
        if (a <= 0.0001) {
            continue;
        }
        let w = angular_weight * a;
        rgb_accum = rgb_accum + probe.rgb * w;
        weight_accum = weight_accum + w;
        alpha_accum = alpha_accum + angular_weight * a;
    }
    if (weight_accum <= 0.0001) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    return vec4f(rgb_accum / weight_accum, saturate(alpha_accum / max(angular_weight_accum, 0.0001)));
}

fn stable_tangent_a(ray_dir: vec3f) -> vec3f {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(ray_dir.y) > 0.82);
    return normalize(cross(up, ray_dir));
}

fn block_coord_from_world(world_pos: vec3f, block_size: f32) -> vec3i {
    return vec3i(
        i32(floor(world_pos.x / block_size)),
        i32(floor(world_pos.y / block_size)),
        i32(floor(world_pos.z / block_size)),
    );
}

struct RcDdaHit {
    hit: u32,
    material: u32,
    block: vec3i,
    normal: vec3f,
    distance: f32,
    visits: u32,
}

fn dda_axis_step(v: f32) -> i32 {
    if (v > 0.000001) {
        return 1;
    }
    if (v < -0.000001) {
        return -1;
    }
    return 0;
}

fn dda_axis_delta(v: f32, block_size: f32) -> f32 {
    if (abs(v) <= 0.000001) {
        return RC_DDA_INF;
    }
    return abs(block_size / v);
}

fn dda_axis_t_max(pos_axis: f32, dir_axis: f32, block_axis: i32, step_axis: i32, block_size: f32) -> f32 {
    if (step_axis > 0) {
        let boundary = f32(block_axis + 1) * block_size;
        return max(0.0, (boundary - pos_axis) / dir_axis);
    }
    if (step_axis < 0) {
        let boundary = f32(block_axis) * block_size;
        return max(0.0, (boundary - pos_axis) / dir_axis);
    }
    return RC_DDA_INF;
}

fn dda_miss(distance: f32, visits: u32) -> RcDdaHit {
    return RcDdaHit(0u, AIR, vec3i(0, 0, 0), vec3f(0.0, 0.0, 0.0), distance, visits);
}

fn trace_first_solid_dda(origin: vec3f, ray_dir: vec3f, start_dist: f32, end_dist: f32, block_size: f32, max_visits: u32) -> RcDdaHit {
    // Amanatides-Woo 3D DDA over Minechunk's actual block grid.  This visits
    // crossed blocks in order instead of probing at fixed distances, so thin
    // voxel blockers near cascade boundaries are much harder to skip.
    let ray_len = length(ray_dir);
    if (ray_len <= 0.000001) {
        return dda_miss(end_dist, 0u);
    }
    let dir_n = ray_dir / ray_len;
    let safe_start = max(0.0, start_dist) + min(block_size * 0.0025, 0.001);
    let safe_end = max(safe_start, end_dist);
    let budget = max(1u, max_visits);

    let start_pos = origin + dir_n * safe_start;
    var block = block_coord_from_world(start_pos, block_size);
    let step = vec3i(
        dda_axis_step(dir_n.x),
        dda_axis_step(dir_n.y),
        dda_axis_step(dir_n.z),
    );

    var t_max_x = safe_start + dda_axis_t_max(start_pos.x, dir_n.x, block.x, step.x, block_size);
    var t_max_y = safe_start + dda_axis_t_max(start_pos.y, dir_n.y, block.y, step.y, block_size);
    var t_max_z = safe_start + dda_axis_t_max(start_pos.z, dir_n.z, block.z, step.z, block_size);
    let t_delta_x = dda_axis_delta(dir_n.x, block_size);
    let t_delta_y = dda_axis_delta(dir_n.y, block_size);
    let t_delta_z = dda_axis_delta(dir_n.z, block_size);

    var t_enter = safe_start;
    var normal = vec3f(0.0, 0.0, 0.0);
    var visits = 0u;

    loop {
        if (visits >= budget || t_enter > safe_end) {
            break;
        }

        let material = material_at_block(block.x, block.y, block.z);
        if (material != AIR) {
            return RcDdaHit(1u, material, block, normal, t_enter, visits);
        }

        visits = visits + 1u;
        let next_t = min(t_max_x, min(t_max_y, t_max_z));
        if (next_t > safe_end) {
            break;
        }

        // Classic one-axis Amanatides-Woo step.  Ties resolve X -> Y -> Z for
        // determinism and to avoid repeatedly visiting the same corner cells.
        if (t_max_x <= t_max_y && t_max_x <= t_max_z) {
            block.x = block.x + step.x;
            t_enter = t_max_x;
            t_max_x = t_max_x + t_delta_x;
            normal = vec3f(-f32(step.x), 0.0, 0.0);
        } else if (t_max_y <= t_max_z) {
            block.y = block.y + step.y;
            t_enter = t_max_y;
            t_max_y = t_max_y + t_delta_y;
            normal = vec3f(0.0, -f32(step.y), 0.0);
        } else {
            block.z = block.z + step.z;
            t_enter = t_max_z;
            t_max_z = t_max_z + t_delta_z;
            normal = vec3f(0.0, 0.0, -f32(step.z));
        }
    }

    return dda_miss(safe_end, visits);
}


fn local_interval_visibility(origin: vec3f, ray_dir: vec3f, start_dist: f32, end_dist: f32, block_size: f32) -> f32 {
    // Amanatides-Woo guard for far-cascade interval merge.  The old fixed-step
    // probe could jump across one-block occluders, then merge bright/dark stale
    // far probes through local terrain.  DDA visits every crossed block up to a
    // capped budget, which is much safer around small sky holes and cave lips.
    let safe_start = max(0.0, start_dist);
    let safe_end = max(safe_start, end_dist);
    let span = safe_end - safe_start;
    if (span <= block_size * 0.25) {
        return 1.0;
    }
    let visits_needed = u32(ceil(span / max(block_size, 0.000001))) + 3u;
    let visit_budget = min(RC_DDA_MAX_VISITS, max(1u, visits_needed));
    let hit = trace_first_solid_dda(origin, ray_dir, safe_start, safe_end, block_size, visit_budget);
    if (hit.hit != 0u) {
        return 0.0;
    }
    return 1.0;
}

fn sample_src_world_rc_angular_spatial(index: u32, world_pos: vec3f, resolution: i32, ray_dir: vec3f, active_count: i32, footprint_radius: f32) -> vec4f {
    // Spatially forked interval merge: gather the next cascade over a tiny
    // ray-stable footprint around the interval endpoint, then perform the same
    // angular fork. This reduces one-probe far-cascade grid/block artifacts while
    // keeping the canonical RC idea: local miss -> filtered farther interval.
    let dir_n = normalize(ray_dir);
    let ta = stable_tangent_a(dir_n);
    let tb = normalize(cross(dir_n, ta));
    let r = max(0.0, footprint_radius);

    var rgb_accum = vec3f(0.0, 0.0, 0.0);
    var alpha_accum = 0.0;
    var weight_accum = 0.0;
    var luma_accum = 0.0;
    var luma_sq_accum = 0.0;
    var footprint_weight_accum = 0.0;

    for (var tap: i32 = 0; tap < 5; tap = tap + 1) {
        var tap_offset = vec3f(0.0, 0.0, 0.0);
        var tap_weight = 0.42;
        if (tap == 1) {
            tap_offset = ta * r;
            tap_weight = 0.145;
        } else if (tap == 2) {
            tap_offset = -ta * r;
            tap_weight = 0.145;
        } else if (tap == 3) {
            tap_offset = tb * r;
            tap_weight = 0.145;
        } else if (tap == 4) {
            tap_offset = -tb * r;
            tap_weight = 0.145;
        }

        let probe = sample_src_world_rc_angular(index, world_pos + tap_offset, resolution, dir_n, active_count);
        let a = saturate(probe.a);
        if (a <= 0.0001) {
            continue;
        }
        let safe_rgb = rc_clamp_luma(probe.rgb, RC_FAR_MERGE_MAX_LUMA);
        let tap_luma = rc_luma(safe_rgb);
        let w = tap_weight * a;
        rgb_accum = rgb_accum + safe_rgb * w;
        alpha_accum = alpha_accum + a * tap_weight;
        weight_accum = weight_accum + w;
        luma_accum = luma_accum + tap_luma * tap_weight;
        luma_sq_accum = luma_sq_accum + tap_luma * tap_luma * tap_weight;
        footprint_weight_accum = footprint_weight_accum + tap_weight;
    }

    if (weight_accum <= 0.0001) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }

    // Ringing guard: if the spatial fork footprint contains very different
    // radiance values, reduce confidence instead of letting one hot/dark tap
    // create a hard bright/dark ring at the cascade transition.
    let mean_luma = luma_accum / max(footprint_weight_accum, 0.0001);
    let mean_luma_sq = luma_sq_accum / max(footprint_weight_accum, 0.0001);
    let luma_variance = max(0.0, mean_luma_sq - mean_luma * mean_luma);
    let ringing_stability = mix(
        1.0,
        RC_FAR_MERGE_MIN_STABILITY,
        saturate(luma_variance * RC_FAR_MERGE_RINGING_VARIANCE_SCALE),
    );
    let alpha = saturate(alpha_accum / max(0.0001, 0.42 + 0.145 * 4.0)) * ringing_stability;
    return vec4f(rgb_accum / weight_accum, alpha);
}

fn rough_diffuse_feedback_response(normal: vec3f, incoming_dir: vec3f, outgoing_dir: vec3f) -> f32 {
    let n = normalize(normal);
    let l = normalize(incoming_dir);
    let v = normalize(outgoing_dir);
    let ndotl_front = saturate(dot(n, l));
    let ndotl_back = saturate(dot(n, -l)) * 0.18;
    let ndotl = max(ndotl_front, ndotl_back);
    let ndotv = max(saturate(dot(n, v)), 0.08);
    let lt = l - n * dot(n, l);
    let vt = v - n * dot(n, v);
    let lt_len = length(lt);
    let vt_len = length(vt);
    var side_scatter = 0.0;
    if (lt_len > 0.0001 && vt_len > 0.0001) {
        side_scatter = max(0.0, dot(lt / lt_len, vt / vt_len));
    }
    let wrapped = saturate((dot(n, l) + 0.35) / 1.35) * 0.22;
    let rough_energy = 0.84 + 0.16 * side_scatter;
    return clamp(max(ndotl, wrapped) * (0.72 + 0.28 * ndotv) * rough_energy, 0.0, 1.0);
}

fn sample_src_world_rc_bounce(index: u32, world_pos: vec3f, normal: vec3f, outgoing_dir: vec3f, resolution: i32, active_count: i32) -> vec3f {
    // Previous-frame multi-bounce estimate for a hit surface. This samples the
    // directional RC history at the hit point and integrates it with a rough
    // diffuse response toward the outgoing ray direction. The caller gates it
    // with temporal history validity so moved cascades do not smear stale light.
    let count = clamp(active_count, 1, RC_DIRECTION_COUNT);
    var accum = vec3f(0.0, 0.0, 0.0);
    var weight_accum = 0.0;
    for (var slot: i32 = 0; slot < RC_DIRECTION_COUNT; slot = slot + 1) {
        if (slot >= count) {
            continue;
        }
        let probe = sample_src_world_rc_dir(index, world_pos, resolution, slot);
        let a = saturate(probe.a);
        if (a <= 0.0001) {
            continue;
        }
        let incoming_dir = rc_direction_basis(slot, count);
        let response = rough_diffuse_feedback_response(normal, incoming_dir, outgoing_dir);
        let w = a * response;
        accum = accum + probe.rgb * w;
        weight_accum = weight_accum + w;
    }
    if (weight_accum <= 0.0001) {
        return vec3f(0.0, 0.0, 0.0);
    }
    return accum / max(f32(count) * 0.35, 1.0);
}

fn sample_src_world_rc_vis(index: u32, world_pos: vec3f, resolution: i32) -> vec4f {
    let min_corner = world_rc.volume_min[index].xyz;
    let inv_extent = world_rc.volume_inv_extent[index].xyz;
    let uvw = (world_pos - min_corner) * inv_extent;
    if (uvw.x < 0.0 || uvw.y < 0.0 || uvw.z < 0.0 || uvw.x > 1.0 || uvw.y > 1.0 || uvw.z > 1.0) {
        return vec4f(0.0, 0.0, 0.0, 0.0);
    }
    let max_index = max(resolution - 1, 1);
    let grid_f = uvw * f32(max_index);
    let base_x = i32(clamp(floor(grid_f.x), 0.0, f32(max_index - 1)));
    let base_y = i32(clamp(floor(grid_f.y), 0.0, f32(max_index - 1)));
    let base_z = i32(clamp(floor(grid_f.z), 0.0, f32(max_index - 1)));
    let frac = fract(grid_f);
    var accum = vec4f(0.0);
    for (var oz: u32 = 0u; oz < 2u; oz = oz + 1u) {
        let wz = select(1.0 - frac.z, frac.z, oz == 1u);
        for (var oy: u32 = 0u; oy < 2u; oy = oy + 1u) {
            let wy = select(1.0 - frac.y, frac.y, oy == 1u);
            for (var ox: u32 = 0u; ox < 2u; ox = ox + 1u) {
                let wx = select(1.0 - frac.x, frac.x, ox == 1u);
                let coord = vec3i(base_x + i32(ox), base_y + i32(oy), base_z + i32(oz));
                accum = accum + load_src_world_rc_vis(index, coord) * (wx * wy * wz);
            }
        }
    }
    return normalize_direction_descriptor(accum);
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



@compute @workgroup_size(4, 4, 4)
fn trace_main(@builtin(global_invocation_id) gid: vec3u) {
    let resolution = max(1u, u32(rc.meta0.x + 0.5));
    let resolution_i = i32(resolution);
    if (gid.x >= resolution || gid.y >= resolution || gid.z >= resolution) {
        return;
    }

    let coord = vec3i(i32(gid.x), i32(gid.y), i32(gid.z));
    let min_corner = rc.min_corner_and_extent.xyz;
    let full_extent = max(rc.min_corner_and_extent.w, 0.0001);
    let max_distance = max(rc.meta0.z, 0.0001);
    let block_size = max(rc.meta0.w, 0.000001);
    let cascade_index = u32(rc.meta0.y + 0.5);
    // v30: real RC-style distance intervals. Each cascade traces a band:
    // C0 [0, R0], C1 [R0, R1], C2 [R1, R2], C3 [R2, R3].
    // The far-to-near merge then carries radiance past the end of this band.
    let interval_start = clamp(rc.controls.z, 0.0, max_distance);
    let interval_end = max(interval_start + block_size, max(rc.controls.w, interval_start + block_size));
    let interval_length = max(block_size, interval_end - interval_start);
    let active_dir_count = rc_active_direction_count(cascade_index);
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
            for (var clear_slot: i32 = 0; clear_slot < RC_DIRECTION_COUNT; clear_slot = clear_slot + 1) {
                textureStore(dst_volume, vec3i(coord.x + clear_slot * resolution_i, coord.y, coord.z), vec4f(0.0, 0.0, 0.0, 0.0));
            }
            textureStore(dst_visibility, coord, vec4f(0.0, 0.0, 0.0, 0.0));
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
    let step_size = max(block_size, interval_length / f32(max(1u, step_count)));
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
    var dir_buckets: array<vec3f, 16>;
    var dir_opacity: array<f32, 16>;
    for (var init_slot: i32 = 0; init_slot < RC_DIRECTION_COUNT; init_slot = init_slot + 1) {
        dir_buckets[init_slot] = vec3f(0.0, 0.0, 0.0);
        dir_opacity[init_slot] = 0.0;
    }
    let expected_samples_per_direction = max(1.0, f32(max(1u, ray_count)) / f32(max(1, active_dir_count)));
    var direction_vector_accum = vec3f(0.0, 0.0, 0.0);
    var direction_energy_accum = 0.0;
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
        var dist = interval_start + step_size * 0.5;
        var hit = false;
        var hit_distance = interval_end;

        loop {
            if (dist > interval_end) {
                break;
            }
            let sample_pos = origin + dir * dist;
            let b = block_coord_from_world(sample_pos, block_size);
            let material = material_at_block(b.x, b.y, b.z);
            if (material != AIR) {
                let color = material_rgb(material);
                var facing = 1.0;
                var sky_visibility = far_hit_sky_visibility;
                var hit_normal = normalize(-dir);
                var open_hemi = pow(max(0.0, hit_normal.y), 1.5);
                var sun_term = max(0.0, dot(hit_normal, sun_dir));
                if (!cheap_hit_shading) {
                    hit_normal = estimate_hit_normal(b.x, b.y, b.z, dir);
                    facing = saturate(-dot(hit_normal, dir));
                    sun_term = max(0.0, dot(hit_normal, sun_dir));
                    sky_visibility = sky_visibility_at_block(b.x, b.y, b.z);
                    open_hemi = pow(max(0.0, hit_normal.y), 1.5);
                }

                let sky_open = saturate(sky_visibility);
                let cave_gate = pow(sky_open, 1.35);
                let interval_t = saturate((dist - interval_start) / max(interval_length, 0.0001));
                let falloff = 1.0 - interval_t;
                let occluded_bounce = (1.0 - cave_gate) * (indirect_floor + 0.018) * (0.65 + 0.35 * facing) * (0.55 + 0.45 * falloff);
                let ambient_sky = mix(0.055, (0.10 + 0.90 * open_hemi) * 0.30, cave_gate);
                let direct_sun = cave_gate * sun_term * direct_sun_strength * direct_sun_scale * 0.55;
                let range_term = 0.30 + 0.70 * falloff;
                let facing_term = 0.40 + 0.60 * facing;
                let bounce_scale = (indirect_floor + occluded_bounce + ambient_sky + direct_sun) * range_term * facing_term;
                let hit_slot = rc_direction_slot(dir, active_dir_count);
                let local_direction_history = saturate(dir_opacity[hit_slot] / expected_samples_per_direction);
                let feedback_transmittance_gate = 1.0 - local_direction_history * 0.65;
                let outgoing_to_probe = normalize(-dir);
                let hit_feedback_pos = sample_pos + hit_normal * (block_size * 0.75);
                let history_gate = saturate(rc.light_dir.w);
                let previous_bounce_raw = sample_src_world_rc_bounce(cascade_index, hit_feedback_pos, hit_normal, outgoing_to_probe, resolution_i, active_dir_count);
                let previous_bounce = rc_clamp_luma(previous_bounce_raw, RC_FEEDBACK_MAX_LUMA);
                let bounced_radiance_raw = previous_bounce * color * RC_BOUNCE_FEEDBACK_STRENGTH * history_gate * feedback_transmittance_gate * (0.25 + 0.75 * range_term) * (0.35 + 0.65 * facing);
                let bounced_radiance = rc_clamp_luma(bounced_radiance_raw, RC_FEEDBACK_MAX_LUMA);
                let direct_hit_radiance = rc_clamp_luma(color * bounce_scale, RC_INTERVAL_MAX_LUMA);
                let hit_radiance = rc_clamp_luma(direct_hit_radiance + bounced_radiance, RC_INTERVAL_MAX_LUMA);
                accum = accum + hit_radiance;
                let hit_energy = max(max(hit_radiance.r, hit_radiance.g), hit_radiance.b) * (0.35 + 0.65 * facing);
                dir_buckets[hit_slot] = dir_buckets[hit_slot] + hit_radiance;
                let hit_interval_opacity = saturate((0.35 + 0.65 * facing) * (0.35 + 0.65 * (1.0 - interval_t)));
                dir_opacity[hit_slot] = dir_opacity[hit_slot] + hit_interval_opacity;
                direction_vector_accum = direction_vector_accum + normalize(dir) * hit_energy;
                direction_energy_accum = direction_energy_accum + hit_energy;
                hit_sky_visibility_accum = hit_sky_visibility_accum + sky_visibility * (0.35 + 0.65 * facing);
                hit_count = hit_count + 1.0;
                hit = true;
                hit_distance = dist;
                break;
            }
            dist = dist + step_size;
        }

        if (!hit) {
            let current_cascade = u32(rc.meta0.y + 0.5);
            var merged_from_far = false;
            if (current_cascade + 1u < 4u) {
                let merge_world_pos = origin + dir * (interval_end + step_size * 0.5);
                let far_active_dir_count = rc_active_direction_count(current_cascade + 1u);
                let current_merge_slot = rc_direction_slot(dir, active_dir_count);
                let far_footprint_radius = max(block_size * 1.5, interval_length * mix(0.030, 0.055, f32(current_cascade) / 3.0));
                let far_probe = sample_src_world_rc_angular_spatial(current_cascade + 1u, merge_world_pos, resolution_i, dir, far_active_dir_count, far_footprint_radius);
                let far_vis = sample_src_world_rc_vis(current_cascade + 1u, merge_world_pos, resolution_i);
                let local_merge_visibility = local_interval_visibility(
                    origin,
                    dir,
                    interval_start + block_size * 0.35,
                    interval_end + step_size * 0.25,
                    block_size,
                );
                let visibility_gate = smoothstep(0.10, 0.96, local_merge_visibility);
                let far_conf = smoothstep(0.025, 0.22, saturate(far_probe.a)) * visibility_gate;
                if (far_conf > 0.020) {
                    let far_sky_open = saturate(far_vis.w) * visibility_gate;
                    let merge_scale = mix(0.66, 0.36, f32(current_cascade) / 3.0);
                    let local_interval_opacity = saturate(dir_opacity[current_merge_slot] / expected_samples_per_direction);
                    let open_interval = 1.0 - local_interval_opacity;
                    let continuity_feather = clamp(RC_MERGE_CONTINUITY_FEATHER, 0.04, 0.48);
                    let interval_transmittance = smootherstep01(smoothstep(0.04, 0.92, open_interval)) * visibility_gate;
                    let continuity_gate = smoothstep(0.0, continuity_feather, interval_transmittance);
                    let merged_radiance = rc_clamp_luma(far_probe.rgb * far_conf * merge_scale * interval_transmittance * continuity_gate, RC_FAR_MERGE_MAX_LUMA);
                    accum = accum + merged_radiance;
                    dir_buckets[current_merge_slot] = dir_buckets[current_merge_slot] + merged_radiance;
                    dir_opacity[current_merge_slot] = dir_opacity[current_merge_slot] + far_conf * interval_transmittance * 0.70;
                    let merged_energy = max(max(merged_radiance.r, merged_radiance.g), merged_radiance.b);
                    direction_vector_accum = direction_vector_accum + normalize(dir) * merged_energy;
                    direction_energy_accum = direction_energy_accum + merged_energy;
                    sky_count = sky_count + far_sky_open * far_conf * 0.35;
                    merged_from_far = true;
                }
            }
            if (!merged_from_far) {
                let sky_axis = max(0.0, dir.y);
                let sky_ray_weight = (0.04 + 0.96 * (sky_axis * sky_axis)) * pow(max(0.0, origin_sky_visibility), 1.40);
                // Keep sky misses mostly as a visibility signal, but allow a broader,
                // softer contribution so open terrain and cave mouths do not fall off
                // into black while avoiding the harsh cave-mouth hotspots.
                let sky_fill = mix(0.08, 0.13, saturate(origin_sky_visibility));
                let sky_radiance = rc_clamp_luma(sky_rgb * sky_ray_weight * sky_fill, RC_INTERVAL_MAX_LUMA);
                accum = accum + sky_radiance;
                let sky_slot = rc_direction_slot(dir, active_dir_count);
                dir_buckets[sky_slot] = dir_buckets[sky_slot] + sky_radiance;
                dir_opacity[sky_slot] = dir_opacity[sky_slot] + sky_ray_weight * 0.65;
                let sky_energy = max(max(sky_radiance.r, sky_radiance.g), sky_radiance.b);
                direction_vector_accum = direction_vector_accum + normalize(dir) * sky_energy;
                direction_energy_accum = direction_energy_accum + sky_energy;
                sky_count = sky_count + sky_ray_weight;
            }
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
    let temporal_history_weight = saturate(rc.light_dir.w);
    let hit_fraction = hit_count * inv_ray_count;
    let sky_fraction = min(1.0, sky_count * inv_ray_count);
    var observed_sky_access = 0.0;
    if (hit_count > 0.0) {
        observed_sky_access = hit_sky_visibility_accum / hit_count;
    }
    observed_sky_access = max(observed_sky_access, sky_fraction);
    let probe_sky_access = saturate(max(
        0.58 * origin_sky_visibility + 0.42 * observed_sky_access,
        sky_fraction * 0.45
    ));
    let open_probe_validity_floor = smoothstep(0.18, 0.62, probe_sky_access) * 0.16;
    let valid_fraction = max(
        saturate(hit_fraction + 0.75 * sky_fraction) * solid_radiance_scale,
        open_probe_validity_floor * solid_radiance_scale
    );
    let direction_len = length(direction_vector_accum);
    let dominant_dir = select(vec3f(0.0, 1.0, 0.0), direction_vector_accum / max(direction_len, 0.0001), direction_len > 0.0001);
    let directional_anisotropy = saturate(direction_len / max(direction_energy_accum, 0.0001));
    let direction_descriptor = vec4f(dominant_dir * directional_anisotropy, probe_sky_access * valid_fraction);

    let prev_descriptor = sample_src_world_rc_vis(cascade_index, origin, resolution_i);
    let descriptor_has_history = temporal_history_weight > 0.5 && (prev_descriptor.w > 0.001 || length(prev_descriptor.xyz) > 0.001);
    let descriptor_new_weight = select(1.0, clamp(RC_TEMPORAL_ALPHA * 1.20, 0.0, 1.0), descriptor_has_history && valid_fraction > 0.001);
    let temporal_descriptor = normalize_direction_descriptor(mix(prev_descriptor, direction_descriptor, descriptor_new_weight));
    let out_descriptor = vec4f(temporal_descriptor.xyz, max(direction_descriptor.w, temporal_descriptor.w));

    for (var store_slot: i32 = 0; store_slot < RC_DIRECTION_COUNT; store_slot = store_slot + 1) {
        let active_scale = select(0.0, 1.0, store_slot < active_dir_count);
        let directional_interval_opacity = saturate(dir_opacity[store_slot] / expected_samples_per_direction);
        // v10: this atlas stores one radiance value per direction slot. The old
        // code divided every slot by the total ray count, which made each bucket
        // roughly 1/active_dir_count as bright as it should be. That starved true
        // indirect bounce, especially in caves where there is no sky baseline to
        // hide the weak RC radiance. Normalize by the expected rays that map to
        // this slot instead, then let alpha/confidence control validity.
        let directional_norm = 1.0 / max(expected_samples_per_direction, 1.0);
        let traced_rgb = rc_clamp_luma(
            dir_buckets[store_slot] * directional_norm * solid_radiance_scale * active_scale,
            RC_INTERVAL_MAX_LUMA,
        );
        let traced_alpha = valid_fraction * directional_interval_opacity * active_scale;
        let previous = sample_src_world_rc_dir(cascade_index, origin, resolution_i, store_slot);
        let previous_alpha = saturate(previous.a) * active_scale;
        let has_history = temporal_history_weight > 0.5 && previous_alpha > 0.001 && traced_alpha > 0.001;
        let new_weight = select(1.0, RC_TEMPORAL_ALPHA, has_history);
        let temporal_rgb = rc_clamp_luma(mix(previous.rgb, traced_rgb, new_weight), RC_INTERVAL_MAX_LUMA);
        let temporal_alpha = max(traced_alpha, mix(previous_alpha, traced_alpha, new_weight));
        textureStore(
            dst_volume,
            vec3i(coord.x + store_slot * resolution_i, coord.y, coord.z),
            vec4f(temporal_rgb, temporal_alpha),
        );
    }
    textureStore(dst_visibility, coord, out_descriptor);
}
