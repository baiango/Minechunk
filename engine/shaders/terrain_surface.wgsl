
struct TerrainParams {
    sample_origin: vec4f,
    chunk_and_sample: vec4i,
    seed_and_pad: vec4u,
}

struct TerrainBatchParams {
    sample_size: u32,
    chunk_size: u32,
    height_limit: u32,
    seed: u32,
}

@group(0) @binding(0) var<storage, read_write> heights: array<u32>;
@group(0) @binding(1) var<storage, read_write> materials: array<u32>;
@group(0) @binding(2) var<uniform> params: TerrainParams;

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

fn terrain_sample(x: f32, z: f32, seed: u32, height_limit: u32) -> vec2u {
    let terrain_frequency_scale = 0.3;

    let sample_x = x;
    let sample_z = z;
    let broad = value_noise_2d(sample_x, sample_z, seed + 11u, 0.0009765625 * terrain_frequency_scale);
    let ridge = value_noise_2d(sample_x, sample_z, seed + 23u, 0.00390625 * terrain_frequency_scale);
    let detail = value_noise_2d(sample_x, sample_z, seed + 47u, 0.010416667 * terrain_frequency_scale);
    let micro = value_noise_2d(sample_x, sample_z, seed + 71u, 0.020833334 * terrain_frequency_scale);
    let nano = value_noise_2d(sample_x, sample_z, seed + 97u, 0.041666668 * terrain_frequency_scale);

    let upper_bound = height_limit - 1u;
    let upper_bound_f = f32(upper_bound);
    let normalized_height = 24.0 + broad * 11.0 + ridge * 8.0 + detail * 4.5 + micro * 1.75 + nano * 0.75;
    let height_scale = select(1.0, upper_bound_f / 50.0, upper_bound > 0u);
    var height_f = normalized_height * height_scale;
    if (height_f < 4.0) {
        height_f = 4.0;
    }
    if (height_f > upper_bound_f) {
        height_f = upper_bound_f;
    }
    let height_i = u32(height_f);

    let sand_threshold = max(4u, u32(f32(height_limit) * 0.18));
    let stone_threshold = max(sand_threshold + 6u, u32(f32(height_limit) * 0.58));
    let snow_threshold = max(stone_threshold + 6u, u32(f32(height_limit) * 0.82));

    var material = 4u;
    if (height_i >= snow_threshold) {
        material = 6u;
    } else if (height_i <= sand_threshold) {
        material = 5u;
    } else if (height_i >= stone_threshold && (detail + micro * 0.5 + nano * 0.35) > 0.10) {
        material = 2u;
    }
    return vec2u(height_i, material);
}

@compute @workgroup_size(1, 1, 1)
fn sample_surface_profile_at_main() {
    let result = terrain_sample(
        params.sample_origin.x,
        params.sample_origin.y,
        params.seed_and_pad.x,
        u32(params.sample_origin.w),
    );
    heights[0u] = result.x;
    materials[0u] = result.y;
}

@compute @workgroup_size(8, 8, 1)
fn fill_chunk_surface_grids_main(@builtin(global_invocation_id) gid: vec3u) {
    let sample_size = u32(params.chunk_and_sample.z);
    if (gid.x >= sample_size || gid.y >= sample_size) {
        return;
    }

    let chunk_x = params.chunk_and_sample.x;
    let chunk_z = params.chunk_and_sample.y;
    let chunk_size = max(1, params.chunk_and_sample.w);
    let origin_x = chunk_x * chunk_size - 1i;
    let origin_z = chunk_z * chunk_size - 1i;
    let world_x = f32(origin_x + i32(gid.x));
    let world_z = f32(origin_z + i32(gid.y));
    let result = terrain_sample(world_x, world_z, params.seed_and_pad.x, u32(params.sample_origin.w));

    let cell_index = gid.y * sample_size + gid.x;
    heights[cell_index] = result.x;
    materials[cell_index] = result.y;
}

@group(0) @binding(0) var<storage, read_write> batch_heights: array<u32>;
@group(0) @binding(1) var<storage, read_write> batch_materials: array<u32>;
@group(0) @binding(2) var<storage, read> batch_coords: array<vec4i>;
@group(0) @binding(3) var<uniform> batch_params: TerrainBatchParams;

@compute @workgroup_size(8, 8, 1)
fn fill_chunk_surface_batch_main(@builtin(global_invocation_id) gid: vec3u) {
    let sample_size = batch_params.sample_size;
    if (gid.x >= sample_size || gid.y >= sample_size) {
        return;
    }

    let chunk_index = gid.z;
    let coord = batch_coords[chunk_index];
    let chunk_x = coord.x;
    let chunk_z = coord.z;
    let chunk_size = i32(batch_params.chunk_size);
    let origin_x = chunk_x * chunk_size - 1i;
    let origin_z = chunk_z * chunk_size - 1i;
    let world_x = f32(origin_x + i32(gid.x));
    let world_z = f32(origin_z + i32(gid.y));
    let result = terrain_sample(world_x, world_z, batch_params.seed, batch_params.height_limit);

    let cell_index = chunk_index * sample_size * sample_size + gid.y * sample_size + gid.x;
    batch_heights[cell_index] = result.x;
    batch_materials[cell_index] = result.y;
}
