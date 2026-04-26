
#include <metal_stdlib>
using namespace metal;

struct TerrainParams { float4 sample_origin; int4 chunk_and_sample; uint4 seed_and_pad; };
struct TerrainBatchParams { uint sample_size; uint chunk_size; uint height_limit; uint seed; };

inline uint mix_u32(uint value) {
    uint x = value;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

inline float hash2(int ix, int iy, uint seed) {
    uint h = uint(ix) * 0x9e3779b9u;
    h = h ^ (uint(iy) * 0x85ebca6bu);
    h = h ^ (seed * 0xc2b2ae35u);
    h = mix_u32(h);
    return float(h & 0x00ffffffu) / 16777215.0f;
}
inline float fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
inline float lerp_f(float a, float b, float t) { return a + (b - a) * t; }
inline float value_noise_2d(float x, float y, uint seed, float frequency) {
    float px = x * frequency; float py = y * frequency;
    float x0 = floor(px); float y0 = floor(py);
    float xf = px - x0; float yf = py - y0;
    int ix0 = int(x0); int iy0 = int(y0); int ix1 = ix0 + 1; int iy1 = iy0 + 1;
    float u = fade(xf); float v = fade(yf);
    float nx0 = lerp_f(hash2(ix0, iy0, seed), hash2(ix1, iy0, seed), u);
    float nx1 = lerp_f(hash2(ix0, iy1, seed), hash2(ix1, iy1, seed), u);
    return lerp_f(nx0, nx1, v) * 2.0f - 1.0f;
}
inline uint2 terrain_sample(float x, float z, uint seed, uint height_limit) {
    constexpr float terrain_frequency_scale = 0.3f;
    float broad = value_noise_2d(x, z, seed + 11u, 0.0009765625f * terrain_frequency_scale);
    float ridge = value_noise_2d(x, z, seed + 23u, 0.00390625f * terrain_frequency_scale);
    float detail = value_noise_2d(x, z, seed + 47u, 0.010416667f * terrain_frequency_scale);
    float micro = value_noise_2d(x, z, seed + 71u, 0.020833334f * terrain_frequency_scale);
    float nano = value_noise_2d(x, z, seed + 97u, 0.041666668f * terrain_frequency_scale);
    uint upper_bound = height_limit - 1u;
    float upper_bound_f = float(upper_bound);
    float normalized_height = 24.0f + broad * 11.0f + ridge * 8.0f + detail * 4.5f + micro * 1.75f + nano * 0.75f;
    float height_scale = upper_bound > 0u ? upper_bound_f / 50.0f : 1.0f;
    float height_f = clamp(normalized_height * height_scale, 4.0f, upper_bound_f);
    uint height_i = uint(height_f);
    uint sand_threshold = max(4u, uint(float(height_limit) * 0.18f));
    uint stone_threshold = max(sand_threshold + 6u, uint(float(height_limit) * 0.58f));
    uint snow_threshold = max(stone_threshold + 6u, uint(float(height_limit) * 0.82f));
    uint material = 4u;
    if (height_i >= snow_threshold) material = 6u;
    else if (height_i <= sand_threshold) material = 5u;
    else if (height_i >= stone_threshold && (detail + micro * 0.5f + nano * 0.35f) > 0.10f) material = 2u;
    return uint2(height_i, material);
}

kernel void sample_surface_profile_at_main(device uint* heights [[buffer(0)]], device uint* materials [[buffer(1)]], constant TerrainParams& params [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x != 0 || gid.y != 0 || gid.z != 0) return;
    uint2 result = terrain_sample(params.sample_origin.x, params.sample_origin.y, params.seed_and_pad.x, uint(params.sample_origin.w));
    heights[0] = result.x; materials[0] = result.y;
}

kernel void fill_chunk_surface_grids_main(device uint* heights [[buffer(0)]], device uint* materials [[buffer(1)]], constant TerrainParams& params [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint sample_size = uint(params.chunk_and_sample.z);
    if (gid.x >= sample_size || gid.y >= sample_size) return;
    int chunk_x = params.chunk_and_sample.x;
    int chunk_z = params.chunk_and_sample.y;
    int chunk_size = int(params.seed_and_pad.y);
    int origin_x = chunk_x * chunk_size - 1;
    int origin_z = chunk_z * chunk_size - 1;
    uint2 result = terrain_sample(float(origin_x + int(gid.x)), float(origin_z + int(gid.y)), params.seed_and_pad.x, uint(params.sample_origin.w));
    uint cell_index = gid.y * sample_size + gid.x;
    heights[cell_index] = result.x; materials[cell_index] = result.y;
}

kernel void fill_chunk_surface_batch_main(device uint* batch_heights [[buffer(0)]], device uint* batch_materials [[buffer(1)]], device const int4* batch_coords [[buffer(2)]], constant TerrainBatchParams& batch_params [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    uint sample_size = batch_params.sample_size;
    if (gid.x >= sample_size || gid.y >= sample_size) return;
    uint chunk_index = gid.z;
    int4 coord = batch_coords[chunk_index];
    int origin_x = coord.x * int(batch_params.chunk_size) - 1;
    int origin_z = coord.z * int(batch_params.chunk_size) - 1;
    uint2 result = terrain_sample(float(origin_x + int(gid.x)), float(origin_z + int(gid.y)), batch_params.seed, batch_params.height_limit);
    uint cell_index = chunk_index * sample_size * sample_size + gid.y * sample_size + gid.x;
    batch_heights[cell_index] = result.x; batch_materials[cell_index] = result.y;
}
