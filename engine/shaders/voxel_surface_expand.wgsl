
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
