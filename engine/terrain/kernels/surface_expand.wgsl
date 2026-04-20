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