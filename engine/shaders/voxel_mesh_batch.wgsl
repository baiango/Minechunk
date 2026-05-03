
struct BatchParams {
    counts_and_flags: vec4u,
    world_scale_and_pad: vec4f,
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
    values: array<f32>,
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

fn face_color(material: u32, height: u32, _shade: f32) -> vec3f {
    return material_color(material, height);
}

fn emit_vertex(position: vec3f, normal: vec3f, color: vec3f, slot: u32) {
    let base = slot * 9u;
    vertices.values[base + 0u] = position.x;
    vertices.values[base + 1u] = position.y;
    vertices.values[base + 2u] = position.z;
    vertices.values[base + 3u] = normal.x;
    vertices.values[base + 4u] = normal.y;
    vertices.values[base + 5u] = normal.z;
    vertices.values[base + 6u] = color.x;
    vertices.values[base + 7u] = color.y;
    vertices.values[base + 8u] = color.z;
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
    var face_base = base;

    if (blocks.values[cell_index + plane] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x0, y1, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y1, z1),
            vec3f(x0, y1, z1),
            vec3f(0.0, 1.0, 0.0),
            top,
            top,
            top,
            top,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index - plane] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z0),
            vec3f(x0, y0, z1),
            vec3f(x1, y0, z1),
            vec3f(x1, y0, z0),
            vec3f(0.0, -1.0, 0.0),
            bottom,
            bottom,
            bottom,
            bottom,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index + 1u] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x1, y0, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y1, z1),
            vec3f(x1, y0, z1),
            vec3f(1.0, 0.0, 0.0),
            east,
            east,
            east,
            east,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index - 1u] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z0),
            vec3f(x0, y0, z1),
            vec3f(x0, y1, z1),
            vec3f(x0, y1, z0),
            vec3f(-1.0, 0.0, 0.0),
            west,
            west,
            west,
            west,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index + sample_size] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z1),
            vec3f(x1, y0, z1),
            vec3f(x1, y1, z1),
            vec3f(x0, y1, z1),
            vec3f(0.0, 0.0, 1.0),
            south,
            south,
            south,
            south,
        );
        face_base = face_base + 6u;
    }
    if (blocks.values[cell_index - sample_size] == 0u) {
        emit_quad_at(
            face_base,
            vec3f(x0, y0, z0),
            vec3f(x0, y1, z0),
            vec3f(x1, y1, z0),
            vec3f(x1, y0, z0),
            vec3f(0.0, 0.0, -1.0),
            north,
            north,
            north,
            north,
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
