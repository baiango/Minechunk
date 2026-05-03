
struct ChunkParams {
    origin_and_scale: vec4f,
    counts_and_flags: vec4u,
}

struct HeightBuffer {
    values: array<u32>,
}

struct MaterialBuffer {
    values: array<u32>,
}

struct VertexBuffer {
    values: array<f32>,
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
