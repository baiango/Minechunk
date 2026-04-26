
struct CameraUniform {
    position: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
    proj: vec4f,
}

struct VisibilityParams {
    mesh_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct MeshRecord {
    bounds: vec4f,
    draw: vec4u,
}

struct MeshRecordBuffer {
    values: array<MeshRecord>,
}

struct IndirectCommandBuffer {
    values: array<vec4u>,
}

@group(0) @binding(0) var<storage, read> mesh_records: MeshRecordBuffer;
@group(0) @binding(1) var<storage, read_write> indirect_commands: IndirectCommandBuffer;
@group(0) @binding(2) var<uniform> camera: CameraUniform;
@group(0) @binding(3) var<uniform> params: VisibilityParams;

fn sphere_visible(bounds: vec4f) -> bool {
    let center = bounds.xyz;
    let radius = bounds.w;
    let relative = center - camera.position.xyz;
    let view_x = dot(relative, camera.right.xyz);
    let view_y = dot(relative, camera.up.xyz);
    let view_z = dot(relative, camera.forward.xyz);
    let focal = camera.proj.x;
    let aspect = camera.proj.y;
    let near = camera.proj.z;
    let far = camera.proj.w;

    if (view_z + radius <= near) {
        return false;
    }
    if (view_z - radius >= far) {
        return false;
    }

    let depth = max(view_z, near);
    let half_width = depth * aspect / focal;
    let half_height = depth / focal;
    if (view_x < -half_width - radius || view_x > half_width + radius) {
        return false;
    }
    if (view_y < -half_height - radius || view_y > half_height + radius) {
        return false;
    }
    return true;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let mesh_index = gid.x;
    if (mesh_index >= params.mesh_count) {
        return;
    }
    let record = mesh_records.values[mesh_index];
    if (sphere_visible(record.bounds)) {
        indirect_commands.values[mesh_index] = record.draw;
    } else {
        indirect_commands.values[mesh_index] = vec4u(0u, 1u, 0u, 0u);
    }
}
