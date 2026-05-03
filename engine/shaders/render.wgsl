
struct CameraUniform {
    position: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
    proj: vec4f,
}

struct VertexInput {
    @location(0) position: vec4f,
    @location(1) normal: vec4f,
    @location(2) color: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) color: vec3f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

fn face_direction_shade(normal: vec3f) -> f32 {
    let n = normalize(normal);
    let axis = abs(n);
    if (axis.y >= axis.x && axis.y >= axis.z) {
        return select(0.50, 1.0, n.y >= 0.0);
    }
    if (axis.x >= axis.z) {
        return select(0.64, 0.80, n.x >= 0.0);
    }
    return select(0.60, 0.72, n.z >= 0.0);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = input.position.xyz;
    let to_point = world - camera.position.xyz;
    let view_x = dot(to_point, camera.right.xyz);
    let view_y = dot(to_point, camera.up.xyz);
    let view_z = dot(to_point, camera.forward.xyz);

    let focal = camera.proj.x;
    let aspect = camera.proj.y;
    let near = camera.proj.z;
    let far = camera.proj.w;
    let clip_x = view_x * focal / aspect;
    let clip_y = view_y * focal;
    let clip_z = view_z * (far / (far - near)) - (near * far) / (far - near);
    out.position = vec4f(clip_x, clip_y, clip_z, view_z);
    out.normal = normalize(input.normal.xyz);
    out.color = input.color.xyz;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let light = normalize(vec3f(0.35, 0.90, 0.25));
    let diffuse = max(dot(input.normal, light), 0.0);
    let brightness = 0.32 + 0.68 * diffuse;
    return vec4f(input.color * face_direction_shade(input.normal) * brightness, 1.0);
}
