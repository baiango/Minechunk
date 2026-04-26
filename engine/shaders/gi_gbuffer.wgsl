
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
    @location(0) world_normal: vec3f,
    @location(1) color: vec3f,
    @location(2) view_z: f32,
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @location(1) gbuffer: vec4f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = input.position.xyz;
    let to_point = world - camera.position.xyz;
    let view_x = dot(to_point, camera.right.xyz);
    let view_y = dot(to_point, camera.up.xyz);
    let view_z = dot(to_point, camera.forward.xyz);

    if (view_z <= camera.proj.z) {
        out.position = vec4f(2.0, 2.0, 2.0, 1.0);
        out.world_normal = normalize(input.normal.xyz);
        out.color = input.color.xyz;
        out.view_z = 0.0;
        return out;
    }

    let focal = camera.proj.x;
    let aspect = camera.proj.y;
    let near = camera.proj.z;
    let far = camera.proj.w;
    let clip_x = view_x * focal / aspect;
    let clip_y = view_y * focal;
    let clip_z = view_z * (far / (far - near)) - (near * far) / (far - near);
    out.position = vec4f(clip_x, clip_y, clip_z, view_z);
    out.world_normal = normalize(input.normal.xyz);
    out.color = input.color.xyz;
    out.view_z = view_z;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.color = vec4f(input.color, 1.0);
    out.gbuffer = vec4f(input.world_normal, input.view_z);
    return out;
}
