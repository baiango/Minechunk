
struct RcUpdateParams {
    min_corner_and_extent: vec4f,
    meta0: vec4f,
    meta1: vec4f,
    light0: vec4f,
    light_dir: vec4f,
    controls: vec4f,
}

const RC_DIRECTION_COUNT: i32 = 16;

fn rc_active_direction_count(cascade_index: u32) -> i32 {
    if (cascade_index == 0u) { return 8; }
    if (cascade_index == 1u) { return 12; }
    return RC_DIRECTION_COUNT;
}

@group(0) @binding(0) var src_volume: texture_3d<f32>;
@group(0) @binding(1) var src_visibility: texture_3d<f32>;
@group(0) @binding(2) var dst_volume: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var dst_visibility: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<uniform> rc: RcUpdateParams;

fn in_bounds(c: vec3i, resolution: i32) -> bool {
    return c.x >= 0 && c.y >= 0 && c.z >= 0 && c.x < resolution && c.y < resolution && c.z < resolution;
}

fn normalize_direction_descriptor(v: vec4f) -> vec4f {
    let len_v = length(v.xyz);
    let dir = select(vec3f(0.0, 1.0, 0.0), v.xyz / max(len_v, 0.0001), len_v > 0.0001);
    return vec4f(dir * saturate(len_v), saturate(v.w));
}

fn filter_rc_luma(rgb: vec3f) -> f32 {
    return dot(max(rgb, vec3f(0.0, 0.0, 0.0)), vec3f(0.2126, 0.7152, 0.0722));
}

fn filter_clamp_luma(rgb: vec3f, max_luma: f32) -> vec3f {
    let safe_rgb = max(rgb, vec3f(0.0, 0.0, 0.0));
    let luma = filter_rc_luma(safe_rgb);
    let scale = min(1.0, max_luma / max(luma, 0.0001));
    return safe_rgb * scale;
}

@compute @workgroup_size(4, 4, 4)
fn filter_main(@builtin(global_invocation_id) gid: vec3u) {
    let resolution_u = max(1u, u32(rc.meta0.x + 0.5));
    if (gid.x >= resolution_u || gid.y >= resolution_u || gid.z >= resolution_u) {
        return;
    }

    let resolution = i32(resolution_u);
    let coord = vec3i(i32(gid.x), i32(gid.y), i32(gid.z));
    let cascade_index = u32(rc.meta0.y + 0.5);
    let active_dir_count = rc_active_direction_count(cascade_index);
    let c0_desplotch_filter = select(0.0, 0.55, cascade_index == 0u);
    let far_filter = max(c0_desplotch_filter, clamp(f32(cascade_index) * 0.45, 0.0, 1.0));

    let raw_visibility = textureLoad(src_visibility, coord, 0);
    let center_weight = mix(0.34, 0.28, far_filter);
    var filtered_visibility = raw_visibility * center_weight;
    var visibility_weight = center_weight;

    // Filter metadata once per probe. Visibility.w is hard sky/open confidence,
    // so the bilateral term still blocks open probes from flooding caves.
    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                let manhattan = abs(dx) + abs(dy) + abs(dz);
                var geometric_weight = 0.0;
                if (manhattan == 1) {
                    geometric_weight = mix(0.160, 0.135, far_filter);
                } else if (manhattan == 2) {
                    geometric_weight = mix(0.050, 0.070, far_filter);
                } else {
                    geometric_weight = mix(0.030, 0.045, far_filter);
                }
                if (geometric_weight <= 0.0001) {
                    continue;
                }

                let nc = coord + vec3i(dx, dy, dz);
                if (in_bounds(nc, resolution)) {
                    let neighbor_visibility = textureLoad(src_visibility, nc, 0);
                    let sky_delta = abs(clamp(neighbor_visibility.w, 0.0, 1.0) - clamp(raw_visibility.w, 0.0, 1.0));
                    let sky_similarity = clamp(1.0 - smoothstep(mix(0.08, 0.12, far_filter), mix(0.55, 0.82, far_filter), sky_delta), 0.0, 1.0);
                    let neighbor_validity = clamp(neighbor_visibility.w, 0.0, 1.0);
                    let neighbor_weight = geometric_weight * sky_similarity * (0.35 + 0.65 * neighbor_validity);
                    filtered_visibility = filtered_visibility + neighbor_visibility * neighbor_weight;
                    visibility_weight = visibility_weight + neighbor_weight;
                }
            }
        }
    }

    filtered_visibility = filtered_visibility / max(visibility_weight, 0.0001);
    let directional_blend = mix(0.62, 0.86, far_filter);
    var out_visibility = mix(raw_visibility, max(filtered_visibility, vec4f(0.0, 0.0, 0.0, 0.0)), directional_blend);
    out_visibility = normalize_direction_descriptor(out_visibility);
    textureStore(dst_visibility, coord, out_visibility);

    let sky_filter_power = max(0.25, rc.light0.w);
    let directional_filter_gate = 0.35 + 0.65 * pow(clamp(raw_visibility.w, 0.0, 1.0), sky_filter_power);

    // Real 3D RC v25: radiance is stored per angular interval in an X-atlas.
    // Filter each interval independently so RGB direction buckets do not collapse
    // back into an isotropic probe value.
    for (var dir_slot: i32 = 0; dir_slot < RC_DIRECTION_COUNT; dir_slot = dir_slot + 1) {
        let atlas_coord = vec3i(coord.x + dir_slot * resolution, coord.y, coord.z);
        if (dir_slot >= active_dir_count) {
            textureStore(dst_volume, atlas_coord, vec4f(0.0, 0.0, 0.0, 0.0));
            continue;
        }
        let raw_volume = textureLoad(src_volume, atlas_coord, 0);
        var filtered_volume = raw_volume * center_weight;
        var volume_weight = center_weight;

        for (var dz2: i32 = -1; dz2 <= 1; dz2 = dz2 + 1) {
            for (var dy2: i32 = -1; dy2 <= 1; dy2 = dy2 + 1) {
                for (var dx2: i32 = -1; dx2 <= 1; dx2 = dx2 + 1) {
                    if (dx2 == 0 && dy2 == 0 && dz2 == 0) {
                        continue;
                    }
                    let manhattan2 = abs(dx2) + abs(dy2) + abs(dz2);
                    var geometric_weight2 = 0.0;
                    if (manhattan2 == 1) {
                        geometric_weight2 = mix(0.160, 0.135, far_filter);
                    } else if (manhattan2 == 2) {
                        geometric_weight2 = mix(0.050, 0.070, far_filter);
                    } else {
                        geometric_weight2 = mix(0.030, 0.045, far_filter);
                    }
                    if (geometric_weight2 <= 0.0001) {
                        continue;
                    }

                    let nc = coord + vec3i(dx2, dy2, dz2);
                    if (in_bounds(nc, resolution)) {
                        let neighbor_visibility = textureLoad(src_visibility, nc, 0);
                        let sky_delta = abs(clamp(neighbor_visibility.w, 0.0, 1.0) - clamp(raw_visibility.w, 0.0, 1.0));
                        let sky_similarity = clamp(1.0 - smoothstep(mix(0.08, 0.12, far_filter), mix(0.55, 0.82, far_filter), sky_delta), 0.0, 1.0);
                        let neighbor_coord = vec3i(nc.x + dir_slot * resolution, nc.y, nc.z);
                        let neighbor_volume = textureLoad(src_volume, neighbor_coord, 0);
                        let neighbor_validity = clamp(neighbor_volume.a, 0.0, 1.0);
                        let neighbor_weight = geometric_weight2 * sky_similarity * (0.25 + 0.75 * neighbor_validity);
                        filtered_volume = filtered_volume + neighbor_volume * neighbor_weight;
                        volume_weight = volume_weight + neighbor_weight;
                    }
                }
            }
        }

        filtered_volume = filtered_volume / max(volume_weight, 0.0001);
        var out_volume = raw_volume * (1.0 - directional_filter_gate) + filtered_volume * directional_filter_gate;
        out_volume = vec4f(filter_clamp_luma(out_volume.rgb, 1.15), clamp(out_volume.a, 0.0, 1.0));
        textureStore(dst_volume, atlas_coord, out_volume);
    }

}
