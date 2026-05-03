from __future__ import annotations

from . import render_constants as render_consts
from .shader_loader import load_shader_text

INDIRECT_DRAW_COMMAND_STRIDE = render_consts.INDIRECT_DRAW_COMMAND_STRIDE
GPU_VISIBILITY_WORKGROUP_SIZE = render_consts.GPU_VISIBILITY_WORKGROUP_SIZE
MESH_OUTPUT_FREERANGE_SCAN_LIMIT = render_consts.MESH_OUTPUT_FREERANGE_SCAN_LIMIT
MESH_VISIBILITY_RECORD_DTYPE = render_consts.MESH_VISIBILITY_RECORD_DTYPE
VOXEL_SURFACE_EXPAND_BIND_GROUP_CACHE_LIMIT = render_consts.VOXEL_SURFACE_EXPAND_BIND_GROUP_CACHE_LIMIT

HUD_FONT_SCALE = render_consts.HUD_FONT_SCALE
HUD_FONT_CHAR_WIDTH = render_consts.HUD_FONT_CHAR_WIDTH
RADIANCE_CASCADES_BASE_RAY_COUNT = render_consts.RADIANCE_CASCADES_BASE_RAY_COUNT
RADIANCE_CASCADES_RAY_COUNT_DECAY = render_consts.RADIANCE_CASCADES_RAY_COUNT_DECAY
RADIANCE_CASCADES_CASCADE_COUNT = render_consts.RADIANCE_CASCADES_CASCADE_COUNT
RADIANCE_CASCADES_STEPS_PER_CASCADE = render_consts.RADIANCE_CASCADES_STEPS_PER_CASCADE
RADIANCE_CASCADES_MERGE_OVERLAP = render_consts.RADIANCE_CASCADES_MERGE_OVERLAP
RADIANCE_CASCADES_MERGE_STRENGTH = render_consts.RADIANCE_CASCADES_MERGE_STRENGTH
HUD_FONT_CHAR_HEIGHT = render_consts.HUD_FONT_CHAR_HEIGHT
HUD_PANEL_PADDING = render_consts.HUD_PANEL_PADDING
HUD_LINE_SPACING = render_consts.HUD_LINE_SPACING
HUD_GLYPH_SPACING = render_consts.HUD_GLYPH_SPACING
PROFILE_REPORT_INTERVAL = render_consts.PROFILE_REPORT_INTERVAL
FRAME_BREAKDOWN_SAMPLE_WINDOW = render_consts.FRAME_BREAKDOWN_SAMPLE_WINDOW
SWAPCHAIN_MAX_FPS = render_consts.SWAPCHAIN_MAX_FPS
SWAPCHAIN_USE_VSYNC = render_consts.SWAPCHAIN_USE_VSYNC
SPRINT_FLY_SPEED = render_consts.SPRINT_FLY_SPEED


def build_tile_merge_shader(merged_tile_max_chunks: int) -> str:
    source_bindings = "\n".join(
        f"@group(0) @binding({index}) var<storage, read> src_{index}: VertexBuffer;"
        for index in range(merged_tile_max_chunks)
    )
    source_cases = "\n".join(
        f"        case {index}u: {{ return src_{index}.values[local_component]; }}"
        for index in range(merged_tile_max_chunks)
    )
    return f"""
struct MergeMeta {{
    vertex_count: u32,
    dst_first_vertex: u32,
    pad0: u32,
    pad1: u32,
}}

struct MergeMetaBuffer {{
    values: array<MergeMeta>,
}}

struct MergeParams {{
    chunk_count: u32,
    total_vertices: u32,
    pad0: u32,
    pad1: u32,
}}

struct VertexBuffer {{
    values: array<f32>,
}}

{source_bindings}
@group(0) @binding({merged_tile_max_chunks}) var<storage, read> merge_meta: MergeMetaBuffer;
@group(0) @binding({merged_tile_max_chunks + 1}) var<uniform> merge_params: MergeParams;
@group(0) @binding({merged_tile_max_chunks + 2}) var<storage, read_write> merged_vertices: VertexBuffer;

fn read_source_component(chunk_index: u32, local_component: u32) -> f32 {{
    switch (chunk_index) {{
{source_cases}
        default: {{
            return src_0.values[local_component];
        }}
    }}
}}

@compute @workgroup_size(64)
fn combine_main(@builtin(global_invocation_id) gid: vec3u) {{
    if (gid.x >= merge_params.total_vertices) {{
        return;
    }}

    var chunk_index: u32 = 0u;
    var local_vertex: u32 = gid.x;
    var i: u32 = 0u;
    loop {{
        if (i >= merge_params.chunk_count) {{
            return;
        }}
        let vertex_count = merge_meta.values[i].vertex_count;
        if (local_vertex < vertex_count) {{
            chunk_index = i;
            break;
        }}
        local_vertex -= vertex_count;
        i += 1u;
    }}

    let dst_first_vertex = merge_meta.values[chunk_index].dst_first_vertex;
    let dst_first_component = (dst_first_vertex + local_vertex) * 9u;
    let src_first_component = local_vertex * 9u;
    for (var component = 0u; component < 9u; component += 1u) {{
        merged_vertices.values[dst_first_component + component] = read_source_component(chunk_index, src_first_component + component);
    }}
}}
"""


def _replace_all(shader: str, replacements: dict[str, str]) -> str:
    for token, value in replacements.items():
        shader = shader.replace(token, value)
    return shader


TILE_MERGE_SHADER = build_tile_merge_shader(4 * 4)
COMPUTE_SHADER = load_shader_text("compute.wgsl")
RENDER_SHADER = load_shader_text("render.wgsl")
GI_GBUFFER_SHADER = load_shader_text("gi_gbuffer.wgsl")
GI_CASCADE_SHADER = _replace_all(
    load_shader_text("gi_cascade.wgsl"),
    {
        "__MAX_CASCADES__": str(max(1, int(RADIANCE_CASCADES_CASCADE_COUNT))),
        "__MAX_RAYS__": str(max(1, int(RADIANCE_CASCADES_BASE_RAY_COUNT))),
        "__MAX_STEPS__": str(max(4, int(RADIANCE_CASCADES_STEPS_PER_CASCADE))),
        "__BASE_RAY_COUNT__": f"{float(RADIANCE_CASCADES_BASE_RAY_COUNT):.4f}",
        "__RAY_COUNT_DECAY__": f"{float(RADIANCE_CASCADES_RAY_COUNT_DECAY):.4f}",
    },
)
GI_COMPOSE_SHADER = _replace_all(
    load_shader_text("gi_compose.wgsl"),
    {
        "__PROBE_VISIBILITY_BIAS__": f"{float(render_consts.WORLDSPACE_RC_VISIBILITY_BIAS):.4f}",
        "__PROBE_VISIBILITY_VARIANCE_BIAS__": f"{float(render_consts.WORLDSPACE_RC_VISIBILITY_VARIANCE_BIAS):.4f}",
        "__PROBE_VISIBILITY_SHARPNESS__": f"{float(render_consts.WORLDSPACE_RC_VISIBILITY_SHARPNESS):.4f}",
        "__PROBE_MIN_HIT_FRACTION__": f"{float(render_consts.WORLDSPACE_RC_MIN_HIT_FRACTION):.4f}",
        "__PROBE_BACKFACE_SOFTNESS__": f"{float(render_consts.WORLDSPACE_RC_BACKFACE_SOFTNESS):.4f}",
        "__PROBE_CASCADE_BLEND_EDGE_START__": f"{float(render_consts.WORLDSPACE_RC_CASCADE_BLEND_EDGE_START):.4f}",
        "__PROBE_CASCADE_BLEND_EDGE_END__": f"{float(render_consts.WORLDSPACE_RC_CASCADE_BLEND_EDGE_END):.4f}",
        "__PROBE_CASCADE_BLEND_MIN_WEIGHT__": f"{float(render_consts.WORLDSPACE_RC_CASCADE_BLEND_MIN_WEIGHT):.4f}",
        "__PROBE_CASCADE_BLEND_CONFIDENCE_SCALE__": f"{float(render_consts.WORLDSPACE_RC_CASCADE_BLEND_CONFIDENCE_SCALE):.4f}",
        "__PROBE_CASCADE_BLEND_FAR_BIAS__": f"{float(render_consts.WORLDSPACE_RC_CASCADE_BLEND_FAR_BIAS):.4f}",
        "__PROBE_CAVE_MIN_LIGHT__": f"{float(render_consts.WORLDSPACE_RC_CAVE_MIN_LIGHT):.4f}",
        "__PROBE_CAVE_SKY_POWER__": f"{float(render_consts.WORLDSPACE_RC_CAVE_SKY_POWER):.4f}",
        "__PROBE_CAVE_DARKENING__": f"{float(render_consts.WORLDSPACE_RC_CAVE_DARKENING):.4f}",
        "__SKY_HORIZON_R__": f"{float(render_consts.RADIANCE_CASCADES_SKY_HORIZON_RGB[0]):.4f}",
        "__SKY_HORIZON_G__": f"{float(render_consts.RADIANCE_CASCADES_SKY_HORIZON_RGB[1]):.4f}",
        "__SKY_HORIZON_B__": f"{float(render_consts.RADIANCE_CASCADES_SKY_HORIZON_RGB[2]):.4f}",
        "__SKY_ZENITH_R__": f"{float(render_consts.RADIANCE_CASCADES_SKY_ZENITH_RGB[0]):.4f}",
        "__SKY_ZENITH_G__": f"{float(render_consts.RADIANCE_CASCADES_SKY_ZENITH_RGB[1]):.4f}",
        "__SKY_ZENITH_B__": f"{float(render_consts.RADIANCE_CASCADES_SKY_ZENITH_RGB[2]):.4f}",
        "__SKY_GROUND_R__": f"{float(render_consts.RADIANCE_CASCADES_SKY_GROUND_RGB[0]):.4f}",
        "__SKY_GROUND_G__": f"{float(render_consts.RADIANCE_CASCADES_SKY_GROUND_RGB[1]):.4f}",
        "__SKY_GROUND_B__": f"{float(render_consts.RADIANCE_CASCADES_SKY_GROUND_RGB[2]):.4f}",
        "__SKY_SUN_R__": f"{float(render_consts.RADIANCE_CASCADES_SKY_SUN_GLOW_RGB[0]):.4f}",
        "__SKY_SUN_G__": f"{float(render_consts.RADIANCE_CASCADES_SKY_SUN_GLOW_RGB[1]):.4f}",
        "__SKY_SUN_B__": f"{float(render_consts.RADIANCE_CASCADES_SKY_SUN_GLOW_RGB[2]):.4f}",
        "__SKY_SUN_DIR_X__": f"{float(render_consts.LIGHT_DIRECTION[0]):.4f}",
        "__SKY_SUN_DIR_Y__": f"{float(render_consts.LIGHT_DIRECTION[1]):.4f}",
        "__SKY_SUN_DIR_Z__": f"{float(render_consts.LIGHT_DIRECTION[2]):.4f}",
    },
)
GI_POSTPROCESS_SHADER = GI_COMPOSE_SHADER

WORLDSPACE_RC_UPDATE_PARAMS_FLOATS = 24
WORLDSPACE_RC_UPDATE_PARAMS_BYTES = WORLDSPACE_RC_UPDATE_PARAMS_FLOATS * 4
WORLDSPACE_RC_TRACE_SHADER = _replace_all(
    load_shader_text("worldspace_rc_trace.wgsl"),
    {
        "__SKY_VISIBILITY_STEPS__": str(max(1, int(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_STEPS))),
        "__SKY_VISIBILITY_STEP_BLOCKS__": str(max(1, int(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_STEP_BLOCKS))),
        "__SKY_VISIBILITY_SIDE_WEIGHT__": f"{float(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_SIDE_WEIGHT):.8f}",
        "__SKY_VISIBILITY_APERTURE_RADIUS_BLOCKS__": str(max(1, int(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_APERTURE_RADIUS_BLOCKS))),
        "__SKY_VISIBILITY_APERTURE_POWER__": f"{float(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_APERTURE_POWER):.8f}",
        "__SKY_VISIBILITY_MIN_APERTURE__": f"{float(render_consts.WORLDSPACE_RC_SKY_VISIBILITY_MIN_APERTURE):.8f}",
        "__RC_TEMPORAL_ALPHA__": f"{float(render_consts.WORLDSPACE_RC_TEMPORAL_BLEND_ALPHA):.8f}",
        "__RC_BOUNCE_FEEDBACK_STRENGTH__": f"{float(getattr(render_consts, 'WORLDSPACE_RC_BOUNCE_FEEDBACK_STRENGTH', 0.18)):.8f}",
        "__RC_MERGE_CONTINUITY_FEATHER__": f"{float(getattr(render_consts, 'WORLDSPACE_RC_MERGE_CONTINUITY_FEATHER', 0.22)):.8f}",
        "__RC_DDA_MAX_VISITS__": str(max(1, int(getattr(render_consts, 'WORLDSPACE_RC_DDA_MAX_VISITS', 64)))),
    },
)
WORLDSPACE_RC_FILTER_SHADER = load_shader_text("worldspace_rc_filter.wgsl")
FINAL_BLIT_SHADER = load_shader_text("final_blit.wgsl")
HUD_SHADER = load_shader_text("hud.wgsl")
VOXEL_SURFACE_EXPAND_SHADER = load_shader_text("voxel_surface_expand.wgsl")
VOXEL_MESH_BATCH_SHADER = load_shader_text("voxel_mesh_batch.wgsl")
GPU_VISIBILITY_SHADER = load_shader_text("gpu_visibility.wgsl")
