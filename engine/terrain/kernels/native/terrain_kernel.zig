const std = @import("std");

const AIR: u32 = 0;
const BEDROCK: u32 = 1;
const STONE: u32 = 2;
const DIRT: u32 = 3;
const GRASS: u32 = 4;
const SAND: u32 = 5;
const SNOW: u32 = 6;

const TERRAIN_FREQUENCY_SCALE: f32 = 0.3;
const CAVE_FREQUENCY_SCALE: f32 = 1.0;
const CAVE_DETAIL_FREQUENCY_MULTIPLIER: f32 = 3.0;
const CAVE_DETAIL_WEIGHT: f32 = 0.18;
const CAVE_BEDROCK_CLEARANCE: i64 = 3;
const CAVE_ACTIVE_BAND_MIN: f32 = 0.58;
const CAVE_PRIMARY_THRESHOLD: f32 = 0.66;
const CAVE_VERTICAL_BONUS: f32 = 0.06;
const CAVE_DEPTH_BONUS_SCALE: f32 = 0.0015;
const CAVE_DEPTH_BONUS_MAX: f32 = 0.06;

const ABI_VERSION: u32 = 2;
const MAX_PARALLEL_WORKERS: usize = 16;
const SURFACE_PARALLEL_MIN_COLUMNS: usize = 1024;
const VOXEL_CAVE_PARALLEL_MIN_COLUMNS: usize = 512;
const VOXEL_SIMPLE_PARALLEL_MIN_COLUMNS: usize = 8192;
const SURFACE_SIMD_ENABLED: bool = true;
const PERSISTENT_VOXEL_BATCH_POOL_ENABLED: bool = true;
const MAX_CAVE_Y_TERMS: usize = 512;

const VF = @Vector(4, f32);
const VI = @Vector(4, i32);
const VU = @Vector(4, u32);
const VB = @Vector(4, bool);
const X_OFFSETS_F32 = VF{ 0.0, 1.0, 2.0, 3.0 };

const SurfaceProfile = struct {
    height: u32,
    material: u32,
};

const SurfaceProfile4 = struct {
    heights: VU,
    materials: VU,
};

const CaveYTerm = struct {
    world_y_i32: i32,
    primary_noise_y: CaveNoiseYTerm,
    detail_noise_y: CaveNoiseYTerm,
    threshold_base: f32,
    active: bool,
};

const CaveNoiseYTerm = struct {
    iy0: VI,
    iy1: VI,
    v: VF,
};

const CaveNoiseXTerm = struct {
    ix0: VI,
    ix1: VI,
    u: VF,
};

const CaveNoiseZTerm = struct {
    iz0: VI,
    iz1: VI,
    w: VF,
};

const SurfaceGridWork = struct {
    heights: [*]u32,
    materials: [*]u32,
    start: usize,
    end: usize,
    sample_size: usize,
    origin_x: i64,
    origin_z: i64,
    seed: i64,
    height_limit: i32,
};

const SurfaceGridBatchWork = struct {
    heights: [*]u32,
    materials: [*]u32,
    chunk_xs: [*]const i32,
    chunk_zs: [*]const i32,
    start: usize,
    end: usize,
    sample_size: usize,
    plane_cells: usize,
    chunk_size: i32,
    seed: i64,
    height_limit: i32,
};

const VoxelFromSurfaceWork = struct {
    blocks: [*]u8,
    voxel_materials: [*]u32,
    top_plane: [*]u8,
    bottom_plane: [*]u8,
    surface_heights: [*]const u32,
    surface_materials: [*]const u32,
    start: usize,
    end: usize,
    sample_size: usize,
    plane_cells: usize,
    origin_x: i64,
    origin_z: i64,
    origin_y: i64,
    fill_start_y: i64,
    fill_top_y: i64,
    top_world_y: i64,
    bottom_world_y: i64,
    top_in_bounds: bool,
    bottom_in_bounds: bool,
    seed: i64,
    world_height_limit: i32,
    carve_caves: bool,
    cave_y_terms: ?[*]const CaveYTerm,
    cave_y_terms_len: usize,
};

const VoxelFromSurfaceBatchWork = struct {
    blocks: [*]u8,
    voxel_materials: [*]u32,
    top_plane: [*]u8,
    bottom_plane: [*]u8,
    surface_heights: [*]const u32,
    surface_materials: [*]const u32,
    chunk_xs: [*]const i32,
    chunk_ys: [*]const i32,
    chunk_zs: [*]const i32,
    start: usize,
    end: usize,
    sample_size: usize,
    plane_cells: usize,
    block_cells: usize,
    chunk_count: usize,
    local_height: i32,
    chunk_size: i32,
    seed: i64,
    world_height_limit: i32,
    carve_caves: bool,
    cave_y_terms: ?[*]const CaveYTerm,
    cave_y_terms_len: usize,
};

const VoxelFromSurfaceBatchChunkWork = struct {
    work: VoxelFromSurfaceBatchWork,
    worker_index: usize,
    worker_count: usize,
};

const PersistentVoxelBatchPool = struct {
    initialized: std.atomic.Value(bool) = .init(false),
    init_lock: std.atomic.Value(bool) = .init(false),
    busy: std.atomic.Value(bool) = .init(false),
    shutdown_requested: std.atomic.Value(bool) = .init(false),
    worker_count: std.atomic.Value(usize) = .init(0),
    requested_workers: std.atomic.Value(usize) = .init(0),
    generation: std.atomic.Value(usize) = .init(0),
    completed_workers: std.atomic.Value(usize) = .init(0),
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    job_cond: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,
    done_cond: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,
    threads: [MAX_PARALLEL_WORKERS]std.Thread = undefined,
    work: VoxelFromSurfaceBatchWork = undefined,
};

const PersistentVoxelBatchWorkerContext = struct {
    worker_index: usize,
    start_generation: usize,
};

var persistent_voxel_batch_pool = PersistentVoxelBatchPool{};
var persistent_voxel_batch_worker_contexts: [MAX_PARALLEL_WORKERS]PersistentVoxelBatchWorkerContext = undefined;

const VoxelGeneratedWork = struct {
    blocks: [*]u8,
    voxel_materials: [*]u32,
    start: usize,
    end: usize,
    sample_size: usize,
    plane_cells: usize,
    origin_x: i64,
    origin_z: i64,
    origin_y: i64,
    fill_start_y: i64,
    fill_top_y: i64,
    seed: i64,
    world_height_limit: i32,
    carve_caves: bool,
};

pub export fn minechunk_terrain_abi_version() u32 {
    return ABI_VERSION;
}

pub export fn minechunk_shutdown_terrain_workers() void {
    shutdownPersistentVoxelBatchPool();
}

pub export fn minechunk_surface_profile_at(
    x: f64,
    z: f64,
    seed: i64,
    height_limit: i32,
    out_height: *u32,
    out_material: *u32,
) i32 {
    if (height_limit <= 0) return -1;
    const profile = surfaceProfileAt(@floatCast(x), @floatCast(z), seed, height_limit);
    out_height.* = profile.height;
    out_material.* = profile.material;
    return 0;
}

pub export fn minechunk_terrain_block_material_at(
    world_x: i64,
    world_y: i64,
    world_z: i64,
    seed: i64,
    world_height_limit: i32,
    carve_caves_u8: u8,
) u32 {
    if (world_y < 0 or world_y >= @as(i64, world_height_limit)) return AIR;

    const profile = surfaceProfileAt(
        @as(f32, @floatFromInt(world_x)),
        @as(f32, @floatFromInt(world_z)),
        seed,
        world_height_limit,
    );
    const surface_height = @as(i64, profile.height);
    if (world_y >= surface_height) return AIR;
    if (carve_caves_u8 != 0 and
        shouldCarveCave(world_x, world_y, world_z, surface_height, seed, world_height_limit)) return AIR;

    return terrainMaterialFromProfile(
        world_y,
        surface_height - 4,
        surface_height - 1,
        profile.material,
    );
}

pub export fn minechunk_fill_chunk_surface_grids(
    heights: [*]u32,
    materials: [*]u32,
    heights_len: usize,
    materials_len: usize,
    chunk_x: i32,
    chunk_z: i32,
    chunk_size: i32,
    seed: i64,
    height_limit: i32,
) i32 {
    if (chunk_size <= 0 or height_limit <= 0) return -1;
    const sample_size_i64 = @as(i64, chunk_size) + 2;
    const sample_size: usize = @intCast(sample_size_i64);
    const total_columns = sample_size * sample_size;
    if (heights_len < total_columns or materials_len < total_columns) return -2;

    const origin_x = @as(i64, chunk_x) * @as(i64, chunk_size) - 1;
    const origin_z = @as(i64, chunk_z) * @as(i64, chunk_size) - 1;

    const work = SurfaceGridWork{
        .heights = heights,
        .materials = materials,
        .start = 0,
        .end = total_columns,
        .sample_size = sample_size,
        .origin_x = origin_x,
        .origin_z = origin_z,
        .seed = seed,
        .height_limit = height_limit,
    };
    const worker_count = parallelWorkerCount(total_columns, SURFACE_PARALLEL_MIN_COLUMNS);
    if (worker_count <= 1 or !fillSurfaceGridParallel(work, worker_count)) {
        fillSurfaceGridRange(&work);
    }

    return 0;
}

pub export fn minechunk_fill_chunk_surface_grids_batch(
    heights: [*]u32,
    materials: [*]u32,
    chunk_xs: [*]const i32,
    chunk_zs: [*]const i32,
    heights_len: usize,
    materials_len: usize,
    chunk_count: usize,
    chunk_size: i32,
    seed: i64,
    height_limit: i32,
) i32 {
    if (chunk_size <= 0 or height_limit <= 0) return -1;
    if (chunk_count == 0) return 0;
    const sample_size_i64 = @as(i64, chunk_size) + 2;
    const sample_size: usize = @intCast(sample_size_i64);
    const plane_cells = sample_size * sample_size;
    const total_columns = chunk_count * plane_cells;
    if (heights_len < total_columns or materials_len < total_columns) return -2;

    const work = SurfaceGridBatchWork{
        .heights = heights,
        .materials = materials,
        .chunk_xs = chunk_xs,
        .chunk_zs = chunk_zs,
        .start = 0,
        .end = total_columns,
        .sample_size = sample_size,
        .plane_cells = plane_cells,
        .chunk_size = chunk_size,
        .seed = seed,
        .height_limit = height_limit,
    };
    const worker_count = parallelWorkerCount(total_columns, SURFACE_PARALLEL_MIN_COLUMNS);
    if (worker_count <= 1 or !fillSurfaceGridBatchParallel(work, worker_count)) {
        fillSurfaceGridBatchRange(&work);
    }

    return 0;
}

pub export fn minechunk_fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface(
    blocks: [*]u8,
    voxel_materials: [*]u32,
    top_plane: [*]u8,
    bottom_plane: [*]u8,
    surface_heights: [*]const u32,
    surface_materials: [*]const u32,
    blocks_len: usize,
    voxel_materials_len: usize,
    plane_len: usize,
    surface_len: usize,
    local_height: i32,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    chunk_size: i32,
    seed: i64,
    world_height_limit: i32,
    carve_caves_u8: u8,
) i32 {
    if (chunk_size <= 0 or local_height <= 0 or world_height_limit <= 0) return -1;

    const sample_size_i64 = @as(i64, chunk_size) + 2;
    const sample_size: usize = @intCast(sample_size_i64);
    const plane_cells = sample_size * sample_size;
    const block_cells = @as(usize, @intCast(local_height)) * plane_cells;
    if (blocks_len < block_cells or
        voxel_materials_len < block_cells or
        plane_len < plane_cells or
        surface_len < plane_cells) return -2;

    const origin_x = @as(i64, chunk_x) * @as(i64, chunk_size) - 1;
    const origin_z = @as(i64, chunk_z) * @as(i64, chunk_size) - 1;
    const origin_y = @as(i64, chunk_y) * @as(i64, chunk_size);
    const world_limit = @as(i64, world_height_limit);
    const fill_start_y = maxI64(origin_y, 0);
    const fill_top_y = minI64(origin_y + @as(i64, local_height), world_limit);
    const top_world_y = origin_y + @as(i64, local_height);
    const bottom_world_y = origin_y - 1;
    const top_in_bounds = 0 <= top_world_y and top_world_y < world_limit;
    const bottom_in_bounds = 0 <= bottom_world_y and bottom_world_y < world_limit;
    const carve_caves = carve_caves_u8 != 0;

    var work = VoxelFromSurfaceWork{
        .blocks = blocks,
        .voxel_materials = voxel_materials,
        .top_plane = top_plane,
        .bottom_plane = bottom_plane,
        .surface_heights = surface_heights,
        .surface_materials = surface_materials,
        .start = 0,
        .end = plane_cells,
        .sample_size = sample_size,
        .plane_cells = plane_cells,
        .origin_x = origin_x,
        .origin_z = origin_z,
        .origin_y = origin_y,
        .fill_start_y = fill_start_y,
        .fill_top_y = fill_top_y,
        .top_world_y = top_world_y,
        .bottom_world_y = bottom_world_y,
        .top_in_bounds = top_in_bounds,
        .bottom_in_bounds = bottom_in_bounds,
        .seed = seed,
        .world_height_limit = world_height_limit,
        .carve_caves = carve_caves,
        .cave_y_terms = null,
        .cave_y_terms_len = 0,
    };
    if (work.carve_caves and !chunkAndNeighborPlanesHaveCaveActiveY(origin_y, local_height, world_height_limit)) {
        work.carve_caves = false;
    }
    var cave_y_terms: [MAX_CAVE_Y_TERMS]CaveYTerm = undefined;
    if (work.carve_caves and @as(usize, @intCast(local_height)) <= MAX_CAVE_Y_TERMS) {
        work.cave_y_terms_len = prepareCaveYTerms(&cave_y_terms, origin_y, local_height, world_height_limit);
        work.cave_y_terms = &cave_y_terms;
    }
    const parallel_min_columns = if (work.carve_caves) VOXEL_CAVE_PARALLEL_MIN_COLUMNS else VOXEL_SIMPLE_PARALLEL_MIN_COLUMNS;
    const worker_count = parallelWorkerCount(plane_cells, parallel_min_columns);
    if (worker_count <= 1 or !fillVoxelFromSurfaceParallel(work, worker_count)) {
        fillVoxelFromSurfaceRange(&work);
    }

    return 0;
}

pub export fn minechunk_fill_stacked_chunk_voxel_grid_with_neighbor_planes_from_surface_batch(
    blocks: [*]u8,
    voxel_materials: [*]u32,
    top_plane: [*]u8,
    bottom_plane: [*]u8,
    surface_heights: [*]const u32,
    surface_materials: [*]const u32,
    chunk_xs: [*]const i32,
    chunk_ys: [*]const i32,
    chunk_zs: [*]const i32,
    blocks_len: usize,
    voxel_materials_len: usize,
    plane_len: usize,
    surface_len: usize,
    chunk_count: usize,
    local_height: i32,
    chunk_size: i32,
    seed: i64,
    world_height_limit: i32,
    carve_caves_u8: u8,
) i32 {
    if (chunk_size <= 0 or local_height <= 0 or world_height_limit <= 0) return -1;
    if (chunk_count == 0) return 0;

    const sample_size_i64 = @as(i64, chunk_size) + 2;
    const sample_size: usize = @intCast(sample_size_i64);
    const plane_cells = sample_size * sample_size;
    const block_cells = @as(usize, @intCast(local_height)) * plane_cells;
    const total_block_cells = chunk_count * block_cells;
    const total_plane_cells = chunk_count * plane_cells;
    if (blocks_len < total_block_cells or
        voxel_materials_len < total_block_cells or
        plane_len < total_plane_cells or
        surface_len < total_plane_cells) return -2;

    const carve_caves = carve_caves_u8 != 0;
    const work = VoxelFromSurfaceBatchWork{
        .blocks = blocks,
        .voxel_materials = voxel_materials,
        .top_plane = top_plane,
        .bottom_plane = bottom_plane,
        .surface_heights = surface_heights,
        .surface_materials = surface_materials,
        .chunk_xs = chunk_xs,
        .chunk_ys = chunk_ys,
        .chunk_zs = chunk_zs,
        .start = 0,
        .end = total_plane_cells,
        .sample_size = sample_size,
        .plane_cells = plane_cells,
        .block_cells = block_cells,
        .chunk_count = chunk_count,
        .local_height = local_height,
        .chunk_size = chunk_size,
        .seed = seed,
        .world_height_limit = world_height_limit,
        .carve_caves = carve_caves,
        .cave_y_terms = null,
        .cave_y_terms_len = 0,
    };
    const worker_count = parallelWorkerCount(chunk_count, 1);
    if (worker_count <= 1 or !fillVoxelFromSurfaceBatchParallel(work, worker_count)) {
        fillVoxelFromSurfaceBatchRange(&work);
    }

    return 0;
}

pub export fn minechunk_fill_stacked_chunk_voxel_grid(
    blocks: [*]u8,
    voxel_materials: [*]u32,
    blocks_len: usize,
    voxel_materials_len: usize,
    local_height: i32,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    chunk_size: i32,
    seed: i64,
    world_height_limit: i32,
    carve_caves_u8: u8,
) i32 {
    if (chunk_size <= 0 or local_height <= 0 or world_height_limit <= 0) return -1;

    const sample_size_i64 = @as(i64, chunk_size) + 2;
    const sample_size: usize = @intCast(sample_size_i64);
    const plane_cells = sample_size * sample_size;
    const block_cells = @as(usize, @intCast(local_height)) * plane_cells;
    if (blocks_len < block_cells or voxel_materials_len < block_cells) return -2;

    const origin_x = @as(i64, chunk_x) * @as(i64, chunk_size) - 1;
    const origin_z = @as(i64, chunk_z) * @as(i64, chunk_size) - 1;
    const origin_y = @as(i64, chunk_y) * @as(i64, chunk_size);
    const world_limit = @as(i64, world_height_limit);
    const fill_start_y = maxI64(origin_y, 0);
    const fill_top_y = minI64(origin_y + @as(i64, local_height), world_limit);
    const carve_caves = carve_caves_u8 != 0;

    const work = VoxelGeneratedWork{
        .blocks = blocks,
        .voxel_materials = voxel_materials,
        .start = 0,
        .end = plane_cells,
        .sample_size = sample_size,
        .plane_cells = plane_cells,
        .origin_x = origin_x,
        .origin_z = origin_z,
        .origin_y = origin_y,
        .fill_start_y = fill_start_y,
        .fill_top_y = fill_top_y,
        .seed = seed,
        .world_height_limit = world_height_limit,
        .carve_caves = carve_caves,
    };
    const parallel_min_columns = if (carve_caves) VOXEL_CAVE_PARALLEL_MIN_COLUMNS else VOXEL_SIMPLE_PARALLEL_MIN_COLUMNS;
    const worker_count = parallelWorkerCount(plane_cells, parallel_min_columns);
    if (worker_count <= 1 or !fillVoxelGeneratedParallel(work, worker_count)) {
        fillVoxelGeneratedRange(&work);
    }

    return 0;
}

pub export fn minechunk_fill_chunk_voxel_grid(
    blocks: [*]u8,
    voxel_materials: [*]u32,
    blocks_len: usize,
    voxel_materials_len: usize,
    local_height: i32,
    chunk_x: i32,
    chunk_z: i32,
    chunk_size: i32,
    seed: i64,
    height_limit: i32,
    carve_caves_u8: u8,
) i32 {
    return minechunk_fill_stacked_chunk_voxel_grid(
        blocks,
        voxel_materials,
        blocks_len,
        voxel_materials_len,
        local_height,
        chunk_x,
        0,
        chunk_z,
        chunk_size,
        seed,
        height_limit,
        carve_caves_u8,
    );
}

fn fillSurfaceGridRange(work: *const SurfaceGridWork) void {
    var column_index = work.start;
    while (column_index < work.end) {
        const local_z = column_index / work.sample_size;
        const local_x = column_index - local_z * work.sample_size;
        const world_z = work.origin_z + @as(i64, @intCast(local_z));
        const world_x = work.origin_x + @as(i64, @intCast(local_x));
        if (SURFACE_SIMD_ENABLED and local_x + 4 <= work.sample_size and column_index + 4 <= work.end) {
            const profile = surfaceProfileAt4(
                @as(VF, @splat(@as(f32, @floatFromInt(world_x)))) + X_OFFSETS_F32,
                @as(VF, @splat(@as(f32, @floatFromInt(world_z)))),
                work.seed,
                work.height_limit,
            );
            storeU32x4(work.heights + column_index, profile.heights);
            storeU32x4(work.materials + column_index, profile.materials);
            column_index += 4;
            continue;
        }
        const profile = surfaceProfileAt(
            @as(f32, @floatFromInt(world_x)),
            @as(f32, @floatFromInt(world_z)),
            work.seed,
            work.height_limit,
        );
        work.heights[column_index] = profile.height;
        work.materials[column_index] = profile.material;
        column_index += 1;
    }
}

fn fillSurfaceGridBatchRange(work: *const SurfaceGridBatchWork) void {
    var global_index = work.start;
    while (global_index < work.end) {
        const chunk_index = global_index / work.plane_cells;
        const column_index = global_index - chunk_index * work.plane_cells;
        const local_z = column_index / work.sample_size;
        const local_x = column_index - local_z * work.sample_size;
        const origin_x = @as(i64, work.chunk_xs[chunk_index]) * @as(i64, work.chunk_size) - 1;
        const origin_z = @as(i64, work.chunk_zs[chunk_index]) * @as(i64, work.chunk_size) - 1;
        const world_z = origin_z + @as(i64, @intCast(local_z));
        const world_x = origin_x + @as(i64, @intCast(local_x));
        if (SURFACE_SIMD_ENABLED and local_x + 4 <= work.sample_size and global_index + 4 <= work.end) {
            const profile = surfaceProfileAt4(
                @as(VF, @splat(@as(f32, @floatFromInt(world_x)))) + X_OFFSETS_F32,
                @as(VF, @splat(@as(f32, @floatFromInt(world_z)))),
                work.seed,
                work.height_limit,
            );
            storeU32x4(work.heights + global_index, profile.heights);
            storeU32x4(work.materials + global_index, profile.materials);
            global_index += 4;
            continue;
        }
        const profile = surfaceProfileAt(
            @as(f32, @floatFromInt(world_x)),
            @as(f32, @floatFromInt(world_z)),
            work.seed,
            work.height_limit,
        );
        work.heights[global_index] = profile.height;
        work.materials[global_index] = profile.material;
        global_index += 1;
    }
}

fn fillVoxelFromSurfaceRange(work: *const VoxelFromSurfaceWork) void {
    var column_index = work.start;
    while (column_index < work.end) {
        const local_z = column_index / work.sample_size;
        const local_x = column_index - local_z * work.sample_size;
        const world_z = work.origin_z + @as(i64, @intCast(local_z));
        const world_x = work.origin_x + @as(i64, @intCast(local_x));

        if (work.carve_caves and local_x + 4 <= work.sample_size and column_index + 4 <= work.end) {
            const surface_heights4 = loadU32x4(work.surface_heights + column_index);
            const surface_materials4 = loadU32x4(work.surface_materials + column_index);
            const world_x4 = @as(VI, @splat(@as(i32, @intCast(world_x)))) + VI{ 0, 1, 2, 3 };
            const surface_height_i: VI = @intCast(surface_heights4);
            fillColumnWithOptionalCaves4(
                work.blocks,
                work.voxel_materials,
                0,
                work.plane_cells,
                column_index,
                work.origin_y,
                work.fill_start_y,
                work.fill_top_y,
                surface_heights4,
                surface_materials4,
                world_x4,
                world_z,
                work.seed,
                work.world_height_limit,
                work.cave_y_terms,
                work.cave_y_terms_len,
            );

            if (work.top_in_bounds) {
                const top_world_y_i32: i32 = @intCast(work.top_world_y);
                const top_active_mask = @as(VI, @splat(top_world_y_i32)) < surface_height_i;
                const top_solid_mask = top_active_mask & !shouldCarveCave4(
                    world_x4,
                    work.top_world_y,
                    world_z,
                    surface_height_i,
                    work.seed,
                    work.world_height_limit,
                );
                inline for (0..4) |lane| {
                    if (top_solid_mask[lane]) {
                        work.top_plane[column_index + lane] = 1;
                    }
                }
            }
            if (work.bottom_in_bounds) {
                const bottom_world_y_i32: i32 = @intCast(work.bottom_world_y);
                const bottom_active_mask = @as(VI, @splat(bottom_world_y_i32)) < surface_height_i;
                const bottom_solid_mask = bottom_active_mask & !shouldCarveCave4(
                    world_x4,
                    work.bottom_world_y,
                    world_z,
                    surface_height_i,
                    work.seed,
                    work.world_height_limit,
                );
                inline for (0..4) |lane| {
                    if (bottom_solid_mask[lane]) {
                        work.bottom_plane[column_index + lane] = 1;
                    }
                }
            }

            column_index += 4;
            continue;
        }

        if (!work.carve_caves and local_x + 4 <= work.sample_size and column_index + 4 <= work.end) {
            const surface_heights4 = loadU32x4(work.surface_heights + column_index);
            const surface_materials4 = loadU32x4(work.surface_materials + column_index);
            fillSolidColumnsNoCaves4(
                work.blocks,
                work.voxel_materials,
                0,
                work.plane_cells,
                column_index,
                work.origin_y,
                work.fill_start_y,
                work.fill_top_y,
                surface_heights4,
                surface_materials4,
            );

            inline for (0..4) |lane| {
                const lane_surface_height = @as(i64, surface_heights4[lane]);
                if (work.top_in_bounds and work.top_world_y < lane_surface_height) {
                    work.top_plane[column_index + lane] = 1;
                }
                if (work.bottom_in_bounds and work.bottom_world_y < lane_surface_height) {
                    work.bottom_plane[column_index + lane] = 1;
                }
            }

            column_index += 4;
            continue;
        }

        const solid_surface_height = @as(i64, work.surface_heights[column_index]);
        const solid_surface_material = work.surface_materials[column_index];

        const fill_end_y = minI64(work.fill_top_y, solid_surface_height);
        const stone_limit = solid_surface_height - 4;
        const dirt_limit = solid_surface_height - 1;

        if (work.carve_caves) {
            fillColumnWithOptionalCaves(
                work.blocks,
                work.voxel_materials,
                0,
                work.plane_cells,
                column_index,
                work.origin_y,
                work.fill_start_y,
                fill_end_y,
                stone_limit,
                dirt_limit,
                solid_surface_height,
                solid_surface_material,
                world_x,
                world_z,
                work.seed,
                work.world_height_limit,
                true,
            );
        } else {
            fillSolidColumnNoCaves(
                work.blocks,
                work.voxel_materials,
                0,
                work.plane_cells,
                column_index,
                work.origin_y,
                work.fill_start_y,
                fill_end_y,
                stone_limit,
                dirt_limit,
                solid_surface_material,
            );
        }

        if (work.top_in_bounds and work.top_world_y < solid_surface_height) {
            if (!work.carve_caves or !shouldCarveCave(world_x, work.top_world_y, world_z, solid_surface_height, work.seed, work.world_height_limit)) {
                work.top_plane[column_index] = 1;
            }
        }

        if (work.bottom_in_bounds and work.bottom_world_y < solid_surface_height) {
            if (!work.carve_caves or !shouldCarveCave(world_x, work.bottom_world_y, world_z, solid_surface_height, work.seed, work.world_height_limit)) {
                work.bottom_plane[column_index] = 1;
            }
        }
        column_index += 1;
    }
}

fn fillVoxelFromSurfaceBatchRange(work: *const VoxelFromSurfaceBatchWork) void {
    var global_index = work.start;
    while (global_index < work.end) {
        const chunk_index = global_index / work.plane_cells;
        const column_index = global_index - chunk_index * work.plane_cells;
        const local_z = column_index / work.sample_size;
        const local_x = column_index - local_z * work.sample_size;
        const plane_offset = chunk_index * work.plane_cells;
        const block_offset = chunk_index * work.block_cells;
        const origin_x = @as(i64, work.chunk_xs[chunk_index]) * @as(i64, work.chunk_size) - 1;
        const origin_z = @as(i64, work.chunk_zs[chunk_index]) * @as(i64, work.chunk_size) - 1;
        const origin_y = @as(i64, work.chunk_ys[chunk_index]) * @as(i64, work.chunk_size);
        const world_limit = @as(i64, work.world_height_limit);
        const fill_start_y = maxI64(origin_y, 0);
        const fill_top_y = minI64(origin_y + @as(i64, work.local_height), world_limit);
        const top_world_y = origin_y + @as(i64, work.local_height);
        const bottom_world_y = origin_y - 1;
        const top_in_bounds = 0 <= top_world_y and top_world_y < world_limit;
        const bottom_in_bounds = 0 <= bottom_world_y and bottom_world_y < world_limit;
        const world_z = origin_z + @as(i64, @intCast(local_z));
        const world_x = origin_x + @as(i64, @intCast(local_x));

        if (work.carve_caves and local_x + 4 <= work.sample_size and global_index + 4 <= work.end) {
            const surface_heights4 = loadU32x4(work.surface_heights + plane_offset + column_index);
            const surface_materials4 = loadU32x4(work.surface_materials + plane_offset + column_index);
            const world_x4 = @as(VI, @splat(@as(i32, @intCast(world_x)))) + VI{ 0, 1, 2, 3 };
            const surface_height_i: VI = @intCast(surface_heights4);
            fillColumnWithOptionalCaves4(
                work.blocks,
                work.voxel_materials,
                block_offset,
                work.plane_cells,
                column_index,
                origin_y,
                fill_start_y,
                fill_top_y,
                surface_heights4,
                surface_materials4,
                world_x4,
                world_z,
                work.seed,
                work.world_height_limit,
                work.cave_y_terms,
                work.cave_y_terms_len,
            );

            if (top_in_bounds) {
                const top_world_y_i32: i32 = @intCast(top_world_y);
                const top_active_mask = @as(VI, @splat(top_world_y_i32)) < surface_height_i;
                const top_solid_mask = top_active_mask & !shouldCarveCave4(
                    world_x4,
                    top_world_y,
                    world_z,
                    surface_height_i,
                    work.seed,
                    work.world_height_limit,
                );
                inline for (0..4) |lane| {
                    if (top_solid_mask[lane]) {
                        work.top_plane[plane_offset + column_index + lane] = 1;
                    }
                }
            }
            if (bottom_in_bounds) {
                const bottom_world_y_i32: i32 = @intCast(bottom_world_y);
                const bottom_active_mask = @as(VI, @splat(bottom_world_y_i32)) < surface_height_i;
                const bottom_solid_mask = bottom_active_mask & !shouldCarveCave4(
                    world_x4,
                    bottom_world_y,
                    world_z,
                    surface_height_i,
                    work.seed,
                    work.world_height_limit,
                );
                inline for (0..4) |lane| {
                    if (bottom_solid_mask[lane]) {
                        work.bottom_plane[plane_offset + column_index + lane] = 1;
                    }
                }
            }

            global_index += 4;
            continue;
        }

        if (!work.carve_caves and local_x + 4 <= work.sample_size and global_index + 4 <= work.end) {
            const surface_heights4 = loadU32x4(work.surface_heights + plane_offset + column_index);
            const surface_materials4 = loadU32x4(work.surface_materials + plane_offset + column_index);
            fillSolidColumnsNoCaves4(
                work.blocks,
                work.voxel_materials,
                block_offset,
                work.plane_cells,
                column_index,
                origin_y,
                fill_start_y,
                fill_top_y,
                surface_heights4,
                surface_materials4,
            );

            inline for (0..4) |lane| {
                const lane_surface_height = @as(i64, surface_heights4[lane]);
                if (top_in_bounds and top_world_y < lane_surface_height) {
                    work.top_plane[plane_offset + column_index + lane] = 1;
                }
                if (bottom_in_bounds and bottom_world_y < lane_surface_height) {
                    work.bottom_plane[plane_offset + column_index + lane] = 1;
                }
            }

            global_index += 4;
            continue;
        }

        const solid_surface_height = @as(i64, work.surface_heights[plane_offset + column_index]);
        const solid_surface_material = work.surface_materials[plane_offset + column_index];

        const fill_end_y = minI64(fill_top_y, solid_surface_height);
        const stone_limit = solid_surface_height - 4;
        const dirt_limit = solid_surface_height - 1;

        if (work.carve_caves) {
            fillColumnWithOptionalCaves(
                work.blocks,
                work.voxel_materials,
                block_offset,
                work.plane_cells,
                column_index,
                origin_y,
                fill_start_y,
                fill_end_y,
                stone_limit,
                dirt_limit,
                solid_surface_height,
                solid_surface_material,
                world_x,
                world_z,
                work.seed,
                work.world_height_limit,
                true,
            );
        } else {
            fillSolidColumnNoCaves(
                work.blocks,
                work.voxel_materials,
                block_offset,
                work.plane_cells,
                column_index,
                origin_y,
                fill_start_y,
                fill_end_y,
                stone_limit,
                dirt_limit,
                solid_surface_material,
            );
        }

        if (top_in_bounds and top_world_y < solid_surface_height) {
            if (!work.carve_caves or !shouldCarveCave(world_x, top_world_y, world_z, solid_surface_height, work.seed, work.world_height_limit)) {
                work.top_plane[plane_offset + column_index] = 1;
            }
        }

        if (bottom_in_bounds and bottom_world_y < solid_surface_height) {
            if (!work.carve_caves or !shouldCarveCave(world_x, bottom_world_y, world_z, solid_surface_height, work.seed, work.world_height_limit)) {
                work.bottom_plane[plane_offset + column_index] = 1;
            }
        }
        global_index += 1;
    }
}

fn fillVoxelGeneratedRange(work: *const VoxelGeneratedWork) void {
    for (work.start..work.end) |column_index| {
        const local_z = column_index / work.sample_size;
        const local_x = column_index - local_z * work.sample_size;
        const world_z = work.origin_z + @as(i64, @intCast(local_z));
        const world_x = work.origin_x + @as(i64, @intCast(local_x));
        const profile = surfaceProfileAt(
            @as(f32, @floatFromInt(world_x)),
            @as(f32, @floatFromInt(world_z)),
            work.seed,
            work.world_height_limit,
        );
        const solid_surface_height = @as(i64, profile.height);
        const fill_end_y = minI64(work.fill_top_y, solid_surface_height);
        const stone_limit = solid_surface_height - 4;
        const dirt_limit = solid_surface_height - 1;

        fillColumnWithOptionalCaves(
            work.blocks,
            work.voxel_materials,
            0,
            work.plane_cells,
            column_index,
            work.origin_y,
            work.fill_start_y,
            fill_end_y,
            stone_limit,
            dirt_limit,
            solid_surface_height,
            profile.material,
            world_x,
            world_z,
            work.seed,
            work.world_height_limit,
            work.carve_caves,
        );
    }
}

fn fillSurfaceGridParallel(work: SurfaceGridWork, worker_count: usize) bool {
    var contexts: [MAX_PARALLEL_WORKERS]SurfaceGridWork = undefined;
    var threads: [MAX_PARALLEL_WORKERS]std.Thread = undefined;
    var spawned: usize = 0;
    for (0..worker_count) |worker_index| {
        const start = work.start + (worker_index * (work.end - work.start)) / worker_count;
        const end = work.start + ((worker_index + 1) * (work.end - work.start)) / worker_count;
        contexts[worker_index] = work;
        contexts[worker_index].start = start;
        contexts[worker_index].end = end;
        threads[worker_index] = std.Thread.spawn(.{}, fillSurfaceGridRange, .{&contexts[worker_index]}) catch {
            for (0..spawned) |join_index| {
                threads[join_index].join();
            }
            return false;
        };
        spawned += 1;
    }
    for (0..spawned) |join_index| {
        threads[join_index].join();
    }
    return true;
}

fn fillSurfaceGridBatchParallel(work: SurfaceGridBatchWork, worker_count: usize) bool {
    var contexts: [MAX_PARALLEL_WORKERS]SurfaceGridBatchWork = undefined;
    var threads: [MAX_PARALLEL_WORKERS]std.Thread = undefined;
    var spawned: usize = 0;
    for (0..worker_count) |worker_index| {
        const start = work.start + (worker_index * (work.end - work.start)) / worker_count;
        const end = work.start + ((worker_index + 1) * (work.end - work.start)) / worker_count;
        contexts[worker_index] = work;
        contexts[worker_index].start = start;
        contexts[worker_index].end = end;
        threads[worker_index] = std.Thread.spawn(.{}, fillSurfaceGridBatchRange, .{&contexts[worker_index]}) catch {
            for (0..spawned) |join_index| {
                threads[join_index].join();
            }
            return false;
        };
        spawned += 1;
    }
    for (0..spawned) |join_index| {
        threads[join_index].join();
    }
    return true;
}

fn fillVoxelFromSurfaceParallel(work: VoxelFromSurfaceWork, worker_count: usize) bool {
    var contexts: [MAX_PARALLEL_WORKERS]VoxelFromSurfaceWork = undefined;
    var threads: [MAX_PARALLEL_WORKERS]std.Thread = undefined;
    var spawned: usize = 0;
    for (0..worker_count) |worker_index| {
        const start = work.start + (worker_index * (work.end - work.start)) / worker_count;
        const end = work.start + ((worker_index + 1) * (work.end - work.start)) / worker_count;
        contexts[worker_index] = work;
        contexts[worker_index].start = start;
        contexts[worker_index].end = end;
        threads[worker_index] = std.Thread.spawn(.{}, fillVoxelFromSurfaceRange, .{&contexts[worker_index]}) catch {
            for (0..spawned) |join_index| {
                threads[join_index].join();
            }
            return false;
        };
        spawned += 1;
    }
    for (0..spawned) |join_index| {
        threads[join_index].join();
    }
    return true;
}

fn fillVoxelFromSurfaceBatchParallel(work: VoxelFromSurfaceBatchWork, worker_count: usize) bool {
    if (PERSISTENT_VOXEL_BATCH_POOL_ENABLED and fillVoxelFromSurfaceBatchParallelPersistent(work, worker_count)) {
        return true;
    }
    return fillVoxelFromSurfaceBatchParallelSpawn(work, worker_count);
}

fn fillVoxelFromSurfaceBatchParallelSpawn(work: VoxelFromSurfaceBatchWork, worker_count: usize) bool {
    var contexts: [MAX_PARALLEL_WORKERS]VoxelFromSurfaceBatchChunkWork = undefined;
    var threads: [MAX_PARALLEL_WORKERS]std.Thread = undefined;
    var spawned: usize = 0;
    for (0..worker_count) |worker_index| {
        contexts[worker_index] = VoxelFromSurfaceBatchChunkWork{
            .work = work,
            .worker_index = worker_index,
            .worker_count = worker_count,
        };
        threads[worker_index] = std.Thread.spawn(.{}, fillVoxelFromSurfaceBatchChunkStride, .{&contexts[worker_index]}) catch {
            for (0..spawned) |join_index| {
                threads[join_index].join();
            }
            return false;
        };
        spawned += 1;
    }
    for (0..spawned) |join_index| {
        threads[join_index].join();
    }
    return true;
}

fn fillVoxelFromSurfaceBatchParallelPersistent(work: VoxelFromSurfaceBatchWork, worker_count: usize) bool {
    if (persistent_voxel_batch_pool.busy.cmpxchgStrong(false, true, .acquire, .monotonic) != null) return false;
    defer persistent_voxel_batch_pool.busy.store(false, .release);
    if (!ensurePersistentVoxelBatchPool(worker_count)) return false;
    if (worker_count > persistent_voxel_batch_pool.worker_count.load(.acquire)) return false;

    pthreadLock(&persistent_voxel_batch_pool.mutex);
    defer pthreadUnlock(&persistent_voxel_batch_pool.mutex);

    persistent_voxel_batch_pool.work = work;
    persistent_voxel_batch_pool.completed_workers.store(0, .release);
    persistent_voxel_batch_pool.requested_workers.store(worker_count, .release);
    _ = persistent_voxel_batch_pool.generation.fetchAdd(1, .release);
    pthreadBroadcast(&persistent_voxel_batch_pool.job_cond);

    while (persistent_voxel_batch_pool.completed_workers.load(.acquire) < worker_count) {
        _ = std.c.pthread_cond_wait(&persistent_voxel_batch_pool.done_cond, &persistent_voxel_batch_pool.mutex);
    }
    return true;
}

fn ensurePersistentVoxelBatchPool(required_workers: usize) bool {
    if (persistent_voxel_batch_pool.initialized.load(.acquire)) {
        return persistent_voxel_batch_pool.worker_count.load(.acquire) >= required_workers;
    }

    if (persistent_voxel_batch_pool.init_lock.cmpxchgStrong(false, true, .acquire, .monotonic) != null) {
        while (persistent_voxel_batch_pool.init_lock.load(.acquire)) {
            std.Thread.yield() catch {};
        }
        return persistent_voxel_batch_pool.initialized.load(.acquire) and
            persistent_voxel_batch_pool.worker_count.load(.acquire) >= required_workers;
    }
    defer persistent_voxel_batch_pool.init_lock.store(false, .release);

    if (persistent_voxel_batch_pool.initialized.load(.acquire)) {
        return persistent_voxel_batch_pool.worker_count.load(.acquire) >= required_workers;
    }

    const cpu_count = std.Thread.getCpuCount() catch required_workers;
    const target_count = minUsize(MAX_PARALLEL_WORKERS, maxUsize(required_workers, cpu_count));
    persistent_voxel_batch_pool.shutdown_requested.store(false, .release);
    const start_generation = persistent_voxel_batch_pool.generation.load(.acquire);
    var spawned: usize = 0;
    while (spawned < target_count) : (spawned += 1) {
        persistent_voxel_batch_worker_contexts[spawned] = .{
            .worker_index = spawned,
            .start_generation = start_generation,
        };
        persistent_voxel_batch_pool.threads[spawned] = std.Thread.spawn(.{}, persistentVoxelBatchWorker, .{&persistent_voxel_batch_worker_contexts[spawned]}) catch {
            break;
        };
    }

    persistent_voxel_batch_pool.worker_count.store(spawned, .release);
    persistent_voxel_batch_pool.initialized.store(spawned > 0, .release);
    return spawned >= required_workers;
}

fn persistentVoxelBatchWorker(context: *const PersistentVoxelBatchWorkerContext) void {
    var seen_generation = context.start_generation;
    while (true) {
        pthreadLock(&persistent_voxel_batch_pool.mutex);
        while (persistent_voxel_batch_pool.generation.load(.acquire) == seen_generation and
            !persistent_voxel_batch_pool.shutdown_requested.load(.acquire))
        {
            _ = std.c.pthread_cond_wait(&persistent_voxel_batch_pool.job_cond, &persistent_voxel_batch_pool.mutex);
        }

        if (persistent_voxel_batch_pool.shutdown_requested.load(.acquire)) {
            pthreadUnlock(&persistent_voxel_batch_pool.mutex);
            return;
        }

        const generation = persistent_voxel_batch_pool.generation.load(.acquire);
        const requested_workers = persistent_voxel_batch_pool.requested_workers.load(.acquire);
        if (context.worker_index >= requested_workers) {
            seen_generation = generation;
            pthreadUnlock(&persistent_voxel_batch_pool.mutex);
            continue;
        }

        const work = persistent_voxel_batch_pool.work;
        seen_generation = generation;
        pthreadUnlock(&persistent_voxel_batch_pool.mutex);

        const chunk_context = VoxelFromSurfaceBatchChunkWork{
            .work = work,
            .worker_index = context.worker_index,
            .worker_count = requested_workers,
        };
        fillVoxelFromSurfaceBatchChunkStride(&chunk_context);

        pthreadLock(&persistent_voxel_batch_pool.mutex);
        const completed_workers = persistent_voxel_batch_pool.completed_workers.fetchAdd(1, .acq_rel) + 1;
        if (completed_workers >= requested_workers) {
            pthreadSignal(&persistent_voxel_batch_pool.done_cond);
        }
        pthreadUnlock(&persistent_voxel_batch_pool.mutex);
    }
}

fn shutdownPersistentVoxelBatchPool() void {
    if (persistent_voxel_batch_pool.init_lock.cmpxchgStrong(false, true, .acquire, .monotonic) != null) return;
    defer persistent_voxel_batch_pool.init_lock.store(false, .release);

    const worker_count = persistent_voxel_batch_pool.worker_count.load(.acquire);
    if (worker_count == 0) return;

    pthreadLock(&persistent_voxel_batch_pool.mutex);
    persistent_voxel_batch_pool.shutdown_requested.store(true, .release);
    pthreadBroadcast(&persistent_voxel_batch_pool.job_cond);
    pthreadUnlock(&persistent_voxel_batch_pool.mutex);

    for (0..worker_count) |worker_index| {
        persistent_voxel_batch_pool.threads[worker_index].join();
    }

    persistent_voxel_batch_pool.worker_count.store(0, .release);
    persistent_voxel_batch_pool.requested_workers.store(0, .release);
    persistent_voxel_batch_pool.completed_workers.store(0, .release);
    persistent_voxel_batch_pool.initialized.store(false, .release);
    persistent_voxel_batch_pool.shutdown_requested.store(false, .release);
}

fn pthreadLock(mutex: *std.c.pthread_mutex_t) void {
    _ = std.c.pthread_mutex_lock(mutex);
}

fn pthreadUnlock(mutex: *std.c.pthread_mutex_t) void {
    _ = std.c.pthread_mutex_unlock(mutex);
}

fn pthreadSignal(cond: *std.c.pthread_cond_t) void {
    _ = std.c.pthread_cond_signal(cond);
}

fn pthreadBroadcast(cond: *std.c.pthread_cond_t) void {
    _ = std.c.pthread_cond_broadcast(cond);
}

fn fillVoxelFromSurfaceBatchChunkStride(context: *const VoxelFromSurfaceBatchChunkWork) void {
    var chunk_index = context.worker_index;
    while (chunk_index < context.work.chunk_count) : (chunk_index += context.worker_count) {
        var chunk_work = context.work;
        chunk_work.start = chunk_index * chunk_work.plane_cells;
        chunk_work.end = chunk_work.start + chunk_work.plane_cells;
        const origin_y = @as(i64, chunk_work.chunk_ys[chunk_index]) * @as(i64, chunk_work.chunk_size);
        if (chunk_work.carve_caves and !chunkAndNeighborPlanesHaveCaveActiveY(origin_y, chunk_work.local_height, chunk_work.world_height_limit)) {
            chunk_work.carve_caves = false;
        }
        var cave_y_terms: [MAX_CAVE_Y_TERMS]CaveYTerm = undefined;
        if (chunk_work.carve_caves and @as(usize, @intCast(chunk_work.local_height)) <= MAX_CAVE_Y_TERMS) {
            chunk_work.cave_y_terms_len = prepareCaveYTerms(&cave_y_terms, origin_y, chunk_work.local_height, chunk_work.world_height_limit);
            chunk_work.cave_y_terms = &cave_y_terms;
        }
        fillVoxelFromSurfaceBatchRange(&chunk_work);
    }
}

fn fillVoxelGeneratedParallel(work: VoxelGeneratedWork, worker_count: usize) bool {
    var contexts: [MAX_PARALLEL_WORKERS]VoxelGeneratedWork = undefined;
    var threads: [MAX_PARALLEL_WORKERS]std.Thread = undefined;
    var spawned: usize = 0;
    for (0..worker_count) |worker_index| {
        const start = work.start + (worker_index * (work.end - work.start)) / worker_count;
        const end = work.start + ((worker_index + 1) * (work.end - work.start)) / worker_count;
        contexts[worker_index] = work;
        contexts[worker_index].start = start;
        contexts[worker_index].end = end;
        threads[worker_index] = std.Thread.spawn(.{}, fillVoxelGeneratedRange, .{&contexts[worker_index]}) catch {
            for (0..spawned) |join_index| {
                threads[join_index].join();
            }
            return false;
        };
        spawned += 1;
    }
    for (0..spawned) |join_index| {
        threads[join_index].join();
    }
    return true;
}

fn parallelWorkerCount(total_items: usize, min_items_per_worker: usize) usize {
    if (total_items < min_items_per_worker * 2) return 1;
    const cpu_count = std.Thread.getCpuCount() catch 1;
    const by_work = maxUsize(1, total_items / min_items_per_worker);
    return maxUsize(1, minUsize(minUsize(cpu_count, MAX_PARALLEL_WORKERS), by_work));
}

fn surfaceProfileAt(x: f32, z: f32, seed: i64, height_limit: i32) SurfaceProfile {
    const broad = valueNoise2D(x, z, seed + 11, 0.0009765625 * TERRAIN_FREQUENCY_SCALE);
    const ridge = valueNoise2D(x, z, seed + 23, 0.00390625 * TERRAIN_FREQUENCY_SCALE);
    const detail = valueNoise2D(x, z, seed + 47, 0.010416667 * TERRAIN_FREQUENCY_SCALE);
    const micro = valueNoise2D(x, z, seed + 71, 0.020833334 * TERRAIN_FREQUENCY_SCALE);
    const nano = valueNoise2D(x, z, seed + 97, 0.041666668 * TERRAIN_FREQUENCY_SCALE);

    const upper_bound = height_limit - 1;
    const normalized_height = 24.0 + broad * 11.0 + ridge * 8.0 + detail * 4.5 + micro * 1.75 + nano * 0.75;
    const height_scale = if (upper_bound > 0) @as(f32, @floatFromInt(upper_bound)) / 50.0 else 1.0;
    var height_i: i32 = @intFromFloat(normalized_height * height_scale);
    if (height_i < 4) height_i = 4;
    if (height_i > upper_bound) height_i = upper_bound;

    const sand_threshold = maxI32(4, @intFromFloat(@as(f32, @floatFromInt(height_limit)) * 0.18));
    const stone_threshold = maxI32(sand_threshold + 6, @intFromFloat(@as(f32, @floatFromInt(height_limit)) * 0.58));
    const snow_threshold = maxI32(stone_threshold + 6, @intFromFloat(@as(f32, @floatFromInt(height_limit)) * 0.82));

    const height_u32: u32 = @intCast(height_i);
    if (height_i >= snow_threshold) return .{ .height = height_u32, .material = SNOW };
    if (height_i <= sand_threshold) return .{ .height = height_u32, .material = SAND };
    if (height_i >= stone_threshold and (detail + micro * 0.5 + nano * 0.35) > 0.10) {
        return .{ .height = height_u32, .material = STONE };
    }
    return .{ .height = height_u32, .material = GRASS };
}

fn surfaceProfileAt4(x: VF, z: VF, seed: i64, height_limit: i32) SurfaceProfile4 {
    const broad = valueNoise2D4(x, z, seed + 11, 0.0009765625 * TERRAIN_FREQUENCY_SCALE);
    const ridge = valueNoise2D4(x, z, seed + 23, 0.00390625 * TERRAIN_FREQUENCY_SCALE);
    const detail = valueNoise2D4(x, z, seed + 47, 0.010416667 * TERRAIN_FREQUENCY_SCALE);
    const micro = valueNoise2D4(x, z, seed + 71, 0.020833334 * TERRAIN_FREQUENCY_SCALE);
    const nano = valueNoise2D4(x, z, seed + 97, 0.041666668 * TERRAIN_FREQUENCY_SCALE);

    const upper_bound = height_limit - 1;
    const normalized_height = @as(VF, @splat(@as(f32, 24.0))) +
        broad * @as(VF, @splat(@as(f32, 11.0))) +
        ridge * @as(VF, @splat(@as(f32, 8.0))) +
        detail * @as(VF, @splat(@as(f32, 4.5))) +
        micro * @as(VF, @splat(@as(f32, 1.75))) +
        nano * @as(VF, @splat(@as(f32, 0.75)));
    const height_scale = if (upper_bound > 0) @as(f32, @floatFromInt(upper_bound)) / 50.0 else 1.0;
    var height_i: VI = @intFromFloat(normalized_height * @as(VF, @splat(height_scale)));
    height_i = @max(height_i, @as(VI, @splat(4)));
    height_i = @min(height_i, @as(VI, @splat(upper_bound)));

    const sand_threshold = maxI32(4, @intFromFloat(@as(f32, @floatFromInt(height_limit)) * 0.18));
    const stone_threshold = maxI32(sand_threshold + 6, @intFromFloat(@as(f32, @floatFromInt(height_limit)) * 0.58));
    const snow_threshold = maxI32(stone_threshold + 6, @intFromFloat(@as(f32, @floatFromInt(height_limit)) * 0.82));

    const stone_score = detail + micro * @as(VF, @splat(@as(f32, 0.5))) + nano * @as(VF, @splat(@as(f32, 0.35)));
    const stone_mask = (height_i >= @as(VI, @splat(stone_threshold))) & (stone_score > @as(VF, @splat(@as(f32, 0.10))));
    const sand_mask = height_i <= @as(VI, @splat(sand_threshold));
    const snow_mask = height_i >= @as(VI, @splat(snow_threshold));

    var materials = @as(VU, @splat(GRASS));
    materials = @select(u32, stone_mask, @as(VU, @splat(STONE)), materials);
    materials = @select(u32, sand_mask, @as(VU, @splat(SAND)), materials);
    materials = @select(u32, snow_mask, @as(VU, @splat(SNOW)), materials);

    return .{
        .heights = @intCast(height_i),
        .materials = materials,
    };
}

fn shouldCarveCave(
    world_x: i64,
    world_y: i64,
    world_z: i64,
    surface_height: i64,
    seed: i64,
    world_height_limit: i32,
) bool {
    if (world_y <= CAVE_BEDROCK_CLEARANCE) return false;
    const depth_below_surface = surface_height - world_y;
    if (depth_below_surface <= 0) return false;
    const world_limit = @as(i64, world_height_limit);
    if (world_y >= world_limit - 2) return false;

    const denominator = @as(f32, @floatFromInt(maxI64(1, world_limit - 1)));
    const normalized_y = @as(f32, @floatFromInt(world_y)) / denominator;
    const vertical_band = if (normalized_y <= 0.45)
        @as(f32, 1.0)
    else
        clamp01(1.0 - (normalized_y - 0.45) * 1.6);

    if (vertical_band <= CAVE_ACTIVE_BAND_MIN) return false;

    const xf = @as(f32, @floatFromInt(world_x));
    const yf = @as(f32, @floatFromInt(world_y));
    const zf = @as(f32, @floatFromInt(world_z));
    const cave_frequency = 0.018 * CAVE_FREQUENCY_SCALE;
    const cave_primary = valueNoise3D(xf, yf * 0.85, zf, seed + 101, cave_frequency);

    var depth_bonus = @as(f32, @floatFromInt(depth_below_surface)) * CAVE_DEPTH_BONUS_SCALE;
    if (depth_bonus > CAVE_DEPTH_BONUS_MAX) depth_bonus = CAVE_DEPTH_BONUS_MAX;

    const threshold = CAVE_PRIMARY_THRESHOLD - vertical_band * CAVE_VERTICAL_BONUS - depth_bonus;
    if (cave_primary + CAVE_DETAIL_WEIGHT <= threshold) return false;

    const cave_detail = valueNoise3D(xf, yf * 0.85, zf, seed + 157, cave_frequency * CAVE_DETAIL_FREQUENCY_MULTIPLIER);
    return cave_primary + cave_detail * CAVE_DETAIL_WEIGHT > threshold;
}

fn shouldCarveCave4(
    world_x: VI,
    world_y: i64,
    world_z: i64,
    surface_height: VI,
    seed: i64,
    world_height_limit: i32,
) VB {
    if (world_y <= CAVE_BEDROCK_CLEARANCE) return @as(VB, @splat(false));
    const world_limit = @as(i64, world_height_limit);
    if (world_y >= world_limit - 2) return @as(VB, @splat(false));

    const denominator = @as(f32, @floatFromInt(maxI64(1, world_limit - 1)));
    const normalized_y = @as(f32, @floatFromInt(world_y)) / denominator;
    const vertical_band = if (normalized_y <= 0.45)
        @as(f32, 1.0)
    else
        clamp01(1.0 - (normalized_y - 0.45) * 1.6);

    const world_y_i32: i32 = @intCast(world_y);
    const depth_below_surface = surface_height - @as(VI, @splat(world_y_i32));
    const positive_depth_mask = depth_below_surface > @as(VI, @splat(0));
    const deep_mask = positive_depth_mask & @as(VB, @splat(vertical_band > CAVE_ACTIVE_BAND_MIN));
    if (!@reduce(.Or, deep_mask)) return @as(VB, @splat(false));

    const xf: VF = @floatFromInt(world_x);
    const yf = @as(f32, @floatFromInt(world_y));
    const zf = @as(f32, @floatFromInt(world_z));
    const cave_frequency = 0.018 * CAVE_FREQUENCY_SCALE;
    const cave_primary = valueNoise3D4(
        xf,
        @as(VF, @splat(yf * 0.85)),
        @as(VF, @splat(zf)),
        seed + 101,
        cave_frequency,
    );

    var depth_bonus: VF = @as(VF, @floatFromInt(depth_below_surface)) * @as(VF, @splat(CAVE_DEPTH_BONUS_SCALE));
    depth_bonus = @min(depth_bonus, @as(VF, @splat(CAVE_DEPTH_BONUS_MAX)));

    const threshold = @as(VF, @splat(CAVE_PRIMARY_THRESHOLD - vertical_band * CAVE_VERTICAL_BONUS)) - depth_bonus;
    const possible_mask = deep_mask & (cave_primary + @as(VF, @splat(CAVE_DETAIL_WEIGHT)) > threshold);
    if (!@reduce(.Or, possible_mask)) return @as(VB, @splat(false));

    const cave_detail = valueNoise3D4(
        xf,
        @as(VF, @splat(yf * 0.85)),
        @as(VF, @splat(zf)),
        seed + 157,
        cave_frequency * CAVE_DETAIL_FREQUENCY_MULTIPLIER,
    );
    const cave_value = cave_primary + cave_detail * @as(VF, @splat(CAVE_DETAIL_WEIGHT));
    return possible_mask & (cave_value > threshold);
}

fn shouldCarveCave4WithTerm(
    surface_height: VI,
    seed: i64,
    term: CaveYTerm,
    primary_x_term: CaveNoiseXTerm,
    primary_z_term: CaveNoiseZTerm,
    detail_x_term: CaveNoiseXTerm,
    detail_z_term: CaveNoiseZTerm,
) VB {
    if (!term.active) return @as(VB, @splat(false));

    const depth_below_surface = surface_height - @as(VI, @splat(term.world_y_i32));
    const positive_depth_mask = depth_below_surface > @as(VI, @splat(0));
    if (!@reduce(.Or, positive_depth_mask)) return @as(VB, @splat(false));

    const cave_primary = valueNoise3D4WithTerms(
        seed + 101,
        primary_x_term,
        term.primary_noise_y,
        primary_z_term,
    );

    var depth_bonus: VF = @as(VF, @floatFromInt(depth_below_surface)) * @as(VF, @splat(CAVE_DEPTH_BONUS_SCALE));
    depth_bonus = @min(depth_bonus, @as(VF, @splat(CAVE_DEPTH_BONUS_MAX)));

    const threshold = @as(VF, @splat(term.threshold_base)) - depth_bonus;
    const possible_mask = positive_depth_mask & (cave_primary + @as(VF, @splat(CAVE_DETAIL_WEIGHT)) > threshold);
    if (!@reduce(.Or, possible_mask)) return @as(VB, @splat(false));

    const cave_detail = valueNoise3D4WithTerms(
        seed + 157,
        detail_x_term,
        term.detail_noise_y,
        detail_z_term,
    );
    const cave_value = cave_primary + cave_detail * @as(VF, @splat(CAVE_DETAIL_WEIGHT));
    return possible_mask & (cave_value > threshold);
}

fn caveYTerm(world_y: i64, world_height_limit: i32) CaveYTerm {
    const world_y_i32: i32 = @intCast(world_y);
    const yf = @as(f32, @floatFromInt(world_y));
    const y_noise = yf * 0.85;
    const cave_frequency = 0.018 * CAVE_FREQUENCY_SCALE;
    var term = CaveYTerm{
        .world_y_i32 = world_y_i32,
        .primary_noise_y = caveNoiseYTerm(y_noise, cave_frequency),
        .detail_noise_y = caveNoiseYTerm(y_noise, cave_frequency * CAVE_DETAIL_FREQUENCY_MULTIPLIER),
        .threshold_base = CAVE_PRIMARY_THRESHOLD,
        .active = false,
    };

    const world_limit = @as(i64, world_height_limit);
    if (world_y <= CAVE_BEDROCK_CLEARANCE or world_y >= world_limit - 2) return term;

    const denominator = @as(f32, @floatFromInt(maxI64(1, world_limit - 1)));
    const normalized_y = yf / denominator;
    const vertical_band = if (normalized_y <= 0.45)
        @as(f32, 1.0)
    else
        clamp01(1.0 - (normalized_y - 0.45) * 1.6);
    if (vertical_band <= CAVE_ACTIVE_BAND_MIN) return term;

    term.threshold_base = CAVE_PRIMARY_THRESHOLD - vertical_band * CAVE_VERTICAL_BONUS;
    term.active = true;
    return term;
}

fn prepareCaveYTerms(terms: *[MAX_CAVE_Y_TERMS]CaveYTerm, origin_y: i64, local_height: i32, world_height_limit: i32) usize {
    const count = minUsize(@as(usize, @intCast(local_height)), MAX_CAVE_Y_TERMS);
    for (0..count) |local_y| {
        terms[local_y] = caveYTerm(origin_y + @as(i64, @intCast(local_y)), world_height_limit);
    }
    return count;
}

fn caveNoiseYTerm(y_in: f32, frequency: f32) CaveNoiseYTerm {
    const y = y_in * frequency;
    const y0 = @floor(y);
    const iy0_scalar: i32 = @intFromFloat(y0);
    return .{
        .iy0 = @as(VI, @splat(iy0_scalar)),
        .iy1 = @as(VI, @splat(iy0_scalar + 1)),
        .v = @as(VF, @splat(fade(y - y0))),
    };
}

fn caveNoiseXTerm(x_in: VF, frequency: f32) CaveNoiseXTerm {
    const x = x_in * @as(VF, @splat(frequency));
    const x0 = @floor(x);
    const ix0: VI = @intFromFloat(x0);
    return .{
        .ix0 = ix0,
        .ix1 = ix0 + @as(VI, @splat(1)),
        .u = fade4(x - x0),
    };
}

fn caveNoiseZTerm(z_in: f32, frequency: f32) CaveNoiseZTerm {
    const z = z_in * frequency;
    const z0 = @floor(z);
    const iz0_scalar: i32 = @intFromFloat(z0);
    return .{
        .iz0 = @as(VI, @splat(iz0_scalar)),
        .iz1 = @as(VI, @splat(iz0_scalar + 1)),
        .w = @as(VF, @splat(fade(z - z0))),
    };
}

fn caveVerticalBandStartY(world_height_limit: i32) i64 {
    _ = world_height_limit;
    return 0;
}

fn caveVerticalBandEndY(world_height_limit: i32) i64 {
    const denominator = @as(f32, @floatFromInt(maxI64(1, @as(i64, world_height_limit) - 1)));
    const active_half_width = (1.0 - CAVE_ACTIVE_BAND_MIN) / 1.6;
    const upper = (0.45 + active_half_width) * denominator;
    const end_y: i64 = @intFromFloat(@ceil(upper));
    return minI64(@as(i64, world_height_limit), end_y + 2);
}

fn caveYMaybeActive(world_y: i64, world_height_limit: i32) bool {
    const world_limit = @as(i64, world_height_limit);
    if (world_y <= CAVE_BEDROCK_CLEARANCE) return false;
    if (world_y >= world_limit - 2) return false;
    if (world_y < caveVerticalBandStartY(world_height_limit)) return false;
    if (world_y >= caveVerticalBandEndY(world_height_limit)) return false;
    return true;
}

fn chunkAndNeighborPlanesHaveCaveActiveY(origin_y: i64, local_height: i32, world_height_limit: i32) bool {
    const world_limit = @as(i64, world_height_limit);
    const fill_start_y = maxI64(origin_y, 0);
    const fill_top_y = minI64(origin_y + @as(i64, local_height), world_limit);
    var cave_start_y = maxI64(fill_start_y, CAVE_BEDROCK_CLEARANCE + 1);
    cave_start_y = maxI64(cave_start_y, caveVerticalBandStartY(world_height_limit));
    var cave_end_y = minI64(fill_top_y, world_limit - 2);
    cave_end_y = minI64(cave_end_y, caveVerticalBandEndY(world_height_limit));
    if (cave_end_y > cave_start_y) return true;
    return caveYMaybeActive(origin_y - 1, world_height_limit) or
        caveYMaybeActive(origin_y + @as(i64, local_height), world_height_limit);
}

fn fillColumnWithOptionalCaves(
    blocks: [*]u8,
    voxel_materials: [*]u32,
    block_offset: usize,
    plane_cells: usize,
    column_index: usize,
    origin_y: i64,
    fill_start_y: i64,
    fill_end_y: i64,
    stone_limit: i64,
    dirt_limit: i64,
    surface_height: i64,
    surface_material: u32,
    world_x: i64,
    world_z: i64,
    seed: i64,
    world_height_limit: i32,
    carve_caves: bool,
) void {
    if (fill_end_y <= fill_start_y) return;
    if (!carve_caves) {
        fillSolidColumnNoCaves(
            blocks,
            voxel_materials,
            block_offset,
            plane_cells,
            column_index,
            origin_y,
            fill_start_y,
            fill_end_y,
            stone_limit,
            dirt_limit,
            surface_material,
        );
        return;
    }

    var cave_start_y = maxI64(fill_start_y, CAVE_BEDROCK_CLEARANCE + 1);
    cave_start_y = maxI64(cave_start_y, caveVerticalBandStartY(world_height_limit));
    var cave_end_y = minI64(fill_end_y, surface_height);
    cave_end_y = minI64(cave_end_y, @as(i64, world_height_limit) - 2);
    cave_end_y = minI64(cave_end_y, caveVerticalBandEndY(world_height_limit));

    var current_y = fill_start_y;
    if (cave_end_y > cave_start_y) {
        fillSolidColumnNoCaves(blocks, voxel_materials, block_offset, plane_cells, column_index, origin_y, current_y, cave_start_y, stone_limit, dirt_limit, surface_material);
        fillCaveCheckedRange(blocks, voxel_materials, block_offset, plane_cells, column_index, origin_y, maxI64(current_y, cave_start_y), cave_end_y, stone_limit, dirt_limit, surface_height, surface_material, world_x, world_z, seed, world_height_limit);
        current_y = cave_end_y;
    }

    fillSolidColumnNoCaves(blocks, voxel_materials, block_offset, plane_cells, column_index, origin_y, current_y, fill_end_y, stone_limit, dirt_limit, surface_material);
}

fn fillCaveCheckedRange(
    blocks: [*]u8,
    voxel_materials: [*]u32,
    block_offset: usize,
    plane_cells: usize,
    column_index: usize,
    origin_y: i64,
    start_y: i64,
    end_y: i64,
    stone_limit: i64,
    dirt_limit: i64,
    surface_height: i64,
    surface_material: u32,
    world_x: i64,
    world_z: i64,
    seed: i64,
    world_height_limit: i32,
) void {
    var world_y = start_y;
    while (world_y < end_y) : (world_y += 1) {
        if (shouldCarveCave(world_x, world_y, world_z, surface_height, seed, world_height_limit)) {
            continue;
        }

        const local_y = world_y - origin_y;
        const block_index = block_offset + @as(usize, @intCast(local_y)) * plane_cells + column_index;
        blocks[block_index] = 1;
        voxel_materials[block_index] = terrainMaterialFromProfile(
            world_y,
            stone_limit,
            dirt_limit,
            surface_material,
        );
    }
}

fn fillColumnWithOptionalCaves4(
    blocks: [*]u8,
    voxel_materials: [*]u32,
    block_offset: usize,
    plane_cells: usize,
    column_index: usize,
    origin_y: i64,
    fill_start_y: i64,
    fill_top_y: i64,
    surface_heights: VU,
    surface_materials: VU,
    world_x: VI,
    world_z: i64,
    seed: i64,
    world_height_limit: i32,
    cave_y_terms: ?[*]const CaveYTerm,
    cave_y_terms_len: usize,
) void {
    const surface_height_i: VI = @intCast(surface_heights);
    const fill_top_i32: i32 = @intCast(fill_top_y);
    const fill_end_i = @min(surface_height_i, @as(VI, @splat(fill_top_i32)));
    const stone_limit_i = surface_height_i - @as(VI, @splat(@as(i32, 4)));
    const dirt_limit_i = surface_height_i - @as(VI, @splat(@as(i32, 1)));
    const deep_start_y = maxI64(
        maxI64(fill_start_y, CAVE_BEDROCK_CLEARANCE + 1),
        caveVerticalBandStartY(world_height_limit),
    );
    const cave_start_i = @as(VI, @splat(@as(i32, @intCast(deep_start_y))));
    var cave_end_i = @min(fill_end_i, surface_height_i);
    cave_end_i = @min(cave_end_i, @as(VI, @splat(@as(i32, world_height_limit - 2))));
    cave_end_i = @min(cave_end_i, @as(VI, @splat(@as(i32, @intCast(caveVerticalBandEndY(world_height_limit))))));
    const cave_valid = cave_end_i > cave_start_i;
    if (!@reduce(.Or, cave_valid)) {
        fillSolidColumnsNoCaves4(
            blocks,
            voxel_materials,
            block_offset,
            plane_cells,
            column_index,
            origin_y,
            fill_start_y,
            fill_top_y,
            surface_heights,
            surface_materials,
        );
        return;
    }
    const active_cave_start_i = @select(i32, cave_valid, cave_start_i, fill_end_i);
    const active_cave_end_i = @select(i32, cave_valid, cave_end_i, fill_end_i);
    const cave_frequency = 0.018 * CAVE_FREQUENCY_SCALE;
    const world_x_f: VF = @floatFromInt(world_x);
    const world_z_f = @as(f32, @floatFromInt(world_z));
    const primary_x_term = caveNoiseXTerm(world_x_f, cave_frequency);
    const primary_z_term = caveNoiseZTerm(world_z_f, cave_frequency);
    const detail_x_term = caveNoiseXTerm(world_x_f, cave_frequency * CAVE_DETAIL_FREQUENCY_MULTIPLIER);
    const detail_z_term = caveNoiseZTerm(world_z_f, cave_frequency * CAVE_DETAIL_FREQUENCY_MULTIPLIER);

    if (deep_start_y > fill_start_y) {
        fillSolidColumnsNoCaves4(
            blocks,
            voxel_materials,
            block_offset,
            plane_cells,
            column_index,
            origin_y,
            fill_start_y,
            deep_start_y,
            surface_heights,
            surface_materials,
        );
    }

    const max_cave_end_y = @as(i64, @reduce(.Max, active_cave_end_i));
    var world_y = @as(i64, @reduce(.Min, active_cave_start_i));
    while (world_y < max_cave_end_y) : (world_y += 1) {
        const world_y_i32: i32 = @intCast(world_y);
        const active_y = @as(VI, @splat(world_y_i32));
        const active_mask = (active_y >= active_cave_start_i) & (active_y < active_cave_end_i);
        if (!@reduce(.Or, active_mask)) continue;

        const carve_mask = carve: {
            const local_y = world_y - origin_y;
            if (cave_y_terms) |terms| {
                if (local_y >= 0) {
                    const local_y_index: usize = @intCast(local_y);
                    if (local_y_index < cave_y_terms_len) {
                        break :carve shouldCarveCave4WithTerm(
                            surface_height_i,
                            seed,
                            terms[local_y_index],
                            primary_x_term,
                            primary_z_term,
                            detail_x_term,
                            detail_z_term,
                        ) & active_mask;
                    }
                }
            }
            break :carve shouldCarveCave4(
                world_x,
                world_y,
                world_z,
                surface_height_i,
                seed,
                world_height_limit,
            ) & active_mask;
        };
        const solid_mask = active_mask & !carve_mask;
        if (!@reduce(.Or, solid_mask)) continue;

        const material = @select(
            u32,
            @as(VI, @splat(world_y_i32)) < stone_limit_i,
            @as(VU, @splat(STONE)),
            @select(
                u32,
                @as(VI, @splat(world_y_i32)) < dirt_limit_i,
                @as(VU, @splat(DIRT)),
                surface_materials,
            ),
        );

        const local_y = world_y - origin_y;
        const block_index = block_offset + @as(usize, @intCast(local_y)) * plane_cells + column_index;
        if (@reduce(.And, solid_mask)) {
            inline for (0..4) |lane| {
                blocks[block_index + lane] = 1;
            }
            storeU32x4(voxel_materials + block_index, material);
        } else {
            inline for (0..4) |lane| {
                if (solid_mask[lane]) {
                    blocks[block_index + lane] = 1;
                    voxel_materials[block_index + lane] = material[lane];
                }
            }
        }
    }

    inline for (0..4) |lane| {
        const lane_fill_end = @as(i64, fill_end_i[lane]);
        const lane_cave_start = @as(i64, active_cave_start_i[lane]);
        const lane_cave_end = @as(i64, active_cave_end_i[lane]);
        const lane_stone_limit = @as(i64, stone_limit_i[lane]);
        const lane_dirt_limit = @as(i64, dirt_limit_i[lane]);
        const lane_material = surface_materials[lane];
        fillSolidColumnNoCaves(
            blocks,
            voxel_materials,
            block_offset,
            plane_cells,
            column_index + lane,
            origin_y,
            maxI64(lane_cave_start, lane_cave_end),
            lane_fill_end,
            lane_stone_limit,
            lane_dirt_limit,
            lane_material,
        );
    }
}

fn fillSolidColumnsNoCaves4(
    blocks: [*]u8,
    voxel_materials: [*]u32,
    block_offset: usize,
    plane_cells: usize,
    column_index: usize,
    origin_y: i64,
    fill_start_y: i64,
    fill_top_y: i64,
    surface_heights: VU,
    surface_materials: VU,
) void {
    const surface_height_i: VI = @intCast(surface_heights);
    const fill_top_i32: i32 = @intCast(fill_top_y);
    const fill_end_i = @min(surface_height_i, @as(VI, @splat(fill_top_i32)));
    const max_fill_end_y = @as(i64, @reduce(.Max, fill_end_i));
    if (max_fill_end_y <= fill_start_y) return;

    const stone_limit_i = surface_height_i - @as(VI, @splat(@as(i32, 4)));
    const dirt_limit_i = surface_height_i - @as(VI, @splat(@as(i32, 1)));
    var world_y = fill_start_y;
    while (world_y < max_fill_end_y) : (world_y += 1) {
        const world_y_i32: i32 = @intCast(world_y);
        const active_mask = @as(VI, @splat(world_y_i32)) < fill_end_i;
        if (!@reduce(.Or, active_mask)) continue;

        const material = if (world_y == 0)
            @as(VU, @splat(BEDROCK))
        else
            @select(
                u32,
                @as(VI, @splat(world_y_i32)) < stone_limit_i,
                @as(VU, @splat(STONE)),
                @select(
                    u32,
                    @as(VI, @splat(world_y_i32)) < dirt_limit_i,
                    @as(VU, @splat(DIRT)),
                    surface_materials,
                ),
            );

        const local_y = world_y - origin_y;
        const block_index = block_offset + @as(usize, @intCast(local_y)) * plane_cells + column_index;
        if (@reduce(.And, active_mask)) {
            inline for (0..4) |lane| {
                blocks[block_index + lane] = 1;
            }
            storeU32x4(voxel_materials + block_index, material);
        } else {
            inline for (0..4) |lane| {
                if (active_mask[lane]) {
                    blocks[block_index + lane] = 1;
                    voxel_materials[block_index + lane] = material[lane];
                }
            }
        }
    }
}

fn fillSolidColumnNoCaves(
    blocks: [*]u8,
    voxel_materials: [*]u32,
    block_offset: usize,
    plane_cells: usize,
    column_index: usize,
    origin_y: i64,
    fill_start_y: i64,
    fill_end_y: i64,
    stone_limit: i64,
    dirt_limit: i64,
    surface_material: u32,
) void {
    if (fill_start_y == 0 and fill_end_y > 0) {
        writeMaterialSegment(blocks, voxel_materials, block_offset, plane_cells, column_index, origin_y, 0, 1, BEDROCK);
    }
    writeMaterialSegment(
        blocks,
        voxel_materials,
        block_offset,
        plane_cells,
        column_index,
        origin_y,
        maxI64(fill_start_y, 1),
        minI64(fill_end_y, stone_limit),
        STONE,
    );
    writeMaterialSegment(
        blocks,
        voxel_materials,
        block_offset,
        plane_cells,
        column_index,
        origin_y,
        maxI64(fill_start_y, stone_limit),
        minI64(fill_end_y, dirt_limit),
        DIRT,
    );
    writeMaterialSegment(
        blocks,
        voxel_materials,
        block_offset,
        plane_cells,
        column_index,
        origin_y,
        maxI64(fill_start_y, dirt_limit),
        fill_end_y,
        surface_material,
    );
}

fn writeMaterialSegment(
    blocks: [*]u8,
    voxel_materials: [*]u32,
    block_offset: usize,
    plane_cells: usize,
    column_index: usize,
    origin_y: i64,
    start_y: i64,
    end_y: i64,
    material: u32,
) void {
    if (end_y <= start_y) return;
    var world_y = start_y;
    while (world_y < end_y) : (world_y += 1) {
        const local_y = world_y - origin_y;
        const block_index = block_offset + @as(usize, @intCast(local_y)) * plane_cells + column_index;
        blocks[block_index] = 1;
        voxel_materials[block_index] = material;
    }
}

fn terrainMaterialFromProfile(world_y: i64, stone_limit: i64, dirt_limit: i64, surface_material: u32) u32 {
    if (world_y == 0) return BEDROCK;
    if (world_y < stone_limit) return STONE;
    if (world_y < dirt_limit) return DIRT;
    return surface_material;
}

fn valueNoise2D(x_in: f32, y_in: f32, seed: i64, frequency: f32) f32 {
    const x = x_in * frequency;
    const y = y_in * frequency;

    const x0 = @floor(x);
    const y0 = @floor(y);
    const xf = x - x0;
    const yf = y - y0;

    const ix0: i64 = @intFromFloat(x0);
    const iy0: i64 = @intFromFloat(y0);
    const ix1 = ix0 + 1;
    const iy1 = iy0 + 1;

    const v00 = hash2(ix0, iy0, seed);
    const v10 = hash2(ix1, iy0, seed);
    const v01 = hash2(ix0, iy1, seed);
    const v11 = hash2(ix1, iy1, seed);

    const u = fade(xf);
    const v = fade(yf);
    const nx0 = lerp(v00, v10, u);
    const nx1 = lerp(v01, v11, u);
    return lerp(nx0, nx1, v) * 2.0 - 1.0;
}

fn valueNoise2D4(x_in: VF, y_in: VF, seed: i64, frequency: f32) VF {
    const frequency_v = @as(VF, @splat(frequency));
    const x = x_in * frequency_v;
    const y = y_in * frequency_v;

    const x0 = @floor(x);
    const y0 = @floor(y);
    const xf = x - x0;
    const yf = y - y0;

    const ix0: VI = @intFromFloat(x0);
    const iy0: VI = @intFromFloat(y0);
    const ix1 = ix0 + @as(VI, @splat(1));
    const iy1 = iy0 + @as(VI, @splat(1));

    const v00 = hash2x4(ix0, iy0, seed);
    const v10 = hash2x4(ix1, iy0, seed);
    const v01 = hash2x4(ix0, iy1, seed);
    const v11 = hash2x4(ix1, iy1, seed);

    const u = fade4(xf);
    const v = fade4(yf);
    const nx0 = lerp4(v00, v10, u);
    const nx1 = lerp4(v01, v11, u);
    return lerp4(nx0, nx1, v) * @as(VF, @splat(@as(f32, 2.0))) - @as(VF, @splat(@as(f32, 1.0)));
}

fn valueNoise3D(x_in: f32, y_in: f32, z_in: f32, seed: i64, frequency: f32) f32 {
    const x = x_in * frequency;
    const y = y_in * frequency;
    const z = z_in * frequency;

    const x0 = @floor(x);
    const y0 = @floor(y);
    const z0 = @floor(z);
    const xf = x - x0;
    const yf = y - y0;
    const zf = z - z0;

    const ix0: i64 = @intFromFloat(x0);
    const iy0: i64 = @intFromFloat(y0);
    const iz0: i64 = @intFromFloat(z0);
    const ix1 = ix0 + 1;
    const iy1 = iy0 + 1;
    const iz1 = iz0 + 1;

    const u = fade(xf);
    const v = fade(yf);
    const w = fade(zf);

    const c000 = hash3(ix0, iy0, iz0, seed);
    const c100 = hash3(ix1, iy0, iz0, seed);
    const c010 = hash3(ix0, iy1, iz0, seed);
    const c110 = hash3(ix1, iy1, iz0, seed);
    const c001 = hash3(ix0, iy0, iz1, seed);
    const c101 = hash3(ix1, iy0, iz1, seed);
    const c011 = hash3(ix0, iy1, iz1, seed);
    const c111 = hash3(ix1, iy1, iz1, seed);

    const x00 = lerp(c000, c100, u);
    const x10 = lerp(c010, c110, u);
    const x01 = lerp(c001, c101, u);
    const x11 = lerp(c011, c111, u);
    const y0v = lerp(x00, x10, v);
    const y1v = lerp(x01, x11, v);
    return lerp(y0v, y1v, w) * 2.0 - 1.0;
}

fn valueNoise3D4(x_in: VF, y_in: VF, z_in: VF, seed: i64, frequency: f32) VF {
    const frequency_v = @as(VF, @splat(frequency));
    const x = x_in * frequency_v;
    const y = y_in * frequency_v;
    const z = z_in * frequency_v;

    const x0 = @floor(x);
    const y0 = @floor(y);
    const z0 = @floor(z);
    const xf = x - x0;
    const yf = y - y0;
    const zf = z - z0;

    const ix0: VI = @intFromFloat(x0);
    const iy0: VI = @intFromFloat(y0);
    const iz0: VI = @intFromFloat(z0);
    const ix1 = ix0 + @as(VI, @splat(1));
    const iy1 = iy0 + @as(VI, @splat(1));
    const iz1 = iz0 + @as(VI, @splat(1));

    const u = fade4(xf);
    const v = fade4(yf);
    const w = fade4(zf);

    const c000 = hash3x4(ix0, iy0, iz0, seed);
    const c100 = hash3x4(ix1, iy0, iz0, seed);
    const c010 = hash3x4(ix0, iy1, iz0, seed);
    const c110 = hash3x4(ix1, iy1, iz0, seed);
    const c001 = hash3x4(ix0, iy0, iz1, seed);
    const c101 = hash3x4(ix1, iy0, iz1, seed);
    const c011 = hash3x4(ix0, iy1, iz1, seed);
    const c111 = hash3x4(ix1, iy1, iz1, seed);

    const x00 = lerp4(c000, c100, u);
    const x10 = lerp4(c010, c110, u);
    const x01 = lerp4(c001, c101, u);
    const x11 = lerp4(c011, c111, u);
    const y0v = lerp4(x00, x10, v);
    const y1v = lerp4(x01, x11, v);
    return lerp4(y0v, y1v, w) * @as(VF, @splat(@as(f32, 2.0))) - @as(VF, @splat(@as(f32, 1.0)));
}

fn valueNoise3D4WithYTerm(x_in: VF, z_in: f32, seed: i64, frequency: f32, y_term: CaveNoiseYTerm) VF {
    const frequency_v = @as(VF, @splat(frequency));
    const x = x_in * frequency_v;
    const z = z_in * frequency;

    const x0 = @floor(x);
    const z0 = @floor(z);
    const xf = x - x0;
    const zf = z - z0;

    const ix0: VI = @intFromFloat(x0);
    const iz0 = @as(VI, @splat(@as(i32, @intFromFloat(z0))));
    const ix1 = ix0 + @as(VI, @splat(1));
    const iz1 = iz0 + @as(VI, @splat(1));

    const u = fade4(xf);
    const w = @as(VF, @splat(fade(zf)));

    const c000 = hash3x4(ix0, y_term.iy0, iz0, seed);
    const c100 = hash3x4(ix1, y_term.iy0, iz0, seed);
    const c010 = hash3x4(ix0, y_term.iy1, iz0, seed);
    const c110 = hash3x4(ix1, y_term.iy1, iz0, seed);
    const c001 = hash3x4(ix0, y_term.iy0, iz1, seed);
    const c101 = hash3x4(ix1, y_term.iy0, iz1, seed);
    const c011 = hash3x4(ix0, y_term.iy1, iz1, seed);
    const c111 = hash3x4(ix1, y_term.iy1, iz1, seed);

    const x00 = lerp4(c000, c100, u);
    const x10 = lerp4(c010, c110, u);
    const x01 = lerp4(c001, c101, u);
    const x11 = lerp4(c011, c111, u);
    const y0v = lerp4(x00, x10, y_term.v);
    const y1v = lerp4(x01, x11, y_term.v);
    return lerp4(y0v, y1v, w) * @as(VF, @splat(@as(f32, 2.0))) - @as(VF, @splat(@as(f32, 1.0)));
}

fn valueNoise3D4WithTerms(seed: i64, x_term: CaveNoiseXTerm, y_term: CaveNoiseYTerm, z_term: CaveNoiseZTerm) VF {
    const c000 = hash3x4(x_term.ix0, y_term.iy0, z_term.iz0, seed);
    const c100 = hash3x4(x_term.ix1, y_term.iy0, z_term.iz0, seed);
    const c010 = hash3x4(x_term.ix0, y_term.iy1, z_term.iz0, seed);
    const c110 = hash3x4(x_term.ix1, y_term.iy1, z_term.iz0, seed);
    const c001 = hash3x4(x_term.ix0, y_term.iy0, z_term.iz1, seed);
    const c101 = hash3x4(x_term.ix1, y_term.iy0, z_term.iz1, seed);
    const c011 = hash3x4(x_term.ix0, y_term.iy1, z_term.iz1, seed);
    const c111 = hash3x4(x_term.ix1, y_term.iy1, z_term.iz1, seed);

    const x00 = lerp4(c000, c100, x_term.u);
    const x10 = lerp4(c010, c110, x_term.u);
    const x01 = lerp4(c001, c101, x_term.u);
    const x11 = lerp4(c011, c111, x_term.u);
    const y0v = lerp4(x00, x10, y_term.v);
    const y1v = lerp4(x01, x11, y_term.v);
    return lerp4(y0v, y1v, z_term.w) * @as(VF, @splat(@as(f32, 2.0))) - @as(VF, @splat(@as(f32, 1.0)));
}

fn hash2(ix: i64, iy: i64, seed: i64) f32 {
    var h = u32FromI64(ix) *% 0x9E3779B9;
    h = (h ^ (u32FromI64(iy) *% 0x85EBCA6B));
    h = (h ^ (u32FromI64(seed) *% 0xC2B2AE35));
    h = mixU32(h);
    return @as(f32, @floatFromInt(h & 0x00FFFFFF)) / 16777215.0;
}

fn hash2x4(ix: VI, iy: VI, seed: i64) VF {
    var h = u32FromI32x4(ix) *% @as(VU, @splat(@as(u32, 0x9E3779B9)));
    h = h ^ (u32FromI32x4(iy) *% @as(VU, @splat(@as(u32, 0x85EBCA6B))));
    h = h ^ @as(VU, @splat(u32FromI64(seed) *% 0xC2B2AE35));
    h = mixU32x4(h);
    return @as(VF, @floatFromInt(h & @as(VU, @splat(@as(u32, 0x00FFFFFF))))) / @as(VF, @splat(@as(f32, 16777215.0)));
}

fn hash3(ix: i64, iy: i64, iz: i64, seed: i64) f32 {
    var h = u32FromI64(ix) *% 0x9E3779B9;
    h = (h ^ (u32FromI64(iy) *% 0x85EBCA6B));
    h = (h ^ (u32FromI64(iz) *% 0xC2B2AE35));
    h = (h ^ (u32FromI64(seed) *% 0x27D4EB2F));
    h = mixU32(h);
    return @as(f32, @floatFromInt(h & 0x00FFFFFF)) / 16777215.0;
}

fn hash3x4(ix: VI, iy: VI, iz: VI, seed: i64) VF {
    var h = u32FromI32x4(ix) *% @as(VU, @splat(@as(u32, 0x9E3779B9)));
    h = h ^ (u32FromI32x4(iy) *% @as(VU, @splat(@as(u32, 0x85EBCA6B))));
    h = h ^ (u32FromI32x4(iz) *% @as(VU, @splat(@as(u32, 0xC2B2AE35))));
    h = h ^ @as(VU, @splat(u32FromI64(seed) *% 0x27D4EB2F));
    h = mixU32x4(h);
    return @as(VF, @floatFromInt(h & @as(VU, @splat(@as(u32, 0x00FFFFFF))))) / @as(VF, @splat(@as(f32, 16777215.0)));
}

fn mixU32(value_in: u32) u32 {
    var value = value_in;
    value = value ^ (value >> 16);
    value *%= 0x7FEB352D;
    value = value ^ (value >> 15);
    value *%= 0x846CA68B;
    value = value ^ (value >> 16);
    return value;
}

fn mixU32x4(value_in: VU) VU {
    var value = value_in;
    value = value ^ (value >> @as(VU, @splat(@as(u32, 16))));
    value *%= @as(VU, @splat(@as(u32, 0x7FEB352D)));
    value = value ^ (value >> @as(VU, @splat(@as(u32, 15))));
    value *%= @as(VU, @splat(@as(u32, 0x846CA68B)));
    value = value ^ (value >> @as(VU, @splat(@as(u32, 16))));
    return value;
}

fn u32FromI64(value: i64) u32 {
    return @truncate(@as(u64, @bitCast(value)));
}

fn u32FromI32x4(value: VI) VU {
    return @bitCast(value);
}

fn fade(t: f32) f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn fade4(t: VF) VF {
    return t * t * t * (t * (t * @as(VF, @splat(@as(f32, 6.0))) - @as(VF, @splat(@as(f32, 15.0)))) + @as(VF, @splat(@as(f32, 10.0))));
}

fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

fn lerp4(a: VF, b: VF, t: VF) VF {
    return a + (b - a) * t;
}

fn loadU32x4(pointer: [*]const u32) VU {
    const vector_pointer: *align(4) const VU = @ptrCast(pointer);
    return vector_pointer.*;
}

fn storeU32x4(pointer: [*]u32, value: VU) void {
    const vector_pointer: *align(4) VU = @ptrCast(pointer);
    vector_pointer.* = value;
}

fn clamp01(value: f32) f32 {
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

fn absF32(value: f32) f32 {
    return if (value < 0.0) -value else value;
}

fn maxI32(a: i32, b: i32) i32 {
    return if (a > b) a else b;
}

fn maxI64(a: i64, b: i64) i64 {
    return if (a > b) a else b;
}

fn minI64(a: i64, b: i64) i64 {
    return if (a < b) a else b;
}

fn maxUsize(a: usize, b: usize) usize {
    return if (a > b) a else b;
}

fn minUsize(a: usize, b: usize) usize {
    return if (a < b) a else b;
}
