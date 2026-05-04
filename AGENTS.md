# AGENTS.md

Use `.venv/bin/python` for Minechunk Python commands.

Run WGPU terrain/meshing code:

```bash
.venv/bin/python main.py --renderer-backend wgpu --terrain-backend wgpu --meshing-backend wgpu
```

Run Metal terrain/meshing code:

```bash
.venv/bin/python main.py --renderer-backend wgpu --terrain-backend metal --meshing-backend metal
```

Optional Zig terrain path:

```bash
.venv/bin/python tools/build_zig_terrain.py
```
