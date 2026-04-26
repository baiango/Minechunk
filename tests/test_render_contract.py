from __future__ import annotations

from types import SimpleNamespace

from engine import render_contract


def test_align_up_handles_common_allocator_boundaries():
    assert render_contract.align_up(0, 256) == 0
    assert render_contract.align_up(1, 256) == 256
    assert render_contract.align_up(255, 256) == 256
    assert render_contract.align_up(256, 256) == 256
    assert render_contract.align_up(257, 256) == 512
    assert render_contract.align_up(19, 1) == 19
    assert render_contract.align_up(19, 0) == 19


def test_device_limit_supports_mapping_and_attribute_shapes():
    mapping_device = SimpleNamespace(limits={"min_storage_buffer_offset_alignment": "512"})
    assert render_contract.device_limit(mapping_device, "min_storage_buffer_offset_alignment", 256) == 512
    assert render_contract.device_limit(mapping_device, "missing", 128) == 128

    attr_device = SimpleNamespace(limits=SimpleNamespace(max_storage_buffers_per_shader_stage=10))
    assert render_contract.device_limit(attr_device, "max_storage_buffers_per_shader_stage", 0) == 10


def test_describe_adapter_prefers_structured_info_then_summary():
    adapter = SimpleNamespace(
        info={
            "backend_type": "Metal",
            "adapter_type": "DiscreteGPU",
            "description": "Test GPU",
        },
        summary="fallback summary",
    )
    assert render_contract.describe_adapter(adapter) == "Metal / DiscreteGPU / Test GPU"

    summary_only = SimpleNamespace(info=None, summary="Only Summary")
    assert render_contract.describe_adapter(summary_only) == "Only Summary"

    empty = SimpleNamespace(info=None, summary="")
    assert render_contract.describe_adapter(empty) == "unknown"
