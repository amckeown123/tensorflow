load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

package(
    default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/backends/gpu/tests:friends"],
    licenses = ["notice"],
    transitive_visibility = "//xla:restricted",
)

cc_library(
    name = "gpu_codegen_test",
    testonly = True,
    srcs = ["gpu_codegen_test.cc"],
    hdrs = ["gpu_codegen_test.h"],
    tags = tf_gpu_tests_tags(),
    deps = [
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:executable",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor:platform_manager",
        "//xla/tests/restricted:llvm_irgen_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_for_library",
    ],
)

xla_test(
    name = "float_conversions_test",
    srcs = ["float_conversions_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_alignment_test",
    srcs = ["gpu_alignment_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_copy_alone_test",
    srcs = ["gpu_copy_alone_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/hlo/testlib:verified_hlo_module",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "gpu_dyn_shape_test",
    srcs = ["gpu_dyn_shape_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gpu_ftz_test",
    srcs = ["gpu_ftz_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:verified_hlo_module",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gpu_fused_mha_test",
    srcs = ["gpu_fused_mha_test.cc"],
    backend_tags = {
        "b200": ["broken"],
        "h100": ["broken"],
    },
    backends = [
        "a100",
        "h100",
        "b200",
        "amdgpu_any",
    ],
    shard_count = 2,
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:array4d",
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service/gpu:stream_executor_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/stream_executor/cuda:cuda_platform_id",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "gpu_int4_test",
    srcs = ["gpu_int4_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_ldg_test",
    srcs = ["gpu_ldg_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_noalias_test",
    srcs = ["gpu_noalias_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "kernel_launch_test",
    srcs = ["kernel_launch_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "pred_arithmetic_test",
    srcs = ["pred_arithmetic_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:literal_util",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "select_and_scatter_test",
    srcs = ["select_and_scatter_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_test(
    name = "swap_conv_operands_test",
    srcs = ["swap_conv_operands_test.cc"],
    backends = ["gpu"],
    tags = ["cuda-only"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "hlo_legacy_gpu_test_base",
    testonly = True,
    hdrs = ["hlo_legacy_gpu_test_base.h"],
    deps = [
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/target_config",
        "//xla/backends/gpu/tests:hlo_gpu_test_base_interface",
        "//xla/service:backend",
        "//xla/service:compiler",
        "//xla/service:interpreter_plugin",  # reference backend
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/tests/restricted:hlo_test_base_legacy",
        "@com_google_absl//absl/base:core_headers",
    ],
)

xla_test(
    name = "collective_pipeline_parallelism_test",
    srcs = ["collective_pipeline_parallelism_test.cc"],
    args = ["--xla_force_host_platform_device_count=4"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
        ],
    },
    backends = [
        "gpu",
    ],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:computation_placer",
        "//xla/service:hlo_module_config",
        "//xla/tests:literal_test_util",
        "//xla/tests:test_utils",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "collective_ops_command_buffer_test",
    srcs = ["collective_ops_command_buffer_test.cc"],
    backend_tags = {
        "gpu": [
            "multi_gpu",
            "no_oss",
        ],
    },
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        "//xla:literal",
        "//xla:literal_util",
        "//xla:xla_proto_cc",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner_interface",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/tests:literal_test_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

xla_test(
    name = "gpu_spmd_e2e_compile_test",
    size = "small",
    #TODO(b/450135639): Remove timeout override once the test is fixed.
    timeout = "moderate",
    srcs = ["gpu_spmd_e2e_compile_test.cc"],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:debug_options_flags",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:executable",
        "//xla/service:hlo_module_config",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "gpu_compilation_parallelism_test",
    srcs = [
        "gpu_compilation_parallelism_test.cc",
    ],
    backend_tags = {
        # TODO(b/445172709): Re-enable once fixed.
        "b200": ["broken"],
    },
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/hlo/testlib:verified_hlo_module",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_test(
    name = "reduction_vectorization_test",
    srcs = [
        "reduction_vectorization_test.cc",
    ],
    backends = ["gpu"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla:error_spec",
        "//xla:xla_proto_cc",
        "//xla/hlo/parser:hlo_parser",
        "//xla/stream_executor:device_description",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "gpu_too_many_blocks_test",
    srcs = [
        "gpu_too_many_blocks_test.cc",
    ],
    backends = ["gpu"],
    tags = ["cuda-only"],
    use_legacy_runtime = True,
    deps = [
        ":gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:statusor",
        "@tsl//tsl/platform:test",
    ],
)
