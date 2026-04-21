load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")

package(
    default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//xla:friends",
        "//xla/service/gpu:__subpackages__",
    ],
    licenses = ["notice"],
    transitive_visibility = "//xla:restricted",
)

xla_cc_test(
    name = "sol_latency_estimator_test",
    srcs = ["sol_latency_estimator_test.cc"],
    tags = ["pjrt_migration_candidate"],
    deps = [
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_module_config",
        "//xla/service:latency_hiding_scheduler",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu/model:collective_interpolator",
        "//xla/service/gpu/model:sol_gpu_cost_model",
        "//xla/service/gpu/model:sol_latency_estimator",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
    ],
)

xla_test(
    name = "analytical_latency_estimator_test",
    srcs = ["analytical_latency_estimator_test.cc"],
    backends = [
        "v100",
        "a100",
        "h100",
        "b200",
        "amdgpu_any",
    ],
    use_legacy_runtime = True,
    deps = [
        "//xla:shape_util",
        "//xla/backends/gpu/tests/restricted:gpu_codegen_test",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:latency_hiding_scheduler",
        "//xla/service/gpu:alias_info",
        "//xla/service/gpu:gpu_compiler",
        "//xla/service/gpu/model:analytical_latency_estimator",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/platform:casts",
    ],
)
