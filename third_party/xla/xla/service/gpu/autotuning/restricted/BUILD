load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")

package(
    default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/service/gpu/autotuning:friends"],
    licenses = ["notice"],
    transitive_visibility = "//xla:restricted",
)

xla_test(
    name = "redzone_buffers_test",
    srcs = ["redzone_buffers_test.cc"],
    backends = ["gpu"],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:platform_util",
        "//xla/service/gpu/autotuning:redzone_buffers",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_memory_allocator",
        "//xla/tests/restricted:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "autotuner_cache_test",
    srcs = ["autotuner_cache_test.cc"],
    data = ["//xla/backends/gpu/target_config:all_gpu_specs"],
    tags = [
        "gpu",
        "pjrt_migration_candidate",
    ],
    deps = [
        "//xla:autotune_results_proto_cc",
        "//xla:autotuning_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:dump",
        "//xla/service/gpu/autotuning:autotune_cache_key",
        "//xla/service/gpu/autotuning:autotuner_cache",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor/host:host_platform",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
    ],
)
