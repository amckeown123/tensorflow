load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/tests:build_defs.bzl", "xla_test")

package(
    default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/tests:friends"],
    licenses = ["notice"],
    transitive_visibility = "//xla:restricted",
)

cc_library(
    name = "hlo_test_base_legacy",
    testonly = True,
    srcs = ["hlo_test_base_legacy.cc"],
    hdrs = ["hlo_test_base_legacy.h"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/service:backend",
        "//xla/service:compiler",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_runner_interface",
        "//xla/service:interpreter_plugin",
        "//xla/service:platform_util",
        "//xla/service/restricted:hlo_runner_legacy",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream_executor_address_allocator",
        "//xla/tests:hlo_runner_agnostic_reference_mixin",
        "//xla/tests:hlo_runner_agnostic_test_base",
        "//xla/tests:pjrt_client_registry",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "hlo_test_base",
    testonly = True,
    hdrs = ["hlo_test_base.h"],
    deps = [
        ":hlo_test_base_legacy",
        "//xla:util",
        "//xla/service/restricted:hlo_runner",
        "//xla/service/restricted:hlo_runner_legacy",
        "//xla/stream_executor:platform",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "llvm_irgen_test_base",
    testonly = True,
    srcs = ["llvm_irgen_test_base.cc"],
    hdrs = ["llvm_irgen_test_base.h"],
    deps = [
        ":hlo_test_base_legacy",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:filecheck",
        "//xla/service:executable",
        "//xla/service:llvm_compiler",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@tsl//tsl/platform:test",
    ],
)

xla_test(
    name = "llvm_compiler_test",
    srcs = ["llvm_compiler_test.cc"],
    backends = [
        "cpu",
        "gpu",
    ],
    tags = ["pjrt_migration_candidate"],
    use_legacy_runtime = True,
    deps = [
        ":hlo_test_base",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:test_helpers",
        "//xla/service:backend",
        "//xla/service:llvm_compiler",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Core",
        "@tsl//tsl/platform:casts",
        "@tsl//tsl/platform:test",
    ],
)
