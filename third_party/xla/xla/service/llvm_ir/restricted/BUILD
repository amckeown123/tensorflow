load("//xla:xla.default.bzl", "xla_cc_test")

package(
    default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/service/llvm_ir:friends"],
    licenses = ["notice"],
    transitive_visibility = "//xla:restricted",
)

xla_cc_test(
    name = "llvm_util_test",
    srcs = ["llvm_util_test.cc"],
    tags = ["pjrt_migration_candidate"],
    deps = [
        "//xla:error_spec",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tests:xla_internal_test_main",
        "//xla/tests/restricted:hlo_test_base",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:ir_headers",
    ],
)
