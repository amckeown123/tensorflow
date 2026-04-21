load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.bzl", "if_google")

package(
    default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/service/debug:friends"],
    licenses = ["notice"],
    transitive_visibility = "//xla:restricted",
)

xla_test(
    name = "float_check_device_test",
    srcs = ["float_check_device_test.cc"],
    backend_args = if_google(
        {
            "b200": ["--heap_check="],
            "h100": ["--heap_check="],
        },
        {},
    ),
    backends = [
        "h100",
        "b200",
    ],
    use_legacy_runtime = True,
    deps = [
        "//xla:xla_proto_cc",
        "//xla/backends/gpu/tests/restricted:gpu_codegen_test",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
