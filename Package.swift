// swift-tools-version: 5.7
//   The swift-tools-version declares the minimum version of Swift required to build this package and MUST be the first
//   line of this file.
//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// A user of the Swift Package Manager (SPM) package will consume this file directly from the ORT github repository.
// For context, the end user's config will look something like:
//
//     dependencies: [
//       .package(url: "https://github.com/microsoft/onnxruntime", branch: "rel-1.14.0"),
//       ...
//     ],
//
// NOTE: The direct consumption creates a somewhat complicated setup to 'release' a new version of the ORT SPM package.
//  PROPOSED steps in release process:
//   - When the builds for the release are all good, update Package.swift to set the url/checksum
//     - the url should be the location where we will publish the release. it will not yet be valid
//     - the checksum should be of the pod-archive-onnxruntime-c-*.zip artifact from the last build
//       - any changes to the source code will invalidate this checksum
//   - Check in Package.swift to the release branch and perform one more final set of builds

import PackageDescription

let package = Package(
    name: "onnxruntime",
    platforms: [.iOS(.v11)],
    products: [
        .library(name: "onnxruntime",
                 type: .static,
                 targets: ["OnnxRuntimeBindings"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "OnnxRuntimeBindings",
                dependencies: ["onnxruntime"],
                path: "objectivec",
                exclude: ["test", "docs"],
                cxxSettings: [
                    .unsafeFlags(["-std=c++17",
                                  "-fobjc-arc-exceptions"
                                 ]),
                ], linkerSettings: [
                    .unsafeFlags(["-ObjC"]),
                ]),
        .testTarget(name: "OnnxRuntimeBindingsTests",
                    dependencies: ["OnnxRuntimeBindings"],
                    path: "swift/OnnxRuntimeBindingsTests",
                    resources: [
                        .copy("Resources/single_add.basic.ort")
                    ]),
    ]
)

// Add the ORT iOS Pod archive as a binary target.
//
// There are 3 scenarios:
//
// Release branch/tag of ORT github repo:
//    Target will be released pod archive and its checksum.
//
// `main` branch of ORT github repo:
//    Invalid. We do not have a pod archive that is guaranteed to work with the latest code on main as the objective-c
//    bindings may have changed.
//
// CI or local testing where you have built/obtained the iOS Pod archive matching the current source code.
//    Requires the ORT_IOS_POD_LOCAL_PATH environment variable to be set to specify the location of the pod.
//    This MUST be a relative path to this Package.swift file.
if ProcessInfo.processInfo.environment["ORT_IOS_POD_LOCAL_PATH"] != nil {
    // ORT_IOS_POD_LOCAL_PATH should be a path that is relative to Package.swift.
    //
    // To build locally, tools/ci_build/github/apple/build_and_assemble_ios_pods.py can be used
    // See https://onnxruntime.ai/docs/build/custom.html#ios
    //  Example command:
    //    python3 tools/ci_build/github/apple/build_and_assemble_ios_pods.py \
    //      --variant Full \
    //      --build-settings-file tools/ci_build/github/apple/default_full_ios_framework_build_settings.json
    //
    // This should produce the pod archive in build/ios_pod_staging, and ORT_IOS_POD_LOCAL_PATH can be set to
    // "build/ios_pod_staging/pod-archive-onnxruntime-c-???.zip" where '???' is replaced by the version info in the
    // actual filename.
    package.targets += [
        .binaryTarget(name: "onnxruntime",
                      file: ProcessInfo.processInfo.environment["ORT_IOS_POD_LOCAL_PATH"])
    ]
} else {
    fatalError("Please use a release branch from https://github.com/microsoft/onnxruntime. It is not valid to use `main` or a non-release branch.")
    // When creating the release version, remove the fatalError, uncomment the below, update the version info for the url
    // and insert the checksum info from the CI output or by downloading the pod archive artifact from the CI
    // and running `shasum -a 256 <path to pod zip>`.
    // The checksum should look something like checksum: "c89cd106ff02eb3892243acd7c4f2bd8e68c2c94f2751b5e35f98722e10c042b"
    //
    // package.targets += [
    //     .binaryTarget(name: "onnxruntime",
    //                   url: "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-c-<major.minor.patch>.zip",
    //                   checksum: "Insert checksum here")
    // ]

}
