import argparse
import hashlib
import os
import pathlib


def calc_checksum(filename: pathlib.Path):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        while chunk := f.read(4096):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


# find this section in Package.swift and replace the values for url and checksum
# 'url:' has to be converted to 'path:' to work with a filepath
#
#   .binaryTarget(name: "onnxruntime",
#      url: "https://onnxruntimepackages.z14.web.core.windows.net/pod-archive-onnxruntime-c-1.14.0.zip",
#      checksum: "c89cd106ff02eb3892243acd7c4f2bd8e68c2c94f2751b5e35f98722e10c042b"),
#

def update_swift_package(spm_config_path: pathlib.Path, ort_package_path: pathlib.Path):
    updated = False
    new_config = []
    checksum = calc_checksum(ort_package_path)

    with open(spm_config_path, "r") as config:
        while line := config.readline():
            if '.binaryTarget(name: "onnxruntime"' in line:
                # get and update the following 2 lines with url and checksum
                url_line = config.readline()
                checksum_line = config.readline()
                assert "url:" in url_line
                assert "checksum:" in checksum_line

                start_url = url_line.find('url:')
                start_checksum_value = checksum_line.find('"')
                new_url_line = url_line[: start_url] + f'file: "{ort_package_path}",\n'
                new_checksum_line = checksum_line[: start_checksum_value + 1] + checksum + '"),\n'

                new_config.append(line)
                new_config.append(new_url_line)
                new_config.append(new_checksum_line)
                updated = True
            else:
                new_config.append(line)

    assert updated

    # override with new content
    with open(spm_config_path, "w") as new_config_f:
        for line in new_config:
            new_config_f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        f"{os.path.basename(__file__)}",
        description="Update the ORT binary used in the Package.swift config file.",
    )

    parser.add_argument(
        "--spm_config",
        type=pathlib.Path,
        required=True,
        help="Full path to Package.swift, the Swift Package Manager config.",
    )

    parser.add_argument(
        "--ort_package",
        type=pathlib.Path,
        required=True,
        help="ORT native iOS pod to use. Filename should be something like pod-archive-onnxruntime-c-x.y.z.zip",
    )

    args = parser.parse_args()
    spm_config = args.spm_config.resolve(strict=True)
    ort_package = args.ort_package.resolve(strict=True)

    update_swift_package(spm_config, ort_package)
