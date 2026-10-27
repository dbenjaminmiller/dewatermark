# dewatermark

A tool for removing translucent watermarks from a directory of images.

## Basic usage

```
dpdfcomp [options] input.pdf output.pdf
```

The program will automatically create a subdirectory `dewatermark_out` for output TIFFs. If an existing output subdirectory is present, a suffix (`-1`, `-2`, etc.) will be added.

## Options

* `--out [directory]`: Choose custom output directory.
* `--save-watermark [x]`: Save watermark info as `[x].yml` and `[x]_{1-4}.tif`.
* `--load-watermark [x]`: Load watermark from `[x].yml` and `[x]_{1-4}.tif` (skips estimation).
* `--max-est [n]`: Maximum number of pages [n] used to estimate the watermark.
* `--search [n]`: Search for watermarks shifted by up to [n] pixels (default: 64).
* `--overwrite`: Overwrite existing outputs (if present).
* `--help`: Show help.

# Installation

* MacOS users can install via [Homebrew](https://brew.sh): `brew install dbenjaminmiller/tap/dewatermark`.
* Windows users can download a binary build from the "Releases" tab.
* Linux users can install from source. Repositories are available for Fedora and Ubuntu.

# Building

## MacOS

```
xcode-select --install
brew install opencv pkg-config
make macos
```

## Linux

Example build configuration using Ubuntu:

```
sudo apt update
sudo apt install build-essential pkg-config libopencv-dev
make linux
```

## Windows

Install MSYS2 from https://www.msys2.org and open the "MSYS2 UCRT64" shell.

```
pacman -S --needed make git mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-pkgconf
make windows
```

You can also run `make windows` under Linux or MacOS to cross-compile if you have the proper MinGW toolchain installed.