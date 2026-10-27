#!/usr/bin/env bash
# usage: build_deps.sh <native|macos-arm64|macos-x86_64|windows-x86_64|linux-musl-x86_64>-<static|shared>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CV_VER="${CV_VER:-5.0.0}"
SRC="$ROOT/deps/opencv-$CV_VER"
export CMAKE_POLICY_VERSION_MINIMUM=3.5  # OpenCV 4.x helper scripts predate CMake 4
[ -d "$SRC" ] || git clone --depth 1 --branch "$CV_VER" https://github.com/opencv/opencv.git "$SRC"

COMMON=(
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_INSTALL_LIBDIR=lib
  -DBUILD_LIST=core,imgproc,imgcodecs
  -DBUILD_ZLIB=ON -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON
  -DWITH_JPEG=ON -DWITH_PNG=ON -DWITH_TIFF=ON
  -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_AVIF=OFF -DWITH_SPNG=OFF
  -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_SUNRASTER=OFF -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_PFM=OFF
  -DWITH_GDAL=OFF -DWITH_GDCM=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_1394=OFF
  -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_OPENCL=OFF -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF -DWITH_PROTOBUF=OFF
  -DWITH_ADE=OFF -DWITH_QUIRC=OFF -DWITH_FLATBUFFERS=OFF -DWITH_VA=OFF -DWITH_OBSENSOR=OFF
  -DWITH_OPENGL=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_WIN32UI=OFF -DWITH_CAROTENE=OFF -DWITH_KLEIDICV=OFF -DWITH_OPENVX=OFF
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF
  -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python2=OFF -DBUILD_DOCS=OFF -DCV_TRACE=OFF
)

for spec in "${@:-native-shared}"; do
  target=${spec%-*} link=${spec##*-}
  case $link in
    static) args=(-DBUILD_SHARED_LIBS=OFF) ;;
    shared) args=(-DBUILD_SHARED_LIBS=ON) ;;
    *) echo "usage: $0 <target>-<static|shared>" >&2; exit 1 ;;
  esac
  case $target in
    native) ;;
    macos-arm64|macos-x86_64)
      args+=(-DCMAKE_OSX_ARCHITECTURES=${target#macos-} -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 -DCMAKE_INSTALL_NAME_DIR=@rpath) ;;
    windows-x86_64)
      args+=(-DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/mingw-w64-x86_64.cmake"
             "-DCMAKE_SHARED_LINKER_FLAGS=-static-libgcc -static-libstdc++") ;;
    linux-musl-x86_64)
      args+=(-DCMAKE_TOOLCHAIN_FILE="$ROOT/cmake/musl-x86_64.cmake") ;;
    *) echo "unknown target $target" >&2; exit 1 ;;
  esac
  b="$ROOT/deps/build/$CV_VER/$spec" i="$ROOT/deps/install/$CV_VER/$spec"
  [ -f "$i/.done" ] && continue
  echo "== OpenCV $CV_VER for $spec (log: deps/build-$CV_VER-$spec.log)"
  mkdir -p "$ROOT/deps"
  cmake -S "$SRC" -B "$b" "${COMMON[@]}" "${args[@]}" -DCMAKE_INSTALL_PREFIX="$i" > "$ROOT/deps/build-$CV_VER-$spec.log" 2>&1
  cmake --build "$b" --target install >> "$ROOT/deps/build-$CV_VER-$spec.log" 2>&1
  touch "$i/.done"
done
