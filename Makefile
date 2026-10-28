comma := ,
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 -Wall -Wextra -Wno-unused-parameter

CORE := src/registration.cpp src/estimate.cpp src/model.cpp
SRC  := src/main.cpp $(CORE)
HDR  := src/dewatermark.hpp

# OPENCV=system  the installed OpenCV (4.x or 5.x) found by pkg-config, linked dynamically
# OPENCV=static  a minimal OpenCV built into deps/ by scripts/build_deps.sh, linked statically
# OPENCV=shared  the same, built as shared libraries that are shipped next to the binary
OPENCV ?= system
CV_VER ?= 5.0.0
DEPS    = deps/install/$(CV_VER)
CV_PC  := $(if $(filter 4.%,$(CV_VER)),opencv4,opencv5)

deps_pc = PKG_CONFIG_LIBDIR=$(DEPS)/$(1)-$(2)/lib/pkgconfig pkg-config $(if $(filter static,$(2)),--static)

$(DEPS)/%/.done:
	CV_VER=$(CV_VER) scripts/build_deps.sh $*

ifeq ($(OPENCV),system)
SYS_PC        := $(shell pkg-config --exists opencv5 && echo opencv5 || echo opencv4)
AS_NEEDED     := $(if $(filter Darwin,$(shell uname -s)),-Wl$(comma)-dead_strip_dylibs,-Wl$(comma)--as-needed)
NATIVE_CFLAGS  = $(shell pkg-config --cflags $(SYS_PC))
NATIVE_LIBS    = $(AS_NEEDED) $(shell pkg-config --libs $(SYS_PC))
NATIVE_DEPS   :=
else
NATIVE_CFLAGS  = $(shell $(call deps_pc,native,$(OPENCV)) --cflags $(CV_PC))
NATIVE_LIBS    = $(shell $(call deps_pc,native,$(OPENCV)) --libs $(CV_PC)) \
                 $(if $(filter shared,$(OPENCV)),-Wl$(comma)-rpath$(comma)$(abspath $(DEPS)/native-shared/lib))
NATIVE_DEPS   := $(DEPS)/native-$(OPENCV)/.done
endif

all: build/dewatermark build/synth build/compare

build/dewatermark: $(SRC) $(HDR) $(NATIVE_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(NATIVE_CFLAGS) $(SRC) -o $@ $(LDFLAGS) $(NATIVE_LIBS)

build/synth: tools/synth.cpp $(CORE) $(HDR) $(NATIVE_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Isrc $(NATIVE_CFLAGS) tools/synth.cpp $(CORE) -o $@ $(LDFLAGS) $(NATIVE_LIBS)

build/compare: tools/compare.cpp $(NATIVE_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(NATIVE_CFLAGS) tools/compare.cpp -o $@ $(LDFLAGS) $(NATIVE_LIBS)

linux: build/dewatermark

PREFIX ?= /usr/local
BIN    ?= build/dewatermark
install: $(BIN)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(BIN) $(DESTDIR)$(PREFIX)/bin/dewatermark

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/dewatermark

REL := $(if $(filter shared,$(OPENCV)),shared,static)

MUSL_PREFIX ?= x86_64-linux-musl-
export MUSL_PREFIX
linux-musl: $(DEPS)/linux-musl-x86_64-static/.done
	@mkdir -p build/linux-musl-x86_64
	$(MUSL_PREFIX)g++ $(CXXFLAGS) $$($(call deps_pc,linux-musl-x86_64,static) --cflags $(CV_PC)) $(SRC) \
	    -o build/linux-musl-x86_64/dewatermark -static -s $$($(call deps_pc,linux-musl-x86_64,static) --libs $(CV_PC))

MINGW_PREFIX ?= x86_64-w64-mingw32-
WIN := build/windows-x86_64
windows: $(DEPS)/windows-x86_64-$(REL)/.done
	@mkdir -p $(WIN)
	$(MINGW_PREFIX)g++ $(CXXFLAGS) $$($(call deps_pc,windows-x86_64,$(REL)) --cflags $(CV_PC)) $(SRC) \
	    -o $(WIN)/dewatermark.exe -static-libgcc -static-libstdc++ $(if $(filter static,$(REL)),-static) \
	    $$($(call deps_pc,windows-x86_64,$(REL)) --libs $(CV_PC)) -lole32
	$(MINGW_PREFIX)strip $(WIN)/dewatermark.exe
	cp COPYING ACKNOWLEDGMENTS $(WIN)/
ifeq ($(REL),shared)
	cp $(DEPS)/windows-x86_64-shared/bin/*.dll $(WIN)/
	cp "$$($(MINGW_PREFIX)g++ -print-file-name=libwinpthread-1.dll)" $(WIN)/ 2>/dev/null || true
endif

MACOS_MIN ?= 11.0
SIGN_ID   ?= Developer ID Application
MAC := build/macos
# OpenCV 5's pkg-config file lists frameworks as -l<Name>.framework; the sed turns them into -framework.
macos: $(DEPS)/macos-arm64-$(REL)/.done $(DEPS)/macos-x86_64-$(REL)/.done
	@mkdir -p $(MAC)
	for a in arm64 x86_64; do \
	    clang++ -arch $$a -mmacosx-version-min=$(MACOS_MIN) $(CXXFLAGS) \
	        $$($(call deps_pc,macos-$$a,$(REL)) --cflags $(CV_PC)) $(SRC) -o $(MAC)/dewatermark-$$a \
	        $$($(call deps_pc,macos-$$a,$(REL)) --libs $(CV_PC) | sed -E 's/-l([A-Za-z]+)\.framework/-framework \1/g') \
	        -Wl,-dead_strip_dylibs $(if $(filter shared,$(REL)),-Wl$(comma)-rpath$(comma)@executable_path/lib) || exit 1; \
	done
	lipo -create $(MAC)/dewatermark-arm64 $(MAC)/dewatermark-x86_64 -output $(MAC)/dewatermark
	rm $(MAC)/dewatermark-arm64 $(MAC)/dewatermark-x86_64
	strip -x $(MAC)/dewatermark
ifeq ($(REL),shared)
	rm -rf $(MAC)/lib && cp -a $(DEPS)/macos-arm64-shared/lib $(MAC)/lib && rm -rf $(MAC)/lib/pkgconfig $(MAC)/lib/cmake
	for f in $(MAC)/lib/*.dylib; do [ -L "$$f" ] || { \
	    lipo -create "$$f" "$(DEPS)/macos-x86_64-shared/lib/$$(basename $$f)" -output "$$f.u" && mv "$$f.u" "$$f" && \
	    codesign --force --options runtime --timestamp --sign "$(SIGN_ID)" "$$f"; }; done
endif
	codesign --force --options runtime --timestamp --sign "$(SIGN_ID)" $(MAC)/dewatermark
	codesign --verify --strict --verbose=2 $(MAC)/dewatermark

notarize:
	cd $(MAC) && rm -f dewatermark-macos.zip && zip -qry dewatermark-macos.zip dewatermark $(if $(wildcard $(MAC)/lib),lib)
	xcrun notarytool submit $(MAC)/dewatermark-macos.zip --keychain-profile "$(NOTARY_PROFILE)" --wait

VERSION ?= 1.1
TAR     := $(shell command -v gtar || echo tar)
DIST    := build/dist/dewatermark-$(VERSION).tar.gz

REPO    ?= https://github.com/dbenjaminmiller/dewatermark

dist:
	rm -rf build/dist && mkdir -p build/dist
	curl -fsSL -o $(DIST) $(REPO)/archive/refs/tags/v$(VERSION).tar.gz

# Debian source packages for a Launchpad PPA: one shared orig tarball, one upload per series.
DEBREV  ?= 1
SERIES  ?= jammy noble
DEB     := build/deb
deb-src: dist
	rm -rf $(DEB) && mkdir -p $(DEB)
	cp $(DIST) $(DEB)/dewatermark_$(VERSION).orig.tar.gz
	$(TAR) -C $(DEB) -xzf $(DIST)
	first=-sa; for s in $(SERIES); do \
	    v=$(VERSION)-$(DEBREV)~$${s}1; \
	    rm -rf $(DEB)/dewatermark-$(VERSION)/debian && cp -R debian $(DEB)/dewatermark-$(VERSION)/ && \
	    sed -e "1s/.*/dewatermark ($$v) $$s; urgency=medium/" debian/changelog > $(DEB)/dewatermark-$(VERSION)/debian/changelog && \
	    (cd $(DEB) && dpkg-source -b dewatermark-$(VERSION)) && \
	    (cd $(DEB)/dewatermark-$(VERSION) && dpkg-genchanges -S $$first > ../dewatermark_$${v}_source.changes) || exit 1; \
	    first=-sd; \
	done

# Source RPM for Fedora Copr.
srpm: dist
	rm -rf build/rpm && mkdir -p build/rpm
	rpmbuild -bs --define "_rpmformat 4" --define "_sourcedir $(abspath build/dist)" --define "_srcrpmdir $(abspath build/rpm)" \
	    --define "_topdir $(abspath build/rpm/top)" dewatermark.spec

clean:
	rm -rf build

.PHONY: all linux install uninstall linux-musl windows macos notarize dist deb-src srpm clean
