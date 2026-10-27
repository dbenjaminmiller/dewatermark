Name:           dewatermark
Version:        1.0
Release:        1%{?dist}
Summary:        Remove a translucent watermark shared by a set of page images
License:        0BSD
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  gcc-c++
BuildRequires:  make
BuildRequires:  (pkgconfig(opencv5) or pkgconfig(opencv4))

%description
dewatermark isolates the translucent watermark that a set of page images has
in common, aligns pages on which it is shifted or scaled, and restores every
page by exactly inverting the watermark's per-pixel blend. Nothing is painted
in. Results are written as LZW-compressed TIFF files.

%prep
%autosetup

%build
%set_build_flags
%make_build build/dewatermark

%install
%make_install PREFIX=%{_prefix}

%files
%license COPYING
%{_bindir}/dewatermark

%changelog
* Tue Oct 27 2026 Daniel Benjamin Miller <dbmiller@dbmiller.org> - 1.0-1
- Initial release.
