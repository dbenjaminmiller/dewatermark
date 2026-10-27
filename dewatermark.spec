Name:           dewatermark
Version:        1.0
Release:        1%{?dist}
Summary:        Remove a translucent watermark shared by a set of page images
License:        0BSD
URL:            https://github.com/dbenjaminmiller/dewatermark
Source0:        %{url}/archive/v%{version}/%{name}-%{version}.tar.gz

BuildRequires:  gcc-c++
BuildRequires:  make
BuildRequires:  (pkgconfig(opencv5) or pkgconfig(opencv4))

%description
dewatermark: remove a translucent watermark from a set of images.

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
