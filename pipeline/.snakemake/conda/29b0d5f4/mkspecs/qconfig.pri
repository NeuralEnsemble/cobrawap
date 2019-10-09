#configuration
CONFIG +=  shared qpa no_mocdepend release qt_no_framework
host_build {
    QT_ARCH = x86_64
    QT_TARGET_ARCH = x86_64
} else {
    QT_ARCH = x86_64
    QMAKE_DEFAULT_LIBDIRS = /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/x86_64-conda_cos6-linux-gnu/sysroot/lib /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/x86_64-conda_cos6-linux-gnu/sysroot/usr/lib /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/x86_64-conda_cos6-linux-gnu/lib /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/lib/gcc/x86_64-conda_cos6-linux-gnu/7.3.0 /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/lib/gcc
    QMAKE_DEFAULT_INCDIRS = /home/rgutzen/Projects/wavescalephant/snakemake_workflow/.snakemake/conda/29b0d5f4/include /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/x86_64-conda_cos6-linux-gnu/include/c++/7.3.0 /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/x86_64-conda_cos6-linux-gnu/include/c++/7.3.0/x86_64-conda_cos6-linux-gnu /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/x86_64-conda_cos6-linux-gnu/include/c++/7.3.0/backward /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/lib/gcc/x86_64-conda_cos6-linux-gnu/7.3.0/include /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/lib/gcc/x86_64-conda_cos6-linux-gnu/7.3.0/include-fixed /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/x86_64-conda_cos6-linux-gnu/include /home/conda/feedstock_root/build_artifacts/qt_1548879054661/_build_env/x86_64-conda_cos6-linux-gnu/sysroot/usr/include
}
QT_CONFIG +=  minimal-config small-config medium-config large-config full-config gtk2 gtkstyle fontconfig evdev xlib xrender xcb-plugin xcb-qt xcb-glx xcb-xlib xcb-sm xkbcommon-qt accessibility-atspi-bridge c++11 accessibility opengl shared qpa reduce_exports reduce_relocations clock-gettime clock-monotonic posix_fallocate mremap getaddrinfo ipv6ifname getifaddrs inotify eventfd threadsafe-cloexec system-jpeg system-png png system-freetype harfbuzz system-zlib glib dbus dbus-linked openssl xcb rpath icu concurrent audio-backend release

#versioning
QT_VERSION = 5.6.2
QT_MAJOR_VERSION = 5
QT_MINOR_VERSION = 6
QT_PATCH_VERSION = 2

#namespaces
QT_LIBINFIX = 
QT_NAMESPACE = 

QT_EDITION = OpenSource

QT_COMPILER_STDCXX = 201103
