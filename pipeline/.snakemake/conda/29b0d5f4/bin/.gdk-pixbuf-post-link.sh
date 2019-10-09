#! /bin/bash

# Make sure that gdk-pixbuf's loaders.cache is fully up-to-date. We don't want
# to assume that pkg-config is available, so we hack a bit to avoid running it
# when we get the cache location.
#
# Packages that install gdk-pixbuf loaders (such as librsvg) should have
# post-link and post-unlink scripts that just execute this one, which will be
# available as `$PREFIX/bin/.gdk-pixbuf-post-link.sh`.

set -e
eval $(grep -v : $PREFIX/lib/pkgconfig/gdk-pixbuf-2.0.pc)
if [ -z "$gdk_pixbuf_cache_file" ] ; then
    bindir="$(echo $PREFIX/lib/gdk-pixbuf-2.0/*)"
    if [ ! -d $bindir ] ; then
       echo >&2 "error: no such gdk-pixbuf binary directory $PREFIX/lib/gdk-pixbuf-2.0/*"
       exit 1
    fi
    gdk_pixbuf_cache_file=$bindir/loaders.cache
fi
$PREFIX/bin/gdk-pixbuf-query-loaders >$gdk_pixbuf_cache_file
