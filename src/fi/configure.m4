dnl SPDX-License-Identifier: Apache-2.0
dnl The libfabric provider of nccl-srdsim, built inside a libfabric source tree by scripts/build_fi_provider.sh.
dnl It is only ever built as a DSO (--enable-srdsim=dl), so that a process loads it only when it names it.
dnl
dnl $1: action if configured successfully
dnl $2: action if not configured successfully
AC_DEFUN([FI_SRDSIM_CONFIGURE],[
	srdsim_happy=0
	dnl Built only when asked for as a DSO; a plain configure of the tree leaves it out.
	AS_IF([test x"$srdsim_dl" = x"1"],
	      [AS_CASE([$host_os], [*linux*], [srdsim_happy=1],
	               [AC_MSG_ERROR([the srdsim provider builds on Linux only])])],
	      [test x"$enable_srdsim" = x"yes"],
	      [AC_MSG_ERROR([the srdsim provider is built only as a DSO: use --enable-srdsim=dl])])
	AS_IF([test $srdsim_happy -eq 1], [$1], [$2])
])
