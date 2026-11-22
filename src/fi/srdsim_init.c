/* SPDX-License-Identifier: Apache-2.0 */
#include "srdsim.h"

static int srdsim_getinfo(uint32_t version, const char *node, const char *service, uint64_t flags,
			  const struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *cur, *prev = NULL, *next;
	int ret;

	/* EFA address strings are not parsed; the consumers this provider serves pass neither. */
	if (node || service || (flags & FI_SOURCE)) {
		SRDSIM_INFO(FI_LOG_CORE, "a node, a service or FI_SOURCE was given; not supported\n");
		return -FI_ENODATA;
	}

	ret = util_getinfo(&srdsim_util_prov, version, NULL, NULL, 0, hints, info);
	if (ret)
		return ret;

	for (cur = *info; cur; cur = next) {
		next = cur->next;
		if (srdsim_getinfo_alter(version, hints, cur) == 0) {
			prev = cur;
			continue;
		}
		if (prev)
			prev->next = next;
		else
			*info = next;
		cur->next = NULL;
		fi_freeinfo(cur);
	}
	return *info ? 0 : -FI_ENODATA;
}

static void srdsim_fini(void)
{
	srdsim_free_infos();
	ofi_monitors_cleanup();
	ofi_hmem_cleanup();
	ofi_mem_fini();
}

struct fi_provider srdsim_prov = {
	.name = SRDSIM_PROV_NAME,
	.version = OFI_VERSION_DEF_PROV,
	.fi_version = OFI_VERSION_LATEST,
	.getinfo = srdsim_getinfo,
	.fabric = srdsim_fabric,
	.cleanup = srdsim_fini,
};

/* The provider is only built as a DSO; this is the fi_prov_ini libfabric looks up after loading it. */
FI_EXT_INI
{
	ofi_mem_init();
	ofi_hmem_init();
	ofi_monitors_init();
	if (srdsim_build_infos())
		return NULL;
	return &srdsim_prov;
}
