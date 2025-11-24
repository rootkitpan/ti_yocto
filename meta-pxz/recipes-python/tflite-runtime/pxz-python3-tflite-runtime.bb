SUMMARY = "TensorFlow Lite Runtime Python package"
DESCRIPTION = ""
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/Apache-2.0;md5=89aea4e17d99a7cacdbeed46a0096b10"

SRC_URI = "file://tflite_runtime-2.13.0-cp310-cp310-manylinux2014_armv7l.whl"

S = "${WORKDIR}"

inherit python3native

do_configure[noexec] = "1"
do_compile[noexec] = "1"


do_install() {
	install -d ${D}${PYTHON_SITEPACKAGES_DIR}
	${PYTHON} -m zipfile \
		-e ${WORKDIR}/tflite_runtime-2.13.0-cp310-cp310-manylinux2014_armv7l.whl \
		${D}${PYTHON_SITEPACKAGES_DIR}
}


# Tell Yocto which files belong to this package
FILES:${PN} += "${PYTHON_SITEPACKAGES_DIR}/tflite_runtime* \
				${PYTHON_SITEPACKAGES_DIR}/tflite_runtime-*.dist-info"

RDEPENDS:${PN} += "python3-core python3-numpy"





