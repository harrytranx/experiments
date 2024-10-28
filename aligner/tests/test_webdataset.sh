ALIGNER_PARENT_DIR=/fsx_0/user/tranx/rsync

ALIGNER_DEP_DIR=/fsx_0/user/tranx/llm_mm_aligner/replicated
PYTHONPATH=${PYTHONPATH}:${ALIGNER_PARENT_DIR}:${ALIGNER_DEP_DIR} python3 test_webdataset.py