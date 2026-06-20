🅱 commands (download tested in a scratch CWD — downloads land relative to where you run it):
``` bash
python -u src/utils/hf_outputs.py upload-full outputs/ 2>&1 | tee logs/hf_outputs_upload_full_outputs_sanity_poc_$(date +%Y%m%d_%H%M%S).log   # answer: 1 (delete)
python -u src/utils/hf_outputs.py verify-full outputs/ 2>&1 | tee logs/hf_outputs_verify_full_outputs_sanity_poc_$(date +%Y%m%d_%H%M%S).log  # MUST say PASS

mkdir -p /workspace/dl_test && \
cd /workspace/dl_test && \
python -u /workspace/factorjepa/src/utils/hf_outputs.py download-full  outputs/poc 2>&1 | tee /workspace/factorjepa/logs/hf_outputs_download_full_test_$(date +%Y%m%d_%H%M%S).log
diff -rq /workspace/dl_test/outputs/ /workspace/factorjepa/outputs/  
# only _full-manifest.json may differ; 
# after
rm -rf /workspace/dl_test 
```