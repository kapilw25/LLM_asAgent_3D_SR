🅱 commands (download tested in a scratch CWD — downloads land relative to where you run it):
``` bash
python -u src/utils/hf_outputs.py upload-full outputs/poc 2>&1 | tee logs/upload_full_outputs_poc_$(date +%Y%m%d_%H%M%S).log
python -u src/utils/hf_outputs.py verify-full outputs/poc 2>&1 | tee logs/verify_full_outputs_poc_$(date +%Y%m%d_%H%M%S).log   
# expect VERIFY-FULL: PASS


mkdir -p /workspace/dl_test && \
cd /workspace/dl_test && \
python -u /workspace/factorjepa/src/utils/hf_outputs.py download-full  outputs/poc 2>&1 | tee /workspace/factorjepa/logs/download_full_test_$(date +%Y%m%d_%H%M%S).log
diff -rq /workspace/dl_test/outputs/poc /workspace/factorjepa/outputs/poc   
# only _full-manifest.json may differ; 
# after
rm -rf /workspace/dl_test 
```