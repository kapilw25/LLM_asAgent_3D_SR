1) numbering "m"odules for sorting: [src/m01_<name>.py, src/m02_<name>.py, ] (start with letter "m" to not get into import error with number as prefix)
2) whem moving form current/previous phase to next phase, move the current python files to @src/legacy/ directory and rename the existing files as per nature of next phase
3) crete common / UTILS files here @src/utils
4) note: currently phase0 or API based / noGPU needed tasks are being executed on local M1 macbook >> but difficult LLM/ VLM (intance based , not API)  finetuning  or inference will be executed on Nvidia's GPU 
5) in each python script, keep Docstring limited to python commands to be executed starting max 2 lines of explanation about the code
6) TESTING `source venv_3Denv/bin/activate && python -m py_compile src/m0*.py && echo "All syntax checks passed"`