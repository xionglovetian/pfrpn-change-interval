# Run the M1 GPU pipeline smoke test (not a paper experiment).
$env:PATH = 'C:\Users\ASUS\.conda\envs\pf-rpn;' +
  'C:\Users\ASUS\.conda\envs\pf-rpn\Library\mingw-w64\bin;' +
  'C:\Users\ASUS\.conda\envs\pf-rpn\Library\usr\bin;' +
  'C:\Users\ASUS\.conda\envs\pf-rpn\Library\bin;' +
  'C:\Users\ASUS\.conda\envs\pf-rpn\Scripts;' + $env:PATH
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'

C:\Users\ASUS\.conda\envs\pf-rpn\python.exe tools\train.py `
  configs/m2-pf-rpn/m2_pf_rpn_gpu_smoke.py `
  --work-dir work_dirs/m2_pf_rpn/gpu_smoke