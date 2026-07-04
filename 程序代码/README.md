# 通道 4 水泵六分类故障诊断

在本文件所在的 `程序代码/` 目录运行以下命令。

## Run Order

1. `python -m pump_diagnosis.runner --stage inspect`
2. `python -m pump_diagnosis.runner --stage split`
3. `python -m pump_diagnosis.runner --stage features`
4. `python -m pump_diagnosis.runner --stage train`
5. `python -m pump_diagnosis.runner --stage all`

## Feature Groups

- 18 time-domain features
- 20 frequency-domain features
- 9 rotational features
- 17 wavelet-packet features
- 20 envelope features

## 结果目录

- 基础流程结果：`../实验结果/通道4六分类_基础结果/`
- 修正倍频最终结果：`../实验结果/修正倍频_六分类最终结果/`
