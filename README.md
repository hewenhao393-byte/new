# Channel 4 Six-Class Pump Diagnosis

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

## Output Root

All generated artifacts are written under `outputs/channel4_six_class_v1/`.
