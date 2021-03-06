# Model Zoo and Baselines

This page includes the experimental results and pre-trained models available for download.

## SWiG-HOI
|             | mAP-full | mAP-rare | mAP-nonrare | mAP-novel | download |
|:-----------|:--------:|:--------:|:-----------:|:---------:|:--------:|
| HOIR (Res101)|     |      |        |      |         |
| HOIR (Swin) |   15.41  |   14.20   |    18.04    |    4.80  | [configs](https://github.com/scwangdyd/large_vocabulary_hoi_detection/blob/master/configs/swig_hoi/hoir_swin.yaml) \| [model](https://drive.google.com/file/d/1-MG9Ef7uXgmVWwM_OppXap1ecvWgowcy/view?usp=sharing)        |
| CHOIR (Res101-FPN) |     |     |        |     | [configs]() \| [model]()        |
| CHOIR (Swin-T) |     |     |        |     | [configs]() \| [model]()        |


## HICO-DET
Our model mainly aims to detect the human interactions with objects. In the HICO-DET dataset, we ignore all `non-interaction` classes as the model is trained to detect the interacting objects. The following results are calculated over 520 HOIs (excluding 80 non-interactions). During the evaluation, images without any valid interaction annotations are also removed the test set.

|             | mAP-full | mAP-rare | mAP-nonrare | download |
|:-----------|:--------:|:--------:|:-----------:|:---------:|
| HOIR (Res101) |     |      |       | [configs]() \| [model]()        |
| HOIR (Swin) |     |      |       | [configs]() \| [model]()        |
| CHOIR (Res101-FPN) |     |      |       | [configs]() \| [model]()        |
| CHOIR (Swin-T) |  30.75   |  25.55    |   32.52    | [configs](https://github.com/scwangdyd/large_vocabulary_hoi_detection/blob/master/configs/hico_det/choir_swin.yaml) \| [model](https://drive.google.com/file/d/1kWv18fJLkqDkJCnEGz1YjwdMB2FEYzFU/view?usp=sharing)        |
## DOH-HOI