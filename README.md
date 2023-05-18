# Face recognition system
Calculated evaluation metrics of the original (threshold = 1) and fine-tuned (threshold = 0.75) models
| | Original model | Fine-tuned model
| --- | --- | --- 
| Loss | 0.02	| 0.01
| Averaged positive distance | 0.73 |	0.41
| Averaged negative distance | 1.26	| 1.38
| VAL	| 0.89 | 0.94
| FAR | 0.05 | 0.04

Distances under normal and poor lighting of [pretrained](https://github.com/timesler/facenet-pytorch) and fine-tuned models:

![image](https://github.com/shpirm/facerecognitionsystem/assets/99517424/a3c7cb5c-7e95-4102-9d51-8de1d36bc5f5)


Implemented face recognition system:

![image](https://github.com/shpirm/facerecognitionsystem/assets/99517424/3545740b-2b57-4c6e-ad20-52bfd823d731)
