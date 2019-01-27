# depth_estimation_MCCNN
Depth estimation on mtk data with a CNN model trained for cost computation (MCCNN)

## Workflow
* cost computation using MCCNN
* cost aggregation by opencv blur filter
* cost optimization simply adopting winner take all
* refinement with opencv weighted median filter 

## Model
![alt text](https://github.com/leduoyang/Screen2Camera-comm/blob/master/Result/model.png)

## Result
![alt text](https://github.com/leduoyang/Screen2Camera-comm/blob/master/Result/result1.png)
![alt text](https://github.com/leduoyang/Screen2Camera-comm/blob/master/Result/result2.png)

source:
https://arxiv.org/pdf/1409.4326.pdf
https://github.com/jzbontar/mc-cnn
