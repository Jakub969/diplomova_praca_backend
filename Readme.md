Toto je repozitár pre diplomovu prácu: Mobilná aplikácia pre podporu rozhodovania pri reze ovocných stromov
Aplikácia beží pomocou pointnet++, ktorý používa staršie verzie tensorflow preto je predpripravený dockerfile, ktorý stačí spustit príkazmy:
```
docker build -t pointnet2_inference .
docker run --gpus all -it -v $(pwd):/workspace pointnet2_inference
```
![Obrazok](https://stanford.edu/~rqi/pointnet2/images/pnpp.jpg)