## Solar Irradiance - IFT6759

### Team 08

### To run the evaluation script:

```console
1. cd scripts/
2. Update submit_evalution.sh 
3. sbatch submit_evalution.sh
```
OR
```console
1. cd scripts/
2. Update run_evaluatior.sh 
3. Run run_evaluatior.sh
```

### K-Fold Strategy

* Hold out 1 year of data (2015)
* No use of k-fold until pipeline is optimized

### To setup a new local environment:

```console
module load python/3.7
virtualenv ../local_env
source ../local_env/bin/activate
pip install -r requirements_local.txt
```

### To setup a new server node environment:

```console
module load python/3.7
virtualenv ../server_env --no-download
source ../server_env/bin/activate
pip install --no-index -r requirements.txt
```
OR, if no requirement.txt file is available:
```console
pip install --no-index tensorflow-gpu==2 pandas numpy tqdm
```

### To evaluate results from server locally using tensorboard:

Run the commands to synchronize data from the server and to launch tensorboard:
```console
./rsync_data.sh
./run_tensorboard.sh
```
Use a web browser to visit: http://localhost:6006/
 
 ## CNN Architecture 
 * CNN followed by dense layers along with the metadata information
![CNN](https://github.com/harmanpreet93/Solar-irradiance-prediction/blob/master/Notebooks/CNN.png)  
![NN](https://github.com/harmanpreet93/Solar-irradiance-prediction/blob/master/Notebooks/NN.png)
 
 
### Coding Standards

* Lint your code as per PEP8 before submitting a pull request
* Pull requests are required for merging to master for major changes
* Use git branch
* No large files allowed in git  

