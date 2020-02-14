## Solar Irradiance - IFT6759

### Team 08

### K-Fold Strategy

* Hold out 1 year of data
* No use of k-fold until pipeline is optimized

### Coding Standards

* Lint your code as per PEP8 before submitting a pull request
* Pull requests are required for merging to master for major changes
* Use your own branch for major work, don't use master
* No large files allowed in git
* Mark task in progress on Kanban before starting work

### To run the evaluation script:

```console
source ../default_env/bin/activate
./run_evaluator.sh
```

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

