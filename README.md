## Solar Irradiance - IFT6759

### Team 08

### K-Fold Strategy

* Hold out 1 year of data
* No use of k-fold until pipline is optimized

### Coding Standards

* Lint your code as per PEP8 before submiting a pull request
* Pull requests are required for merging to master for major changes
* Use your own branch for major work, don't use master
* No large files allowed in git
* Mark task in progress on Kanban before starting work

### To run the evalutation script:

```console
source ../default_env/bin/activate
./run_evaluator.sh
```

### To setup a new environment:

```console
module load python/3.7
virtualenv ../my_env --no-download
source ../my_env/bin/activate
pip install -r requirements.txt
```
