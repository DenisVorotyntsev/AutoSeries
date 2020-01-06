AutoSeries
======================================

## Contents
ingestion/: The code and libraries used on Codalab to run your submmission.

scoring/: The code and libraries used on Codalab to score your submmission.

code_submission/: An example of code submission you can use as template.

data/: Some sample data to test your code before you submit it.

run_local_test.py: A python script to simulate the runtime in codalab

## Local development and testing
1. To make your own submission to AutoSeries challenge, you need to modify the
file `model.py` in `code_submission/`, which implements your algorithm.
2. Test the algorithm on your local computer using Docker,
in the exact same environment as on the CodaLab challenge platform. Advanced
users can also run local test without Docker, if they install all the required
packages.
3. If you are new to docker, install docker from https://docs.docker.com/get-started/.
Then, at the shell, run:
```
cd path/to/autoseries_starting_kit/
docker run -it --rm -v "$(pwd):/app/codalab" vergilgxw/autotable:v3
```
The option `-v "$(pwd):/app/codalab"` mounts current directory
(`autoseries_starting_kit/`) as `/app/codalab`. If you want to mount other
directories on your disk, please replace `$(pwd)` by your own directory.

The Docker image
```
vergilgxw/autotable:v3
```

4. You will then be able to run the `ingestion program` (to produce predictions)
and the `scoring program` (to evaluate your predictions) on toy sample data.
In the AutoSeries challenge, both two programs will run in parallel to give
feedback. So we provide a Python script to simulate this behavior. To test locally, run:
```
python run_local_test.py
```
If the program exits without any errors, you can find the final score from the terminal`s stdout of your solution.
Also you can view the score by opening the `scoring_output/scores.txt`.

The full usage is
```
python run_local_test.py --dataset_dir=./data/demo --code_dir=./code_submission
```
You can change the argument `dataset_dir` to other datasets (e.g. the two
practice datasets we provide). On the other hand,
you can also modify the directory containing your other sample code.

## Download practice datasets
We provide 2 practice datasets for participants. They can use these datasets to:
1. Do local test for their own algorithm;
2. Enable meta-learning.

You may refer to [codalab site](https://autodl.lri.fr/competitions/149#learn_the_details-evaluation) for practice datasets.

Unzip the zip file and you'll get 2 datasets.

## Prepare a ZIP file for submission on CodaLab
Zip the contents of `code_submission`(or any folder containing
your `model.py` file) without the directory structure:
```
cd code_submission/
zip -r mysubmission.zip *
```
then use the "Upload a Submission" button to make a submission to the
competition page on CodaLab platform.

Tip: to look at what's in your submission zip file without unzipping it, you
can do
```
unzip -l mysubmission.zip
```

## Report bugs and create issues

If you run into bugs or issues when using this starting kit, please please contact us via:
<autoseries2020@4paradigm.com>
