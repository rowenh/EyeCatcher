# EyeCatcher

<div>
The official code of "Eye Catcher: Automatic Video-Based Sleep Assessment from Eye Cues in Preterm Infants".
</div>



## INSTALLATION:

<div>
NOTE: To find body landmarks in videos, we use a copy of HigherHRNet (attached in ./hrnet/) from https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.
</div>

<br/>

<div>
Install dependencies.
</div>

    > python3 -m venv ./venv

    > .\venv\Scripts\activate (Windows) or > source ./venv/bin/activate (Linux)

    > (venv) python3 -m pip install -r ./requirements.txt

    > (venv) python3 -m pip install ./hrnet/CrowdPose/crowdpose-api/PythonAPI

    > deactivate

<div>
Finally, download pretrained models for HigherHRNet, from "https://drive.google.com/drive/folders/1bdXVmYrSynPLSk5lptvgyQ8fhziobD50".

Move this "models" folder to ./hrnet/

Activate the virtual environment when running our scripts.
</div>



## TRAINING:

<div>
The pipeline introduces 4 models: Visibility, Open, REM and Sleep. Each model has its own directory, including a samples folder.
Samples need to be added to retrain the models. Below, the procedure is explained per model.
</div>

### Visibility model:

<div>
Add (eye) images to the samples folder satisfying following format:
</div>

    .
        /visibility_model
            /samples
                /<"o" if occluded, "v" if visible>_sub<ID#>_<sample ID, for example date and timestamp>.jpg
                /o_sub000_1970-01-01_00;00.jpg  # EXAMPLE

<div>                
Samples for the visibility model can be generated with the annotation tool:
</div>

    > (venv) python3 ./src/annotate_frames.py <path_to_video.mp4> <ABSOLUTE output directory> sub<ID#>_<sample ID, for example date> o+v <body direction: "left", "right", "up" or "down">

### Open model:

<div>
Add eye images to the samples folder satisfying following format:
</div>

    .
        /open_model
            /samples
                /<"o", "c", "or" or "cr">_sub<ID#>_<sample ID, for example date and timestamp>.jpg
                /o_sub000_1970-01-01_00;00.jpg  # EXAMPLE

<div>                
Samples for the open model can be generated with the annotation tool:
</div>

    > (venv) python3 ./src/annotate_frames.py <path_to_video.mp4> <ABSOLUTE output directory> sub<ID#>_<sample ID, for example date> c+o <body direction: "left", "right", "up" or "down">

### REM model:

<div>
Add video fragments of ~1 second (default assumption) to the respective fragments folder satisfying following format:
</div>

    .
        /rem_model
            /fragments
                /<"o", "c", "or" or "cr">
                    /sub<ID#>_<sample ID, for example date and timestamp>
                        /<body direction: "left", "right", "up" or "down">_<eye: "left" or "right">.mp4
                /o  # EXAMPLE
                    /sub000_1970-01-01_00;00
                        /left_left.mp4

<div>
The samples folder can now be generated/updated with following command:
</div>

    > (venv) python3 ./rem_model/update_samples.py

### Sleep model:

<div>
Samples for the sleep model are generated with the following command:
</div>

    > (venv) python3 ./predict_video.py <path_to_video.mp4> <ABSOLUTE output_directory> sub<ID#>_<specify ID, for example date> <body direction: "left", "right", "up" or "down">

<div>
Add generated samples to the samples folder, satisfying following structure:
</div>

    .
        /sleep_model
            /samples
                /sub<ID#>_<specify ID, for example date>_<minute#>
                /sub000_1970-01-01_0  # EXAMPLE

### Train/validate model:

<div>
To train/validate a model on its samples, use following command:
</div>

    > (venv) python3 ./run_model.py <"open", "visibility", "rem" or "sleep"> --train_set=<name constraints> --val_set=<name constraints> --test_set=<name constraints> --folds=<name constraints>

    IMPORTANT:
    -To perform cross validation, use the folds parameter, and leave train/val/test parameters empty.
    -In cross validation, the default validation ratio is used.
    -To build a model on all available data, leave all parameters empty. Uses default validation and test ratios.
    -To perform LOOCV, set folds parameter to "loocv".
    -With the name constraints, you identify groups of samples. Use "+" to separate multiple constraints.
        -For example, to test on subjects 000 and 001, you may use "train_set='sub000+sub001'".
    -For the folds argument, use "=" to separate different folds.
        -For example, to perform cross validation on the folds "sub000+sub001" and "sub002+sub003", you would use "folds='sub000+sub001=sub002+sub003'".



## USAGE:

<div>
To predict sleep states on a video, use following command:
</div>

    > (venv) python3 ./predict_video.py <path_to_video.mp4> <ABSOLUTE output_directory> sub<ID#>_<specify ID, for example date> <body direction: "left", "right", "up" or "down">

<div>
The active model folders (for example "./rem_model/models/active/") are used for predictions, and can be replaced by retrained models.
</div>
