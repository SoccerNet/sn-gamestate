# How can I seperate the Jersey Number Recognition pipeline from the whole pipeline to make the development faster?

To do this is quite simple, you just need to make changes to 3 files overall.

## Step 1
First, you would want to change the pipeline in ```sn_gamesteate/configs/soccernet.yaml``` to only these components:

```
defaults:
    - dataset: soccernet_gs
    ...
...
pipeline:
    - bbox_detector
    - track
    - jersey_number_detect
...
```

While you're at it, you can also go ahead and change the modules for the track from ```bpbreid_strong_sort``` to ```oc_sort```.

```
defaults:
    - dataset: soccernet_gs
    ...
    - modules/reid: prtreid
    - modules/track: oc_sort
    ...
...
```

## Step 2
Then you are going to change the ```sn_gamestate/jersey/mmocr_api.py``` script. 

*Note: You can implement this in easyocr_api.py too*

Here you are going to make same changes in 2 places. You are going to replace "jersey_number_detection" column in the detection dataframe to "jersey_number".

```python
...
class MMOCR(DetectionLevelModule):
    input_columns = ["bbox_ltwh"]
    output_columns = ["jersey_number", "jersey_number_confidence"]
    collate_fn = default_collate

...

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        ...
        detections['jersey_number'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence
        ...
...
```

## Step 3
Finally you will need to make a minor change in the visualization script in ```sn_gamestate/visualization/pitch_visualization.py```. We do not need ```detection.role=="player"```, so we are just gonna comment it out. It is inside the "if condition" just below the line 
```python
#display jersey number
```

```python
...
class PitchVisualizationEngine(Callback):
        ...
            # display jersey number
            if (
                is_prediction
                and self.cfg.prediction.display_jersey_number
                and hasattr(detection, "jersey_number")
                # and detection.role == "player"
            ):
            ...
...
```

That's it then you can run the pipeline again. Note that, this will only give an output video with jersey number. 

**Additional tips: If you are developing, and want to make your testing easier, please try making these changes in ```soccernet.yaml```:**
1. Introduce ```nframes: 50``` (You can change this value to whatever you want to) under 
```
...
dataset:
    nvid: 1
    nframes: 50
    ...
```
2. Second you can run the pipeline once and save the tracker_state object for the first run. You can then load this tracker_state.pklz file under 
```
...
state:
    save_file: null
    load_file: your_first_tracker_object.pklz
...
```