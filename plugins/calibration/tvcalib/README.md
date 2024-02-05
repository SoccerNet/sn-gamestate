# TVCalib

## Iterative Optimization
To predict all individual camera parameters from the results of the segment localization, we can run: `python -m tvcalib.optimize`:
Arguments:
- `--hparams`: Path to config filem, e.g., `configs/val_main_center_gt.json` (see details below)
- `--output_dir`: Default `./experiments`; extends with (hparams.stem), e.g., `val_main_center_gt`
- `--log_per_step`: If given, log inforamtion like loss during each optimization step for each sample
- `--device`: `cuda` or `cpu`, default ``cuda`

### Config File (hparams) - Example

```json
{
    "temporal_dim": 1,
    "batch_dim": 256,
    "sigma_scale": 1.96,
    "object3d": "SoccerPitchLineCircleSegments",
    "dataset": {
        "file_match_info": "data/datasets/sncalib-test/match_info.json",
        "extremities_annotations": "data/segment_localization/np4_r4_md30/test",
        "extremities_prefix": "extremities_",
        "num_points_on_line_segments": 4,
        "num_points_on_circle_segments": 8,
        "filter_cam_type": null,
        "remove_invalid": true
    },
    "lens_distortion": false,
    "image_width": 960,
    "image_height": 540,
    "optim_steps": 1000
}
```
- `num_points_on_line_segments` and `num_points_on_circle_segments`: Randomly samples points from provided extremities. If the number of given points is lower, the input is padded with zeros.
- `split_circle_central`: If set to `true`, the central sircle is divided into left and right part using a heuristic. Prevents the optimizer to get stuck in local minima.
- `remove_invalid`: Only relevant if `temporal_dim` > 1 as it removes samples from the dataset which can not fulfill the required number of images per stadium and camera type. Note: temporal_dim` > 1 is not implemented, yet.
- `extremities_annotations` refers to a folder comprising `<extremities_prefix><image_id>.json` files with following information. 

Note, that these annotations can be a post-processed output from a pitch element localization model or ground-truth annotations.
```json
{
     "semantic_class_name_1" : [{'x': x1,'y': y1}, {'x': x2,'y': y2}],
     "semantic_class_name_2": [{'x': x3,'y': y3}, {'x': x4,'y': y4}]
      ...
}
```

### Output

The folder `output_dir_prefix` contains at least the predicted camera parameters and additional information (loss, other meta infomration like stadium) for each sample (`per_sample_output.json`):

```json
{
    "batch_idx":"experiments\/wc14-test_full_gt\/batch_0.pt",
    "image_ids":"1.jpg",
    "time_s":1.8033202589,
    "mask_lines":[[[false,false,false],[true,true,false],[true,true,true],[false,false,false],[false,false,false],[false,false,false],[false,false,false],[false,false,false],[false,false,false],[false,false,false],[false,false,false],[false,false,false],[true,true,true],[false,false,false],[true,true,false],[false,false,false],[false,false,false],[false,false,false],[false,false,false],[false,false,false],[false,false,false],[false,false,false],[false,false,false]]],"mask_circles":[[[false,false,false,false,false,false,false,false],[true,true,true,true,true,true,true,true],[false,false,false,false,false,false,false,false]]],
    "loss_ndc_lines_distances_raw":[[[0.0,0.0,0.0],[0.010469364,0.0019166876,0.0],[0.0086851967,0.0050904173,0.0100935074],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0040892977,0.0009114548,0.0050593293],[0.0,0.0,0.0],[0.0000008759,0.0016332311,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]],
    "loss_ndc_lines":0.0047949357,
    "loss_ndc_circles_distances_raw":[[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0100957388,0.0031360432,0.0003662109,0.0027729045,0.004986987,0.0026938571,0.0051269531,0.002303218],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]],
    "loss_ndc_circles":0.0039352393,
    "loss_ndc_total":0.008730175,
    "loss_ndc_lines_distances_max":0.010469364,
    "loss_ndc_circles_distances_max":0.0100957388,
    "loss_ndc_total_max":0.010469364,
    "league":"Fifa WorldCup",
    "season":"2014",
    "match":"None",
    "date":"None",
    "pan_degrees":-17.578212738,
    "tilt_degrees":80.8040008545,
    "roll_degrees":-0.1310900003,
    "position_meters":[-7.7424321175,57.8480377197,-11.1651697159],
    "aov_radian":0.4283054769,
    "aov_degrees":24.540096283,
    "x_focal_length":2942.6950683594,
    "y_focal_length":2942.6950683594,
    "principal_point":[640.0,360.0],
    "radial_distortion":[0.0,0.0,0.0,0.0,0.0,0.0],
    "tangential_distortion":[0.0,0.0],
    "thin_prism_distortion":[0.0,0.0,0.0,0.0],
    "stadium":"None",
}
```