
![](./doc/EVS_LOGO_COLOUR_RGB.png)

# EVS Camera Calibration Challenge


Welcome to the EVS camera calibration challenge ! This challenge is sponsored by EVS Broadcast Equipment, and is
developped in collaboration with the SoccerNet team.

This challenge consists of two distinct tasks which are defined hereafter. We provide sample code and baselines for each
task to help you get started!


Participate in our upcoming Challenge at the [CVSports](https://vap.aau.dk/cvsports/) workshop at CVPR and try to win up to 1000$ sponsored by [EVS](https://evs.com/)! All details are available on the [challenge website](https://eval.ai/web/challenges/challenge-page/1537/overview ), or on the [main page](https://www.soccer-net.org/).

The participation deadline is fixed at the 30th of May 2023. The official rules and guidelines are available on [ChallengeRules.md](ChallengeRules.md).

<a href="https://www.youtube.com/watch?v=nxywN6X_0oE">
<p align="center"><img src="images/Thumbnail.png" width="720"></p>
</a>

### 2023 Leaderboard

<p><table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th style = "background-color: #FFFFFF;font-family: Century Gothic, sans-serif;font-size: medium;color: #305496;text-align: left;border-bottom: 2px solid #305496;padding: 0px 20px 0px 0px;width: auto">Team</th>
      <th style = "background-color: #FFFFFF;font-family: Century Gothic, sans-serif;font-size: medium;color: #305496;text-align: left;border-bottom: 2px solid #305496;padding: 0px 20px 0px 0px;width: auto">Combined Metric</th>
      <th style = "background-color: #FFFFFF;font-family: Century Gothic, sans-serif;font-size: medium;color: #305496;text-align: left;border-bottom: 2px solid #305496;padding: 0px 20px 0px 0px;width: auto">Accuracy@5</th>
      <th style = "background-color: #FFFFFF;font-family: Century Gothic, sans-serif;font-size: medium;color: #305496;text-align: left;border-bottom: 2px solid #305496;padding: 0px 20px 0px 0px;width: auto">Completeness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">Sportlight</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">0.55</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">73.22</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">75.59</td>
    </tr>
    <tr>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">Spiideo</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">0.53</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">52.95</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">99.96</td>
    </tr>
    <tr>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">SAIVA_Calibration</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">0.53</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">60.33</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">87.22</td>
    </tr>
    <tr>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">BPP</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">0.5</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">69.12</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">72.54</td>
    </tr>
    <tr>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">ikapetan</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">0.43</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">53.78</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">79.71</td>
    </tr>
    <tr>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">NASK</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">0.41</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">53.01</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">77.81</td>
    </tr>
    <tr>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">Mike Azatov and Jonas Theiner</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">0.41</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">58.61</td>
      <td style = "background-color: #D9E1F2;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">69.34</td>
    </tr>
    <tr>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">Baseline</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">0.08</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">13.54</td>
      <td style = "background-color: white; color: black;font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 0px 20px 0px 0px;width: auto">61.54</td>
    </tr>
  </tbody>
</table></p>

## Table of content

- Install
- Dataset
    - Soccer pitch annotations
- Camera calibration
    - Definition
    - Evaluation
    - Baseline

## Install

Download the network weights from google drive https://drive.google.com/file/d/1dbN7LdMV03BR1Eda8n7iKNIyYp9r07sM/view?usp=sharing 
and place them in [resources](resources).

With python 3 already installed, the python environment can be installed with the following command:

```
pip install -r requirements.txt
```

## Dataset

SoccerNet is a dataset containing 400 broadcasted videos of whole soccer games. The dataset can be found
here :   https://soccer-net.org/
The authors give access to template code and a python package to get started with the dataset. All the documentation
about these tools can be found here : https://github.com/SilvioGiancola/SoccerNetv2-DevKit

All the data needed for challenge can be downloaded with these lines : 

```python
from SoccerNet.Downloader import SoccerNetDownloader as SNdl
soccerNetDownloader = SNdl(LocalDirectory="path/to/SoccerNet")
soccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train","valid","test","challenge"])
```

Historically, the dataset was first released for an action spotting task. In its first version, the images corresponding
to soccer actions (goals, fouls, etc) were identified. In the following editions, more annotations have been associated
to those images. In the last version of the dataset (SoccerNetV3), the extremities of the lines of the soccer pitch
markings have been annotated. As a partnership with SoccerNet's team, we use these annotations in a new challenge. The
challenge is divided in two tasks, the resolution of the first leading to the second one. The first is a soccer pitch
element localisation task which can then be used for the second task which is a camera calibration task.

**/!\ New** : some annotations have been added: for some images, there are new points annotation along the pitch markings lines.
For straight pitch marking lines, you can always assume that the extremities are annotated, and sometimes, 
if the image has been reannotated, there will be a few extra points along the imaged line. 


### Soccer pitch annotations

Performing camera calibration can be eased by the presence of an object with a known shape in the image. For soccer
content, the soccer pitch can be used as a target for the camera calibration because it has a known shape and its
dimensions are specified in the International Football Association Board's law of the
game (https://digitalhub.fifa.com/m/5371a6dcc42fbb44/original/d6g1medsi8jrrd3e4imp-pdf.pdf).

Moreover, we define a set of semantic labels for each semantic element of the soccer pitch. We also define the bottom
side of the pitch as the one where the main and 16 meters broadcast cameras are installed.

![](./doc/soccernet_classes.png)

1. Big rect. left bottom,
2. Big rect. left main,
3. Big rect. left top,
4. Big rect. right bottom,
5. Big rect. right main,
6. Big rect. right top,
7. Circle central,
8. Circle left,
9. Circle right,
10. Goal left crossbar,
11. Goal left post left ,
12. Goal left post right,
13. Goal right crossbar,
14. Goal right post left,
15. Goal right post right,
16. Middle line,
17. Side line bottom,
18. Side line left,
19. Side line right,
20. Side line top,
21. Small rect. left bottom,
22. Small rect. left main,
23. Small rect. left top,
24. Small rect. right bottom,
25. Small rect. right main,
26. Small rect. right top

In the third version of SoccerNet, there are new annotations for each image of the dataset. These new annotations
consists in the list of all extremities of the semantic elements of the pitch present in the image. The extremities are
a pair of 2D point coordinates.

For the circles drawn on the pitch, the annotations consist in a list of points which give roughly the shape of the
circle when connected. Note that due to new annotations, the sequential order of circle points is no longer guaranteed. 



##  Camera calibration task

As a soccer pitch has known dimensions, it is possible to use the soccer pitch as a calibration target in order to
calibrate the camera. Since the pitch is roughly planar, we can model the transformation applied by the camera to the pitch by a
homography. In order to estimate the homography between the pitch and its image, we only need 4 lines. We provide a
baseline in order to extract camera parameters from this kind of images.

### Definition

In this task, we ask you to provide valid camera parameters for each image of the challenge set.

Given a common 3D pitch template, we will use the camera parameters produced by your algorithm in order to estimate the
reprojection error induced by the camera parameters. The camera parameters include its lens parameters, its orientation,
its translation with respect to the world reference axis system that we define accordingly:

![](./doc/axis-system-resized.jpeg)

#### Rotation convention

Following Euler angles convention, we use a ZXZ succession of intrinsic rotations in order to describe the orientation
of the camera. Starting from the world reference axis system, we first apply a rotation around the Z axis to pan the
camera. Then the obtained axis system is rotated around its x axis in order to tilt the camera. Then the last rotation
around the z axis of the new axis system allows to roll the camera. Note that this z axis is the principal axis of the
camera.

The lens parameters produced must follow the pinhole model. Additionally, the parameters can include radial, tangential
and thin prism distortion parameters. This corresponds to the full model of
OpenCV : https://docs.opencv.org/4.5.0/d9/d0c/group__calib3d.html#details

For each image of the test set, we expect to receive a json file named "**camera_{frame_index}.json**" containing a
dictionary with the camera parameters. 

```
# camera_00001.json
 {
      "pan_degrees": 14.862476218376278,
      "tilt_degrees": 78.83988009048775,
      "roll_degrees": -2.2210919345134497,
      "position_meters": [
          32.6100008989567,
          67.9363036953344,
          -14.898134157887508
      ],
      "x_focal_length": 3921.6013418112757,
      "y_focal_length": 3921.601341812138,
      "principal_point": [
          480.0,
          270.0
      ],
      "radial_distortion": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
      ],
      "tangential_distortion": [
          0.0,
          0.0
      ],
      "thin_prism_distortion": [
          0.0,
          0.0,
          0.0,
          0.0
      ]
 }
```

The results should be organized as follows :


```
test.zip
|__ camera_00001.json
|__ camera_00002.json
```


### Evaluation

We only evaluate the quality of the camera parameters provided in the image world, as it is the only groundtruth that we
have. The annotated points are either extremities or points on circles, and thus they do not always have a mapping to a
3D point, but we can always map them to lines or circles.

#### Dealing with ambiguities

The dataset contains some ambiguities when we consider each image independently. Without context, one can't know if a
camera behind a goal is on the left or the right side of the pitch.

For example, this image taken by a camera behind the goal :

![](./doc/tactical_amibguity_resized.jpeg)

It is impossible to say without any context if it was shot by the camera whose filming range is in blue or in yellow.

![](./doc/ambiguity_soccernet_resized.jpeg)

Therefore, we take this ambiguity into account in the evaluation and we consider both accuracy for the groundtruth
label, and the accuracy for the central projection by the pitch center of the labels. The higher accuracy will be
selected for your evaluation.


We evaluate the best submission based on the accuracy at a specific threshold distance. This metric is explained
hereunder.

#### Accuracy @ threshold

The evaluation is based on the reprojection error which we define here as the L2 distance between one annotated point
and the line to which the point belong. This metric does not account well for false positives and false negatives
(hallucinated/missing lines projections). Thus we formulate our evaluation as a binary classification, with a
distance threshold with a twist : this time, we consider a pitch marking to be one entity, and for it to be correctly
detected, all its extremities (or all points annotated for circles) must have a reprojection error smaller than the
threshold.

As we allow lens distortion, the projection of the pitch line markings can be curvated. This is why we sample the pitch
model every few centimeters, and we consider that the distance between a projected pitch marking and a groundtruth point is in
fact the euclidian distance between the groundtruth point and the polyline given by the projection of sampled points.

* True positives : for classes that belong both to the prediction and the groundtruth, a predicted element is a True
  Positive if all the L2 distances between its groundtruth points and the predicted polyline are lower than a
  certain threshold.

  ![](./doc/tp-condition-2.png)

* False positives : contains elements that were detected with a class that do not belong to the groundtruth classes, and
  elements with valid classes which are distant from at least **t** pixels from one of the groundtruth points associated
  to the element.
* False negatives: Line elements only present in the groundtruth are counted as False Negatives.
* True negatives : There are no True Negatives.

The Accuracy for a threshold of t pixels is given by : **Acc@t = TP/(TP+FN+FP)**. We evaluate the accuracy at 5 pixels. 
We only use images with predicted camera parameters in this evaluation.

#### Completeness rate

We also measure the completeness rate as the number of camera parameters provided divided by the number of images with
more than four semantic line annotations in the dataset.

#### Final score 

The evaluation criterion for a camera calibration method is the following : **Completeness x Acc@5**

#### Per class information

The global accuracy described above has the advantage to treat all soccer pitch marking types equivalently even if the
groundtruth contains more annotated points for a specific class. For instance, a circle is annotated with 9 points on
average, whilst rectilinear elements have usually two points annotated. But this metric might be harder as we consider
all points annotated instead of each of them independently. This is precisely why we propose this per class metric, that
accounts for each point annotated separately. This metric is only for information purposes and will not be used in the
ranking of submissions.

The prediction is obtained by sampling each 3D real world pitch element every few centimeters, which means that the
number of points in the prediction may be variable for the same camera parameters. This gives a very high number of 
predicted points for a certain class, and thus we find a workaround to count false positives.

The confusion matrices are computed per class in the following way :

* True positives : for classes that belong both to the prediction and the groundtruth, a predicted point is counted in
  the True Positives if the L2 distance from this groundtruth point to the predicted polyline is lower than a
  certain threshold **t**.

* False positives : counts groundtruth points that have a distance to the corresponding predicted polyline that is higher
  than the threshold value **t**. For predicted lines that do not belong to the groundtruth, we can not count every
  predicted point as a false positive because the number of points depend on the sampling factor that has to be high
  enough, which can lower a lot our metric. We decided arbitrarily to count for this kind of false positive the number
  of points that a human annotator would have annotated for this class, i.e. two points for rectilinear elements and 9
  for circle elements. 
* False negatives: All groundtruth points whose class is only present in the groundtruth are counted as False Negatives.
* True negatives : There are no True Negatives.

The Accuracy for a threshold of t pixels is given by : **Acc@t = TP/(TP+FN+FP)**. We evaluate the accuracy at 5, 10 and
20 pixels. We only use images with predicted camera parameters in this evaluation.


### Baseline

#### Method

For our camera calibration baseline, we proceed in two steps: first we find the pitch markings location in the image, 
and then given our pitch marking correspondences, we estimate camera parameters.

For this first step, we decided to locate the pitch markings with a neural network trained to perform semantic line 
segmentation. We used DeepLabv3 architecture. The target semantic segmentation masks were generated by joining 
successively all the points annotated for each line in the image. We provide the dataloader that we used for the 
training in the src folder.

The segmentation maps predicted by the neural
network are further processed in order to get the line extremities. First, for each class, we fit circles on the
segmentation mask. All pixels belonging to a same class are thus synthesized by a set of points (i.e. circles centers).
Then we build polylines based on the circles centers : all the points that are close enough are considered to belong to
the same polyline. Finally the extremities of each line class will be the extremities of the longest polyline for that
line class.

You can test the line detection with the following code:

```
python src/detect_extremities.py -s <path_to_soccernet_dataset> -p <path_to_store_predictions>
```


In the second step, we use the extremities of the lines (
not Circles) detected in the image in order to estimate an homography from the soccer pitch model to the image. We
provide a class **soccerpitch.py** to define the pitch model according to the rules of the game. The homography is then
decomposed in camera parameters. All aspects concerning camera parameters are located in the **camera.py**, including
homography decomposition in rotation and translation matrices, including calibration matrix estimation from the
homography, functions for projection...

You can test the baseline with the following line : 

```python src/baseline_cameras.py -s <path to soccernet dataset> -p <path to 1st task prediction>``` 

And to test the evaluation, you can run :

`python src/evaluate_camera.py -s  <path to soccernet dataset> -p <path to predictions> -t <threshold value>`


#### Results

| Acc@t    | Acc@5 | Completeness | Final score | 
|----------|-------|--------------|-------------|
| Baseline | 11.7% | 68%          | 7.96%       |

#### Improvements

The baseline could be directly improved by :

* exploiting the masks rather than the extremities prediction of the first baseline
* using ransac to estimate the homography
* refining the camera parameters using line and ellipses correspondences.

## Citation
For further information check out the paper and supplementary material:
https://arxiv.org/abs/2210.02365

Please cite our work if you use the SoccerNet dataset:
```bibtex
@inproceedings{Giancola_2022,
	doi = {10.1145/3552437.3558545},
	url = {https://doi.org/10.1145%2F3552437.3558545},
	year = 2022,
	month = {oct},
	publisher = {{ACM}},
	author = {Silvio Giancola and Anthony Cioppa and Adrien Deli{\`{e}}ge and Floriane Magera and Vladimir Somers and Le Kang and Xin Zhou and Olivier Barnich and Christophe De Vleeschouwer and Alexandre Alahi and Bernard Ghanem and Marc Van Droogenbroeck and Abdulrahman Darwish and Adrien Maglo and Albert Clap{\'{e}}s and Andreas Luyts and Andrei Boiarov and Artur Xarles and Astrid Orcesi and Avijit Shah and Baoyu Fan and Bharath Comandur and Chen Chen and Chen Zhang and Chen Zhao and Chengzhi Lin and Cheuk-Yiu Chan and Chun Chuen Hui and Dengjie Li and Fan Yang and Fan Liang and Fang Da and Feng Yan and Fufu Yu and Guanshuo Wang and H. Anthony Chan and He Zhu and Hongwei Kan and Jiaming Chu and Jianming Hu and Jianyang Gu and Jin Chen and Jo{\~{a}}o V. B. Soares and Jonas Theiner and Jorge De Corte and Jos{\'{e}} Henrique Brito and Jun Zhang and Junjie Li and Junwei Liang and Leqi Shen and Lin Ma and Lingchi Chen and Miguel Santos Marques and Mike Azatov and Nikita Kasatkin and Ning Wang and Qiong Jia and Quoc Cuong Pham and Ralph Ewerth and Ran Song and Rengang Li and Rikke Gade and Ruben Debien and Runze Zhang and Sangrok Lee and Sergio Escalera and Shan Jiang and Shigeyuki Odashima and Shimin Chen and Shoichi Masui and Shouhong Ding and Sin-wai Chan and Siyu Chen and Tallal El-Shabrawy and Tao He and Thomas B. Moeslund and Wan-Chi Siu and Wei Zhang and Wei Li and Xiangwei Wang and Xiao Tan and Xiaochuan Li and Xiaolin Wei and Xiaoqing Ye and Xing Liu and Xinying Wang and Yandong Guo and Yaqian Zhao and Yi Yu and Yingying Li and Yue He and Yujie Zhong and Zhenhua Guo and Zhiheng Li},
	title = {{SoccerNet} 2022 Challenges Results},
	booktitle = {Proceedings of the 5th International {ACM} Workshop on Multimedia Content Analysis in Sports}
}
```



## Our other Challenges

Check out our other challenges related to SoccerNet!
- [Action Spotting](https://github.com/SoccerNet/sn-spotting)
- [Replay Grounding](https://github.com/SoccerNet/sn-grounding)
- [Calibration](https://github.com/SoccerNet/sn-calibration)
- [Re-Identification](https://github.com/SoccerNet/sn-reid)
- [Tracking](https://github.com/SoccerNet/sn-tracking)
