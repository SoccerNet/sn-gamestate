# Guidelines for the Game State Reconstruction Challenge

Subscribe (watch) the repo to receive the latest info regarding timeline and prizes!

We propose the SoccerNet challenges to encourage the development of state-of-the-art algorithm for Soccer Video Understanding.

 - **Game State Reconstruction**: Maintain a complete understanding of the game by predicting : 
   - top-view player localization (optionally with camera calibration)
   - role classification (player, goalkeeper, referee, other)
   - team affiliation and team side (left, right)
   - jersey number recognition

We provide an [evaluation server](https://eval.ai/web/challenges/challenge-page/2251/overview) for the Game State Reconstruction task. 
The evaluation server handles predictions for the open **test** sets and the segregated **challenge** sets of each challenge.

Winners will be announced at CVSports Workshop at CVPR 2024.
This challenge will be sponsored by Sportradar, with a prize money of 1000$!

## Evaluation metric
   The evaluation metric for the Game State Reconstruction task is the **GS-HOTA** metric, a modified version of the standard HOTA metric.
   A detailed description of this metric can be found in the main [README](sn-gamestate/README.md) and in the official paper.

## Submission format
   When running the baseline with the evaluation enabled, a submission .zip file will be automatically generated in the output folder, e.g. 'output_folder/2024-03-14/09-51-08/eval/pred/SoccerNetGS-challenge.zip'. 
   The submission file is a zipped folder containing a json file for each video in the evaluated (test/challenge) set.
   Each json file should be named after the corresponding video, e.g. 'SNGS-175.json', 'SNGS-013.json', etc.
   The prediction json format is very similar to the ground truth json format, with the following differences:
   - The json contains a dictionary with a single 'predictions' key, which contains a list of dictionaries, one for each predicted detection.
   - Each detection contains the following fields: "category_id", "image_id", "track_id", "supercategory", "confidence", "attributes" (with "role", "jersey", "team"), and "bbox_pitch" (with "x_bottom_middle" and "y_bottom_middle", i.e. the 2 position in meters on the pich).
   The "supercategory" of each detection must be set to "object" and "category_id" to 1.0.
   Other categories are reserved for predictions related to camera calibration and pitch localization, but ignored in the evaluation procedure.
   We provide an example submission file for the test set [here](examples_predictions/SoccerNetGS-test.zip).

## Who can participate / How to participate?

 - Any individual can participate in the challenge, except the organizers.
 - The participants are recommended to form a team to participate.
 - Each team can have one or more members. 
 - An individual/team can compete on both tasks.
 - An individual associated with multiple teams (for a given task) or a team with multiple accounts will be disqualified.
 - A participant can only use the video stream as input (visual and/or audio).
 - The use of private datasets is *not* allowed. Teams using any kind of custom datasets, including additionnal annotations will be disqualified.

## How to win / What is the prize?

 - The winner is the individual/team who reaches the highest performance on the **challenge** set.
 - The metric taken into consideration is the **GS-HOTA**, a modified version of the HOTA metric.
 - The deadline to submit your results is May 30th 2024 at 11.59 pm Pacific Time.
 - To be eligible for the prize, we require the individual/team to provide a short report describing the details of the methodology (CVPR format, max 2 pages), with a short demo video on one SoccerNet challenge sequence showcasing their solution.


## Important dates

Note that these dates are tentative and subject to changes if necessary.
- November 29: Open evaluation server on the test set.
- November 29: Open evaluation server on the challenge set.
- April 24: Close evaluation server.
- May 1: Report submission deadline.
- TBD: CVSports Workshop at CVPR 2025 (awards ceremony).

## Clarifications on data usage

**1. On the restriction of private datasets and additional annotations**

SoccerNet is designed to be a research-focused benchmark, where the primary goal is to compare algorithms on equal footing. This ensures that the focus remains on algorithmic innovation rather than data collection or annotation effort. Therefore:
* Any data used for training or evaluation must be publicly accessible to everyone to prevent unfair advantages.
* By prohibiting additional manual annotations (even on publicly available data), we aim to avoid creating disparities based on resources (e.g., time, budget, or manpower). This aligns with our commitment to open-source research and reproducibility.

**2. On cleaning or correcting existing data**

We recognize that publicly available datasets, including SoccerNet datasets, might have imperfections in their labels (around 5% usually). Cleaning or correcting these labels is allowed outside of the challenge period to ensure fairness:
* Participants can propose corrections or improvements to older labels before the challenge officially starts. Such changes will be reviewed and potentially integrated into future versions of SoccerNet. Label corrections can be submitted before or after the challenge for inclusion in future SoccerNet releases, ensuring a fair and consistent dataset during the competition.
* During the challenge, participants should not manually alter or annotate existing labels, as this introduces inconsistency and undermines the benchmark's fairness.
* Fully automated methods for label refinement or augmentation, however, are encouraged. These methods should be described in the technical report to ensure transparency and reproducibility.

**3. Defining “private datasets”**

A dataset is considered “private” if it is not publicly accessible to all participants under the same conditions. For example:
* Older SoccerNet data are not private, as they are available to everyone.
* However, manually modifying or adding annotations (e.g., bounding boxes or corrected labels) to older SoccerNet data during the challenge creates a disparity and would be considered "private" unless those modifications are shared with the community in advance.

**4. Creative use of public data**

We fully support leveraging older publicly available SoccerNet data in creative and automated ways, as long as:
* The process does not involve manual annotations.
* The methodology is clearly described and reproducible.
* For instance, if you develop an algorithm that derives additional features or labels (e.g., bounding boxes) from existing data, this aligns with the challenge's goals and is permitted.

**5. Data sharing timeline:**

To ensure fairness, we decided that any new data must be published or shared with all participants through Discord at least one month before the challenge deadline. This aligns with the CVsports workshop timeline and allows all teams to retrain their methods on equal footing.


For any further doubt or concern, please raise an issue in that repository, or contact us directly on [Discord](https://discord.gg/SM8uHj9mkP).
