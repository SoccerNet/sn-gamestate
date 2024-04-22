# Guidelines for the Game State Reconstruction Challenge

The [Game State Reconstruction]() will be held at the official [CVSports Workshop](https://vap.aau.dk/cvsports/) at CVPR 2024! 
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
 - On both tasks, a participant can only use the video stream as input (visual and/or audio).

## How to win / What is the prize?

 - The winner is the individual/team who reaches the highest performance on the **challenge** set.
 - The metric taken into consideration is the **GS-HOTA**, a modified version of the HOTA metric.
 - The deadline to submit your results is May 30th 2024 at 11.59 pm Pacific Time.
 - To be eligible for the prize, we require the individual/team to provide a short report describing the details of the methodology (CVPR format, max 2 pages), with a short demo video on one SoccerNet challenge sequence showcasing their solution.


## Important dates

Note that these dates are tentative and subject to changes if necessary.

 - **March 26:** Open evaluation server on the (Open) Test set.
 - **???:** Open evaluation server on the (Seggregated) Challenge set.
 - **May 30:** Close evaluation server.
 - **June 6:** Deadline for submitting the report.
 - **June TBD:** A full-day workshop at CVPR 2024.

For any further doubt or concern, please raise an issue in that repository, or contact us directly on [Discord](https://discord.gg/SM8uHj9mkP).
