# Guidelines for the Game State Reconstruction Challenge

The [Game State Reconstruction]() will be held at the official [CVSports Workshop](https://vap.aau.dk/cvsports/) at CVPR 2024! 
Subscribe (watch) the repo to receive the latest info regarding timeline and prizes!

We propose the SoccerNet challenges to encourage the development of state-of-the-art algorithm for Soccer Video Understanding.

 - **Game State Reconstruction**: Maintain a complete understanding of the game by predicting : 
   - top-view player localization (optionally with camera calibration)
   - role classification (player, referee, other)
   - team affiliation and team side
   - jersey number recognition

We provide an [evaluation server](https://eval.ai/web/challenges/challenge-page/2202/overview) for the Game State Reconstruction task. 
The evaluation server handles predictions for the open **test** sets and the segregated **challenge** sets of each challenge.

Winners will be announced at CVSports Workshop at CVPR 2024.
Prizes and sponsors will be announced soon, stay tuned!


## Who can participate / How to participate?

 - Any individual can participate to the challenge, except the organizers.
 - The participants are recommended to form a team to participate.
 - Each team can have one or more members. 
 - An individual/team can compete on both task.
 - An individual associated with multiple teams (for a given task) or a team with multiple accounts will be disqualified.
 - On both task, a particpant can only use the video stream as input (visual and/or audio).
 - To help the participants, we provide pre-extracted ResNet-152 visual features at 2fps.
 - A particpant is allowed to extract its own visual/audio features with any pre-trained model.

## How to win / What is the prize?

 - The winner is the individual/team who reach the highest performance on the **challenge** set.
 - The metrics taken into consideration is the **Game-state-HOTA**, a modified version of the HOTA metric.
 - The deadline to submit your results is May 30th 2024 at 11.59 pm Pacific Time.
 - In order to be eligible for the prize, we require the individual/team to provide a short report describing the details of the methodology (CVPR format, max 2 pages).


## Important dates

Note that these dates are tentative and subject to changes if necessary.

 - **February 5:** Open evaluation server on the (Open) Test set.
 - **February 5:** Open evaluation server on the (Seggregated) Challenge set.
 - **May 30:** Close evaluation server.
 - **June 6:** Deadline for submitting the report.
 - **June TBD:** A full-day workshop at CVPR 2024.

For any further doubt or concern, please raise an issue in that repository, or contact us directly on [Discord](https://discord.gg/SM8uHj9mkP).
