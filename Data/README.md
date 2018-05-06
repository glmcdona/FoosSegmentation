# Training Data for Standard Metrics
Feel free to contribute your own training data! The more variety in viewing angle and lighting the better. This training data set is for teaching a deep-learning model to extract the following information from a video file:
* Which rod curently has control of the ball
* What the score of the current game is
* What the score in the set of games is
* Whether the gameplay is active on the table (eg, timeout or pre/post match warmup)

By building a model that can extract this information from videos, we can automatically compute player metrics such as 5-to-3 pass success rate, score rate from offense, score rate from defense, success rates on 2-to-X's, block rates, etc.

## Ball Tracking Video Chunks

These video chunks are in the folder BallTracking, and used to teach the AI which rod currently controls the ball. We need contributions of video chunks here! Just follow the guidelines and naming convention, and directly upload your own video chunks to improve the model.

### Ball Tracking: Video Chunk Guidelines

#### Notes on the requirements for video chunks for training the model:

* A video chunk requires that the same Zone has control of the ball for the WHOLE video chunk.
* Chunks should not be from where the ball is still for minutes on-end.
* Doubles, singles matches, or video where you are passing the ball side-to-side by yourself is OK.
* The ball looking blurry because it's moving is great.
* Both player scores should be visible in the video frame.
* The whole table is in the video frame OR zoomed in to the far offense. It is OK to have some of the table obscured by a table light or gaps in visibily when the ball is close to the edge of the table.

#### Notes on Cutting Video Chunks

* Each video chunk should be exclusively With or Without Possession.
  <ul><li>Without Possession of the ball
  <ul><li>Timeout: START just before the hands leave the handles. END right before the ball is served.</li><li>Ball out of play: START as soon as the ball leaves the play surface. END right before the ball is served.</li><li>Between Games: START as soon as the final goal is scored. END right before the ball is served.</li></ul>
  </li><li>With Possession of the ball in a specific Zone</li>
  <ul><li>On the Serve: START after the ready rule, and ball is in play. END when no possession in that Zone.</li>
  <li>During game play: START once a player has control of the ball. END when no possession in that Zone.</li></ul></ul>
     
### Ball Tracking: Video Chunk Naming Convention

The naming convention for the video files defines how the ML model learns from the video chunk and it is important to name them correctly. Before uploading any, try and double-check each filename by quickly checking the video contents. The naming convention is:
* &lt;custom note&gt;\_&lt;rod code&gt;.mp4

Name | Description
----|-----
&lt;custom note&gt; | Match name plus chunk number here.
&lt;zone code&gt; | <ul><li>-2 Without Possession. This includes the time when a player's hand and wrist is fully over the table before/after placing the ball at the 5-bar</li><li>d1 for closest defense rod or goalie</li><li>d2 for furthest defense rod or goalie</li><li>o1 for closest offense rod</li><li>o2 for furthest offense rod</li><li>f1 for closest five-bar rod</li><li>f2 for furthest five-bar rod</li></ul>

For example for a video chunk where the close defensemen has control of the rod the whole time, I might name it:
* SpredemanVsMooreChunk1_d1.mp4

Upload your chunks to the BallTracking folder. Thanks!



## Score Tracking

These video chunks are in the folder ScoreTracking, and used to teach the AI to learn to read the current game score and number of matches won by each player. It is taught to directly read the score marked by the players, not to track goals or games. This information is needed to track when goals are scored, and who won each match.

We need contributions of video chunks here! Just follow the guidelines and naming convention, and directly upload your own video chunks to improve the model.


### Score Tracking: Video Chunk Guidelines

Notes on the requirements for video chunks for training the model:
* Video chunks are expected to be pretty long, eg 1 minute each.
* Export the video chunks at 3 FPS! The score doesn't change often.
* A video chunk requires that the game score is constant for the whole chunk (eg, it is 3 to 2 in the first game for the whole video).
* Chunks should not be from where the ball and rods is still for minutes on-end, action on the table is important.
* Doubles or singles matches is OK.
* Both player scores should be visible in the video frame.
* View of the scores should be visible for the whole chunk
* The whole table is in the video frame. It is OK to have some of the table obscured by a table light or similar.


### Score Tracking: Video Chunk Naming Convention

The naming convention for the video files defines how the ML model learns from the video chunk and it is important to name them correctly. Before uploading any, try and double-check each filename by quickly checking the video contents. The naming convention is:
* &lt;custom note&gt;\_&lt;score from close side of table&gt;\_&lt;score from far side of table&gt;\_&lt;games won from close side of table&gt;\_&lt;games won from far side of table&gt;.mp4

Name | Description
----|-----
&lt;custom note&gt; | Anything. You can write the match name and chunk number here if you want. I use the video name plus chunk number here.
&lt;score from close side of table&gt; | Score from table scoreboard on close side of table (right player's current marked score).
&lt;score from far side of table&gt; | Score from table scoreboard on far side of table (left player's current marked score).
&lt;games won from close side of table&gt; | Matches won from table scoreboard on close side of table (right player's current marked number of games won).
&lt;games won from far side of table&gt; | Matches won from table scoreboard on far side of table (left player's current marked number of games won).


For example for a video chunk where the close scoreboard has 4 points marked and the far scoreboard has 2 points marked in the current game, and the close scorebard marks 1 game won, and far scoreboard marks 0 games won in the set:
* SpredemanVsMooreChunk1_4_2_1_0.mp4

Upload your chunks to the ScoreTracking folder. Thanks!

## Recommended Tool
Shotcut works great:
* https://www.shotcut.org/download/

Steps I use:
1. Open the video with Shotcut
2. Jump to point in video where a player gets control of the ball on a rod
3. Press 'i' to begin the inset
4. Hold the right arrow (and rewind with left arrow) until the rod is still in control of that same rod
5. Press 'o' to mark this as the video outset
6. Hit File -> Export Video to bring up the export view (and keep this up for more chunks that you will create)
7. Export as '.mp4' at original resolution to a '.mp4' file following the video chunk naming convention


