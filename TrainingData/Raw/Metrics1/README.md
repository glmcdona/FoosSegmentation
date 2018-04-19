# Training Data for Standard Metrics
Feel free to contribute your own training data! The more variety in viewing angle and lighting the better. This training data set is for teaching a deep-learning model to extract the following information from a video file:
* Which rod curently has control of the ball
* What the score of the current game is
* What the score in the set of games is
* Whether the gameplay is active on the table (eg, timeout or pre/post match warmup)

Notes on the requirements for video chunks for training the model:
* Both player scores must be visible in the video frame.
* A video chunk requires that the same rod has control of the ball for the whole video chunk, but can be passing the ball around on that rod or can be in a timeout for example.
* Chunks should be 15 seconds or less each.
* Doubles or singles matches is OK.

The naming convention for the video files defines how the ML model learns from the video chunk and it is important to name them correctly. Before uploading any, try and double-check each filename by quickly checking the video contents. The naming convention is:
* chunk_<game status>_<rod code>_<score from close side of table>_<score from far side of table>_<games from close side>_<games from far side>_<custom note>.avi

And each of these are defined as:
Name | Description
--- | ---
<game status> | <ul><li>1 for game in play</li><li>1 for not in play (timeout, or warmup between/before/after games)</li></ul>
<rod code> | <ul><li>-2 for ball out of play for whole chunk</li><li>-1 for ball bouncing around between rods for whole chunk (no rod has control of the ball for any part of the chunk)</li><li>g1 for the closest goalie rod</li><li>g2 for the furthest goalie rod</li><li>d1 for closest defense rod</li><li>d2 for furthest defense rod</li><li>o1 for closest offense rod</li><li>o2 for furthest offense rod</li><li>f1 for closest five-bar rod</li><li>f2 for furthest five-bar rod</li></ul>
<score from close side of table> | Score from table scoreboard on close side of table (right player's current marked score).
<score from far side of table> | Score from table scoreboard on far side of table (left player's current marked score).
<games from close side> | Matches won from table scoreboard on close side of table (right player's current marked number of games won).
<games from far side> | Matches won from table scoreboard on far side of table (left player's current marked number of games won).
<custom note> | Anything. You can write the match name and chunk number here if you want.

For example:
* chunk_1_d2_3_4_0_1_SpredemanVsMooreChunk1.avi

Would mark the video chunk as training for:
* 1: Game is active
* d2: The far defensemen (2bar) has control of the ball for the whole chunk
* 3: Close player has 3 points marked on scoreboard in current game
* 4: Far player has 4 points marked on scoreboard in current game
* 0: Close player has 0 game wins marked on scoreboard in current set
* 1: Far player has 1 game wins marked on scoreboard in current set
* SpredemanVsMooreChunk1: Your note for this chunk

