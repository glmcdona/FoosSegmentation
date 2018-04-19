# Training Data for Standard Metrics
Feel free to contribute your own training data! The more variety in viewing angle and lighting the better. This training data set is for teaching a deep-learning model to extract the following information from a video file:
* Which rod curently has control of the ball
* What the score of the current game is
* What the score in the set of games is
* Whether the gameplay is active on the table (eg, timeout or pre/post match warmup)

Notes on the requirements for video chunks for training the model:
* Both player scores must be visible in the video frame.
* The whole table is in the video frame. It is OK to have some of the table obscured by a table light or gaps in visibily when the ball is close to the edge of the table.
* A video chunk requires that the same rod has control of the ball for the whole video chunk, but can be passing the ball around on that rod or can be in a timeout for example.
* Chunks should be 15 seconds or less each.
* Doubles or singles matches is OK.



The naming convention for the video files defines how the ML model learns from the video chunk and it is important to name them correctly. Before uploading any, try and double-check each filename by quickly checking the video contents. The naming convention is:
* chunk\_&lt;game status&gt;\_&lt;rod code&gt;\_&lt;score from close side of table&gt;\_&lt;score from far side of table&gt;\_&lt;games from close side&gt;\_&lt;games from far side&gt;\_&lt;custom note&gt;.avi



Name | Description
----|-----
&lt;game status&gt; | <ul><li>1 for game in play</li><li>1 for not in play (timeout, or warmup between/before/after games)</li></ul>
&lt;rod code&gt; | <ul><li>-2 for ball out of play for whole chunk</li><li>-1 for ball bouncing around between rods for whole chunk (no rod has control of the ball for any part of the chunk)</li><li>g1 for the closest goalie rod</li><li>g2 for the furthest goalie rod</li><li>d1 for closest defense rod</li><li>d2 for furthest defense rod</li><li>o1 for closest offense rod</li><li>o2 for furthest offense rod</li><li>f1 for closest five-bar rod</li><li>f2 for furthest five-bar rod</li></ul>
&lt;score from close side of table&gt; | Score from table scoreboard on close side of table (right player's current marked score).
&lt;score from far side of table&gt; | Score from table scoreboard on far side of table (left player's current marked score).
&lt;games from close side&gt; | Matches won from table scoreboard on close side of table (right player's current marked number of games won).
&lt;games from far side&gt; | Matches won from table scoreboard on far side of table (left player's current marked number of games won).
&lt;custom note&gt; | Anything. You can write the match name and chunk number here if you want.

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

