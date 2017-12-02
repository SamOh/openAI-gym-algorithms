def getPos(obs):
  return ((obs % 4) + 1, (obs / 4) + 1)





def getTransitionStatesAndProbs(self,state,action):
	# left, down, right, up = 0, 1, 2, 3
	zero , third, tThird = 0, 1.0/3 , 2.0/3
	x, y = state
	left, down, right, up = (x-1,y), (x,y+1), (x+1,y),(x,y-1)


	if state == (1,1):
		return options([(state, tThird), (down, third)], \
				[(down, third), (right, third), (state, third)], \
				[(right, third), (down, third), (state, third)], \
				[(state, tThird), (right, third)], action)


	if state == (1,4):
		return options([(state, tThird), (up, third)], \
				[(state, tThird), (right, third)], \
				[(right, third), (up, third), (state, third)], \
				[(state, third), (right, third), (up, third)], action)
	
	if state == (4,1):
		return options([(state, third), (down, third), (left, third)], \
				[(state, third), (left, third), ( down ,third)], \
				[(state, tThird), (down, third)], \
				[(state, tThird), (left, third)], action)

	if state == (4,4):
		return options([(state, third), (left, third), (up, third)], \
				[(state, tThird), (left, third)], \
				[(state, tThird), (up, third)], \
				[(state, third), (up, third), (left, third)], action)

    if state == (1,2) or state == (1,3):
      return options([(state, third), (up, third), (down, third)], \
                     [(state, third), (down, third), (right, third)], \
                     [(right, third), (up, third), (down, third)], \
                     [(state, third), (up, third), (right, third)], action)
    if state == (2,1) or state == (3,1):
      return options([(state, third), (left, third), (down, third)], \
                     [(left, third), (down, third), (right, third)], \
                     [(state, third), (right, third), (down, third)], \
                     [(state, third), (left, third), (right, third)], action)
    if state == (4,2) or state == (4,3):
      return options([(left, third), (up, third), (down, third)], \
                     [(state, third), (down, third), (left, third)], \
                     [(state, third), (up, third), (down, third)], \
                     [(state, third), (up, third), (left, third)], action)
    if state == (2,4) or state == (3,4):
      return options([(state, third), (up, third), (left, third)], \
                     [(state, third), (left, third), (right, third)], \
                     [(state, third), (up, third), (right, third)], \
                     [(up, third), (left, third), (right, third)], action)
    if state == (2,2) or state == (2,3) or state == (3,2) or state == (3,3):
      return options([(left, third), (up, third), (down, third)], \
                     [(down, third), (left, third), (right, third)], \
                     [(right, third), (up, third), (down, third)], \
                     [(up, third), (left, third), (right, third)], action)

