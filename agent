class Bot:
  def __init__(self,n , goal=50000, end=False, init_stock=31, lr=0.1, gamma=1, debug=False, filename=FILE_BTC, start_date=STARTDATE, end_date=ENDDATE):
    self.actions = ["Sell","Keep"]
    self.state = "HIGHER"  # current state
    self.end = end
    self.n = n
    self.lr = lr
    self.gamma = gamma
    self.debug = debug

    self.start_date=start_date
    self.end_date=end_date

    #Timestep for indexing
    self.time=0

    #First capital #will be profit and loss from sell
    self.capital=0

    #First # of stock
    self.num_stock=init_stock

    #Goal
    self.goal=goal

    #Create environment
    self.df=pd.read_csv(filename, parse_dates=['Date'])

    after_start_date = self.df["Date"] >= self.start_date
    before_end_date = self.df["Date"] <= self.end_date
    between_two_dates = after_start_date & before_end_date
    self.filtered_df = self.df.loc[between_two_dates]
    self.high=self.filtered_df["High"] #Only "High" value
    
    #Price reference (price of day0)
    self.ref_price=(self.filtered_df.iloc[0])["High"]

    # init q estimates
    self.Q_values = {}

    #for i in ["State0","LOWER","HIGHER","END"]:

    states=["LOWER","HIGHER"]

    for i in states:
      self.Q_values[i] = {}
      for a in self.actions:
        self.Q_values[i][a] = 0

  def getState(self, time): #Return state value from index

      index=time+1 #since index 0 is for ref*****

      if index==32:
        self.end=True
        print("End because of time")
        print("____________________________________________________")
      
      elif self.high.iloc[index-1] > self.high.iloc[index]:
        return "LOWER" #lower than the day before

      else:
        return "HIGHER"
      

  def chooseAction(self):
      action = np.random.choice(self.actions)
      return action
  
  def returnhighprice(self, time):
      index=time+1 #since index 0 is for ref time index starts at 0 but date starts at 1*****
      return self.high.iloc[index]

  def takeAction(self, action):

      self.state=self.getState(self.time)
      print("State",self.state,"At date", self.time+1) #time index starts at 0 but date starts at 1*****

      #Action and sequences

      if not self.end:

        if action=="Sell":
          print("Sell")
          self.num_stock = self.num_stock-1
          print("num_stock left", self.num_stock)
          print("capital before", self.capital, "Ref price", self.ref_price)

          self.capital += self.returnhighprice(self.time)-self.ref_price 

          print("capital after", self.capital, "Price that day", self.high.iloc[self.time+1])

        #if action=="Hold" : Nothing happens

        if self.capital >= self.goal: #ตรงนี้ทำเพิ่ม maybe it can converge faster??-??/ if reach goal then episode ends loeyyyyy
          print("End because it reaches goal")
          print("____________________________________________________")
          self.end=True

        elif self.num_stock<=0:
          print("End because num_stock is 0")
          print("____________________________________________________")
          self.end=True


        self.time+=1
        new_state = self.getState(self.time)

      self.state=new_state

      return self.state
          

  def giveReward(self):
      if self.capital >= self.goal:
        return 1
      elif self.capital <0: #If the capital is negative but it  #may delete this later
        return -1
      else:
        return 0

  def reset(self):
      self.state = "HIGHER"
      self.end = False
      self.time=0
      self.capital=0
      self.num_stock=31

  def play(self, rounds=100):
      for _ in range(rounds):
          self.reset()
          t = 0
          T = np.inf
          #T=31
          action = self.chooseAction()

          actions = [action]
          states = [self.state]
          rewards = [0]
          while True:
              if t < T:
                  #print("t",t)
                  #print("T",T)
                  state = self.takeAction(action)  # next state
                  reward = self.giveReward()  # next state-reward

                  states.append(state)
                  rewards.append(reward)

                  if self.end:
                      if self.debug:
                          print("End at state {} | number of states {}".format(state, len(states)))
                      T = t + 1
                  else:
                      action = self.chooseAction()
                      actions.append(action)  # next action
              # state tau being updated
              tau = t - self.n + 1
              if tau >= 0:
                  G = 0
                  for i in range(tau + 1, min(tau + self.n + 1, T + 1)):
                      G += np.power(self.gamma, i - tau - 1) * rewards[i]
                  if tau + self.n < T:
                      state_action = (states[tau + self.n], actions[tau + self.n])
                      #print("State_action",state_action)
                      G += np.power(self.gamma, self.n) * self.Q_values[state_action[0]][state_action[1]]
                  # update Q values
                  state_action = (states[tau], actions[tau])
                  #print(state_action[0])
                  self.Q_values[state_action[0]][state_action[1]] += self.lr * (
                      G - self.Q_values[state_action[0]][state_action[1]])

              if tau == T - 1:
                  break

              t += 1
