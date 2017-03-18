#  Author: Shane Griffith
#  2013.
#  
#  S. Griffith, K. Subramanian, J. Scholz, C. Isbell, and A. Thomaz. ``Policy Shaping:
#  Integrating Human Feedback with Reinforcement Learning.'' in NIPS. 2013.
#
#  This is some sample code to help you get started. It includes an implementation for
#    Advise, assuming you've got the other stuff (simulator, oracle, RL algorithm).
#
#Functions:
#  UpdateOraclePolicy() implements eq. 2 in our paper. 
#  ActionSelection() implements the combination method.
#  AddStateToList() defines what's saved for each state.
#  update(), __init__(), and ActionSelection() are called by the simulator.
#
#Other:
#  This implemenation extends something called BayesianRLEliminateMultipleOptimalActions,
#    which performs Bayesian Q-Learning, generates oracle feedback, and ensures there's
#    one optimal action per state.



class BayesianRLAdvise(BayesianRLEliminateMultipleOptimalActions):
  def __init__(self, gameType, save=False):
      BayesianRLEliminateMultipleOptimalActions.__init__(self, gameType, save=False);


  def AddStateToList(self, state, actions):
  #Defines the state
      if not self.stateindex.has_key(state) :
          sidx = len(self.knownstates);
          self.stateindex[state] = sidx;
          stateinfo = [];

          for action in actions :
              #the information saved for each state-action pair
              #[0) action,
              # 1) BQL hyperparameters,
              # 2) bql probability this state,
              # 3) advise probability this s,a pair
              # ]
              stateinfo.append([action, self.priorhyperparameters, 1.0/len(actions), 1.0/len(actions)]);
          
          self.knownstates.append(stateinfo);



  def UpdateOraclePolicy(self, state, action, bAGREE):
  #This function is called by something that implements an oracle.
  #This implements the policy generation method of Advise for the case
  # with only one optimal action per state (Eq. 2 in our paper).
  #This implementation avoids the problems with underflow  (floating point
  #  values too small) and overflow (exponents too large).
  
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      
      #use the feedback consistency to compute the oracle policy
      #self.fconsistency is an input parameter to this script.
      # (We supply the value of C a priori.)
      if self.fconsistency >=0 or self.fconsistency <= 20 :
          CONSISTENCY = 1.0 - self.fconsistency * 0.05;
      else :
          print 'the test should be 0 through 20.'
          quit()
      
      C = CONSISTENCY;
      if C == 1.0 :
          C = 0.99999999999999;
      elif C == 0.0 :
          C = 0.00000000000001;
      
      
      sum = 0.0;
      try :
          for i in range(0, len(statedata)) :
              if statedata[i][0] == action:
                  if bAGREE:
                      statedata[i][3] = statedata[i][3]*C;
                  else :
                      statedata[i][3] = statedata[i][3]*(1-C);
              else :
                   if bAGREE:
                       statedata[i][3] = statedata[i][3]*(1-C);
                   else :
                       statedata[i][3] = statedata[i][3]*C;
              sum = sum + statedata[i][3];
      except :
          print 'exception... overflow'
          return [];

      try :
          #normalize the distribution
          for i in range(0,len(statedata)) :
              statedata[i][3] = statedata[i][3]/sum;
      except :
          print 'exception... underflow'
          return [];


  def update(self, state, action, nextState, reward):
  #This function is called by the simulator when the state changes.
      BayesianRLEliminateMultipleOptimalActions.update(self, state, action, nextState, reward);
      
      #update the Q_distribution every turn based on the changes to the BQL normal gamma distribution.
      Q_distr = BayesianRLEliminateMultipleOptimalActions.EstimateTheProbabilityThatActionsAreOptimal(self, state);
      
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      for i in range(0, len(statedata)) :
          statedata[i][2] = Q_distr[i];


  def ChooseAction(self, prob_distr):
  #Sample an action according to the probability distribution.
      chosena = random.random();
      sum = 0;
      count = 0
      for actioni in range(0, len(prob_distr)) :
          sum = sum + prob_distr[actioni];
          if chosena < sum :
              break;
          
          count = count + 1
      
      #an edge case: when the probability distribution adds up to slightly less
      #  than 1.0, but the prob distr chose (something near) 1.0
      if count >= len(prob_distr) :
          count = len(prob_distr) - 1;
      
      return count;


  def ActionSelection(self, state):
  #The combination method of Advise.
    sidx = self.stateindex[state];
    statedata = self.knownstates[sidx];
    
    #multiply them together
    C_distr = [];
    sum = 0.0;
    try :
        for i in range(0, len(statedata)) :
            C_distr.append(statedata[i][2]*statedata[i][3]);
            sum = sum + C_distr[i];
        
        #normalize the distribution
        for i in range(0, len(O_distr)) :
            C_distr[i] = C_distr[i]/sum;
    except :
        print 'exception...underflow'
        return [];
    
    #sample an action from the policy
    aidx = self.ChooseAction(C_distr);
    
    #return the action
    return statedata[aidx][0];
