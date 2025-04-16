package src.labs.rl.maze.agents;


// SYSTEM IMPORTS
import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.util.Direction;


import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


// JAVA PROJECT IMPORTS
import edu.bu.labs.rl.maze.agents.StochasticAgent;
import edu.bu.labs.rl.maze.agents.StochasticAgent.RewardFunction;
import edu.bu.labs.rl.maze.agents.StochasticAgent.TransitionModel;
import edu.bu.labs.rl.maze.utilities.Coordinate;
import edu.bu.labs.rl.maze.utilities.Pair;



public class ValueIterationAgent
    extends StochasticAgent
{

    public static final double GAMMA = 1; // feel free to change this around!
    public static final double EPSILON = 1e-6; // don't change this though

    private Map<Coordinate, Double> utilities;

	public ValueIterationAgent(int playerNum)
	{
		super(playerNum);
        this.utilities = null;
	}

    public Map<Coordinate, Double> getUtilities() { return this.utilities; }
    private void setUtilities(Map<Coordinate, Double> u) { this.utilities = u; }

    public boolean isTerminalState(Coordinate c)
    {
        return c.equals(StochasticAgent.POSITIVE_TERMINAL_STATE)
            || c.equals(StochasticAgent.NEGATIVE_TERMINAL_STATE);
    }

    public Map<Coordinate, Double> getZeroMap(StateView state)
    {
        Map<Coordinate, Double> m = new HashMap<Coordinate, Double>();
        for(int x = 0; x < state.getXExtent(); ++x)
        {
            for(int y = 0; y < state.getYExtent(); ++y)
            {
                if(!state.isResourceAt(x, y))
                {
                    // we can go here
                    m.put(new Coordinate(x, y), 0.0);
                }
            }
        }
        return m;
    }

    public void valueIteration(StateView state) {
        if (this.utilities == null) {
            this.setUtilities(getZeroMap(state));
        }
    
        double delta = Double.POSITIVE_INFINITY;
        while (delta > EPSILON * (1 - GAMMA) / GAMMA){
            delta = 0.0;
            Map<Coordinate, Double> newUtilities = new HashMap<>(this.utilities);
    
            for (Coordinate c : this.utilities.keySet()) {
                double utility = RewardFunction.getReward(c);
                if (!isTerminalState(c)){ // not terminal state
        
                    double maxActionUtility = Double.NEGATIVE_INFINITY;
        
                    // Calculate utility for each possible action (direction)
                    for (Direction d : TransitionModel.CARDINAL_DIRECTIONS) {
                        double actionUtility = 0.0;
        
                        // Add the expected utility of the resulting states based on transition probabilities
                        for (Pair<Coordinate, Double> transition : TransitionModel.getTransitionProbs(state, c, d)) {
                            Coordinate nextState = transition.getFirst();
                            double transitionProb = transition.getSecond();
                            actionUtility += transitionProb * this.utilities.get(nextState);
                        }
        
                        // Keep track of the maximum utility
                        maxActionUtility = Math.max(maxActionUtility, actionUtility);
                    }
                    utility += GAMMA * maxActionUtility;
                }
                // Update the utility of the current state
                newUtilities.put(c, utility);
        
                if(Math.abs(newUtilities.get(c) - this.utilities.get(c)) > delta){
                    delta = Math.abs(newUtilities.get(c) - this.utilities.get(c));
                }
            }
    
            // Update utilities map and print debug information
            this.setUtilities(newUtilities);
    
        };
    }
    

    @Override
    public void computePolicy(StateView state,
                              HistoryView history)
    {
        this.valueIteration(state);

        // compute the policy from the utilities
        Map<Coordinate, Direction> policy = new HashMap<Coordinate, Direction>();

        for(Coordinate c : this.getUtilities().keySet())
        {
            // figure out what to do when in this state
            double maxActionUtility = Double.NEGATIVE_INFINITY;
            Direction bestDirection = null;

            for(Direction d : TransitionModel.CARDINAL_DIRECTIONS)
            {
                double thisActionUtility = 0.0;
                for(Pair<Coordinate, Double> transition : TransitionModel.getTransitionProbs(state, c, d))
                {
                    thisActionUtility += transition.getSecond() * this.getUtilities().get(transition.getFirst());
                }

                if(thisActionUtility > maxActionUtility)
                {
                    maxActionUtility = thisActionUtility;
                    bestDirection = d;
                }
            }

            policy.put(c, bestDirection);

        }

        this.setPolicy(policy);
    }

}
