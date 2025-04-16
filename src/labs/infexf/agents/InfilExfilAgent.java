package src.labs.infexf.agents;

import java.util.Set;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;

// SYSTEM IMPORTS
import edu.bu.labs.infexf.agents.SpecOpsAgent;
import edu.bu.labs.infexf.distance.DistanceMetric;
import edu.bu.labs.infexf.graph.Vertex;
import edu.bu.labs.infexf.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;


// JAVA PROJECT IMPORTS
import edu.cwru.sepia.environment.model.state.Unit.UnitView;


public class InfilExfilAgent
    extends SpecOpsAgent
{

    public InfilExfilAgent(int playerNum)
    {
        super(playerNum);
    }

    // if you want to get attack-radius of an enemy, you can do so through the enemy unit's UnitView
    // Every unit is constructed from an xml schema for that unit's type.
    // We can lookup the "range" of the unit using the following line of code (assuming we know the id):
    //     int attackRadius = state.getUnit(enemyUnitID).getTemplateView().getRange();
    @Override
    public float getEdgeWeight(Vertex src,
                               Vertex dst,
                               StateView state)
    {
        int xdst = dst.getXCoordinate();
        int ydst = dst.getYCoordinate();
        Set<Integer> enemies = getOtherEnemyUnitIDs();
        int board = state.getXExtent();
        float risk = 0.0f; //base risk is 1.0f

        for(int enemy : enemies){
            UnitView enemyUnit = state.getUnit(enemy);  // Get the unit associated with this enemy ID
        
            if (enemyUnit != null) {  // Check if the unit exists
                int xdist = Math.abs(enemyUnit.getXPosition() - xdst);
                int ydist = Math.abs(enemyUnit.getYPosition() - ydst);

                double dist = Math.abs(xdist + ydist);

                
                if(dist <=3 ){ //at most 2 moves away
                    return 100000000.0f;
                } else{
                    risk += Math.pow(2, Math.abs(board-dist));
                }
            }
        }
        return risk;
    }

    @Override
    public boolean shouldReplacePlan(StateView state)
    {
        int xpos = state.getUnit(getMyUnitID()).getXPosition();
        int ypos = state.getUnit(getMyUnitID()).getYPosition();
        Vertex src = new Vertex(xpos, ypos);

        Vertex dst = getNextVertexToMoveTo();
        if (dst == null) {
            return true;
        }

        float edgeWeight = getEdgeWeight(src, dst, state);
        if(edgeWeight >= 10.0){
            return true;
        }
        
        return false;
    }

}
