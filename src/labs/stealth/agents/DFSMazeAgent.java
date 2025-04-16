package src.labs.stealth.agents;

// SYSTEM IMPORTS
import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;


import java.util.HashSet;   // will need for dfs
import java.util.Stack;     // will need for dfs
import java.util.Set;       // will need for dfs


// JAVA PROJECT IMPORTS


public class DFSMazeAgent
    extends MazeAgent
{

    public DFSMazeAgent(int playerNum)
    {
        super(playerNum);
    }

    @Override
    public Path search(Vertex src,
                       Vertex goal,
                       StateView state)
    {
        Stack<Path> stack = new Stack<>();
        Set<Vertex> visited = new HashSet<>(); //tracks visited

        stack.add(new Path(src));
        visited.add(src);

        while (!stack.isEmpty()) {
            // Pop a vertex from the stack
            Path currentPath = stack.pop();
            Vertex endOfPath = currentPath.getDestination();
            int currentX = endOfPath.getXCoordinate();
            int currentY = endOfPath.getYCoordinate();

            for( int i=-1; i<=1; i++){
                for(int j =-1; j<=1; j++){
                    if(state.inBounds(currentX+i, currentY+j)){
                        Vertex newNode = new Vertex(currentX+i, currentY+j);
                        if(newNode.equals(goal)){ //return first path to reach adjacent to final position
                            System.out.println(currentPath.toString());
                            return currentPath;
                        }else if(visited.contains(newNode)){ //already visited node
                            continue;
                        }else if(state.resourceAt(currentX+i, currentY+j)!=null){
                            continue;
                        }else{ //add new point to path and path to stack
                            visited.add(newNode);
                            stack.add(new Path(newNode, 1.0f, currentPath) );
                        }
                    }
                }
            }   
        }

        return null;
    }

    @Override
    public boolean shouldReplacePlan(StateView state)
    {
        return false;
    }

}
