package src.labs.stealth.agents;

// SYSTEM IMPORTS
import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;

import java.util.ArrayList;
import java.util.HashSet;       // will need for bfs
import java.util.Queue;         // will need for bfs
import java.util.LinkedList;    // will need for bfs
import java.util.Set;           // will need for bfs


// JAVA PROJECT IMPORTS


public class BFSMazeAgent
    extends MazeAgent
{

    public BFSMazeAgent(int playerNum)
    {
        super(playerNum);
    }

    @Override
    public Path search(Vertex src,
                       Vertex goal,
                       StateView state)
    {
        Queue<Path> queue = new LinkedList<>(); //tracks paths
        Set<Vertex> visited = new HashSet<>(); //tracks visited
        queue.add(new Path(src));
        visited.add(src);

        while (!queue.isEmpty()) {
            // Dequeue a vertex from the queue
            Path currentPath = queue.poll();
            Vertex endOfPath = currentPath.getDestination();
            int currentX = endOfPath.getXCoordinate();
            int currentY = endOfPath.getYCoordinate();

            //get new paths with valid neighbors
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
                        }else{ //add new point to path and path to queue
                            visited.add(newNode);
                            queue.add(new Path(newNode, 1.0f, currentPath) );
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
