package src.labs.stealth.agents;

// SYSTEM IMPORTS
import edu.bu.labs.stealth.agents.MazeAgent;
import edu.bu.labs.stealth.graph.Vertex;
import edu.bu.labs.stealth.graph.Path;


import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.util.Direction;                           // Directions in Sepia


import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue; // heap in java
import java.util.Set;


// JAVA PROJECT IMPORTS


public class DijkstraMazeAgent
    extends MazeAgent
{

    public DijkstraMazeAgent(int playerNum)
    {
        super(playerNum);
    }

    @Override
    public Path search(Vertex src,
                       Vertex goal,
                       StateView state)
    {
		PriorityQueue<Path> queue = new PriorityQueue<>(new Comparator<Path>() {
            @Override
            public int compare(Path p1, Path p2) {
                return Float.compare(p1.getTrueCost(), p2.getTrueCost());
            }
        });
        
        HashMap <Direction, Float> cost = new HashMap<>();
        Set<Vertex> visited = new HashSet<>(); //tracks visited

        queue.add(new Path(src));
        visited.add(src);

        cost.put(Direction.EAST,5.0f);
        cost.put(Direction.WEST,5.0f);
        cost.put(Direction.NORTH,10.0f);
        cost.put(Direction.SOUTH,1.0f);
        cost.put(Direction.NORTHEAST, (float)Math.sqrt(cost.get(Direction.NORTH)*cost.get(Direction.NORTH) +cost.get(Direction.EAST)*cost.get(Direction.EAST)));
        cost.put(Direction.NORTHWEST, (float)Math.sqrt(cost.get(Direction.NORTH)*cost.get(Direction.NORTH) +cost.get(Direction.WEST)*cost.get(Direction.WEST)));
        cost.put(Direction.SOUTHEAST, (float)Math.sqrt(cost.get(Direction.SOUTH)*cost.get(Direction.SOUTH) +cost.get(Direction.EAST)*cost.get(Direction.EAST)));
        cost.put(Direction.SOUTHWEST, (float)Math.sqrt(cost.get(Direction.SOUTH)*cost.get(Direction.SOUTH) +cost.get(Direction.WEST)*cost.get(Direction.WEST)));
        
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
                            return currentPath;
                        }else if(visited.contains(newNode) || state.resourceAt(currentX+i, currentY+j)!=null || state.isUnitAt(currentX+i, currentY+j)){ //already visited node
                            continue;
                        }else{ //add new point to path and path to queue
                            visited.add(newNode);
                            Direction move = getDirectionToMoveTo(endOfPath, newNode);
                            queue.add(new Path(newNode, cost.get(move), currentPath) );
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
