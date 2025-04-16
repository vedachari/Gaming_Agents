package src.pas.chess.moveorder;


// SYSTEM IMPORTS
import edu.bu.chess.search.DFSTreeNode;

import java.util.List;
import java.util.LinkedList;


public class CustomMoveOrderer
    extends Object
{

	/**
	 * This method should perform move ordering. Remember, move ordering is how alpha-beta pruning gets part of its power from.
	 * You want to see nodes which are beneficial FIRST so you can prune as much as possible during the search (i.e. be faster)
	 * @param nodes. The nodes to order (these are children of a DFSTreeNode) that we are about to consider in the search.
	 * @return The ordered nodes.
	 */
	public static List<DFSTreeNode> order(List<DFSTreeNode> nodes)
	{
		List<DFSTreeNode> orderedNodes = new LinkedList<>();

        // Temporary lists for categorizing moves
        List<DFSTreeNode> captureMoves = new LinkedList<>();
        List<DFSTreeNode> castleMoves = new LinkedList<>();
        List<DFSTreeNode> promotePawnMoves = new LinkedList<>();
        List<DFSTreeNode> enPassantMoves = new LinkedList<>();
        List<DFSTreeNode> movementMoves = new LinkedList<>();

        // Categorize moves
        for (DFSTreeNode node : nodes) {
            if (node.getMove() != null) {
                switch (node.getMove().getType()) {
                    case CAPTUREMOVE:
                        captureMoves.add(node);
                        break;
                    case CASTLEMOVE:
                        castleMoves.add(node);
                        break;
                    case PROMOTEPAWNMOVE:
                        promotePawnMoves.add(node);
                        break;
                    case ENPASSANTMOVE:
                        enPassantMoves.add(node);
                        break;
                    case MOVEMENTMOVE:
                        movementMoves.add(node);
                        break;
                }
            } else {
                movementMoves.add(node); // Add null moves to movementMoves
            }
        }

        // Prioritize moves: Captures > Promotions > Castles > En Passant > Other Moves
        orderedNodes.addAll(captureMoves);
        orderedNodes.addAll(promotePawnMoves);
        orderedNodes.addAll(castleMoves);
        orderedNodes.addAll(enPassantMoves);
        orderedNodes.addAll(movementMoves);

        return orderedNodes;
	}

}