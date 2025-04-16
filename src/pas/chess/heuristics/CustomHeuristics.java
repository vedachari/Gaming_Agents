package src.pas.chess.heuristics;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.bu.chess.game.Board;
import edu.bu.chess.game.Game;
import edu.bu.chess.game.move.Move;
import edu.bu.chess.game.move.MoveType;
import edu.bu.chess.search.DFSTreeNode;
import edu.bu.chess.utils.Coordinate;
import edu.bu.chess.game.piece.Pawn;
import edu.bu.chess.game.piece.Piece;
import edu.bu.chess.game.piece.PieceType;
import edu.bu.chess.game.player.Player;


public class CustomHeuristics {

    /**
     * Calculate the custom heuristic value for the max player in the current node.
     * @param node The current state of the game tree node.
     * @return A heuristic value indicating the max player's advantage in this position.
     */
    public static double getMaxPlayerHeuristicValue(DFSTreeNode node) {
        Game game = node.getGame();
        Board board = game.getBoard();
        double materialAdvantage = 0.0;
        double mobilityAdvantage = 0.0;
        double pawnStructure = 0.0;
        double passedPawns = 0.0;
        double developmentLead = 0.0;
        double centerControl = 0.0;
        double pleaseCastle = 0.0;
        double defensiveScore = 0.0;
        double kingSafety = getKingSafety(node);
    
        Player maxPlayer = getMaxPlayer(node);
        Player minPlayer = getMinPlayer(node);
    
        // Determine the color of the max player
        boolean maxIsBlack = maxPlayer.getAlgebraicSymbol().equals("B");
    
        Set<Piece> maxPlayerPieces = board.getPieces(maxPlayer);
        Set<Piece> minPlayerPieces = board.getPieces(minPlayer);
    
        Piece king = null;
        Piece rook1 = null;
        Piece rook2 = null;
    
        int wPieces = maxPlayerPieces.size();
        int bPieces = minPlayerPieces.size();
        String gameState = stageOfGame(wPieces, bPieces);
    
        boolean maxQSurvives = false;
        boolean minQSurvives = false;
    
        // Calculate values for the max player's pieces
        for (Piece piece : maxPlayerPieces) { 
            materialAdvantage += getPieceValue(piece.getType());
            mobilityAdvantage += getLegalMovesCount(node, maxPlayer, piece);
            
            switch (piece.getType()) {
                case PAWN:
                    pawnStructure += evalPawnStructure(node, piece, game, board, gameState, true);
                    if (isPassed(piece, board, maxIsBlack)) {
                        passedPawns += 1.0;
                    }
                    break;
                case QUEEN:
                    mobilityAdvantage += queenMovement(node, piece, board, gameState);
                    maxQSurvives = true;
                    break;
                case BISHOP:
                    mobilityAdvantage += bishopMovement(node, gameState, bPieces, board, maxIsBlack);
                    break;
                case KING:
                    king = piece;
                    break;
                case ROOK:
                    if (rook1 == null) {
                        rook1 = piece;
                    } else {
                        rook2 = piece;
                    }
                    break;
                case KNIGHT:
                    mobilityAdvantage += evaluateKnightPosition(piece, board);
                    break;
                
            }
    
            centerControl += controlsCenter(node, piece, board);
            developmentLead += isDevelopingBoard(piece, board, maxIsBlack);
        }
        pleaseCastle += castle(node, king, rook1, rook2, board, maxIsBlack, gameState); // Add utility for castling
    
        // Calculate values for the min player's pieces
        for (Piece piece : minPlayerPieces) { 
            defensiveScore -= getPieceValue(piece.getType());
            defensiveScore -= getLegalMovesCount(node, minPlayer, piece);
            if (piece.getType() == PieceType.QUEEN) {
                minQSurvives = true;
            }
            if (piece.getType() == PieceType.PAWN) {
                defensiveScore -= evalPawnStructure(node, piece, game, board, gameState, !maxIsBlack);
            }
        }
    
        double heuristicValue = 0.0;
    
        if (!(minQSurvives && maxQSurvives)) { //end phase one queen is gone
            gameState = "end";
        }
    
        // Adjusted weightings for different game stages
        switch (gameState) {
            case "open": // Opening
                heuristicValue = 1.2 * materialAdvantage 
                            + 1.4 * mobilityAdvantage 
                            + 1.3 * kingSafety
                            + 2.0 * pawnStructure
                            + 2.0 * developmentLead
                            + 1.5 * pleaseCastle
                            + 0.9 * defensiveScore; // Adjusted weights for openings
                break;
            case "middle": // Middle game
                heuristicValue = 2.0 * materialAdvantage 
                            + 1.5 * mobilityAdvantage 
                            + 2.2 * kingSafety
                            + 1.6 * centerControl
                            + 1.0 * pawnStructure
                            + 0.8 * developmentLead
                            + 1.2 * pleaseCastle
                            + 1.8 * defensiveScore; // Tuned weights for the middle game
                break;
            default: // Endgame
                heuristicValue = 3.0 * materialAdvantage // Increased caution in endgames
                            + 1.3 * mobilityAdvantage 
                            + 0.6 * kingSafety
                            + 2.5 * centerControl
                            + 4.5 * pawnStructure // Strong focus on pawns
                            + 0.5 * developmentLead
                            + 3.5 * passedPawns // Emphasizing passed pawns
                            + 2.5 * defensiveScore; // Endgame defense focus
                break;
        }
        if(!minQSurvives && maxQSurvives){
            heuristicValue += 5.0; //add a bit if it gets rid of their queen but ours survives
        }
    
        return heuristicValue;
    }
    

    //ALL PIECES

    /**
     * Assign point values to each type of piece.
     * @param pieceType The type of the piece.
     * @return The value of the piece.
     */
    private static double getPieceValue(PieceType pieceType) {
        // Predefined piece values
        switch (pieceType) {
            case PAWN: return 1.0;
            case KNIGHT: return 3.0;
            case BISHOP: return 3.0;
            case ROOK: return 5.0;
            case QUEEN: return 9.0;
            case KING: return 200.0;  // High value to encourage king safety
            default: return 0.0;
        }
    }

    /**
     * Count the number of legal moves a piece can make, with a bonus if it is forking two pieces.
     * @param node The current game state.
     * @param player The player who own's this piece
     * @param piece The piece whose mobility is being measured.
     * @return The number of legal moves.
     */
    private static double getLegalMovesCount(DFSTreeNode node, Player player, Piece piece) {
        double movesUtility = 0.0;
        List<Move> movesForPiece = node.getGame().getAllMovesForPiece(piece.getPlayer(), piece);

        movesUtility += movesForPiece.size();
        movesUtility += isForking(node, player, movesForPiece);
        return movesUtility;
    }

    /**
     * Returns utility for forking multiple pieces (can capture multiple pieces).
     * @param node The current game state.
     * @param piece The piece whose mobility is being measured.
     * @param moves The possible moves list for a piece
     * @return The number of legal moves.
     */
    private static int isForking(DFSTreeNode node, Player player, List<Move> moves) {
        return node.getGame().getAllCaptureMoves(player, moves).size();
    }

    //ALL PIECES CENTER CONTROL

    /**
     * Checks if a piece is in the center (for center control).
     * @param node The current game state.
     * @param piece The pawn being checked for isolation.
     * @param board The current baord set up
     * @return Penalty value for isolated pawns.
     */
    private static double controlsCenter(DFSTreeNode node, Piece piece, Board board) {
        double centerControlUtility = 0.0;
        Coordinate position = piece.getCurrentPosition(board);
        int xCoordinate = position.getXPosition();
        int yCoordinate = position.getYPosition();
        
        if((xCoordinate>=4 && xCoordinate<=5) || (yCoordinate>=4 && yCoordinate<=5) ){
            //very center control
            centerControlUtility += 2.0;
        }else if((xCoordinate>=3 && xCoordinate<=6) || (yCoordinate>=3 && yCoordinate<=6) ){
            //semi center control
            centerControlUtility += 1.0;
        }
        

        return centerControlUtility;
    }

    //ALL PIECES: DEVELOPING BOARD
    /**
     * Calculate if a piece is developed (has moved beyond its starting rank).
     * @param piece The piece being checked for development.
     * @param board The current board
     * @param isBlack Boolean if we are tracking white or black development
     * @return Development score.
     */
    private static double isDevelopingBoard(Piece piece, Board board, boolean isBlack) {
        Coordinate position = piece.getCurrentPosition(board);
        int y = position.getYPosition();
        
        // Bonus for non-king pieces that are actively developing
        if (isBlack && y > 2 && piece.getType() != PieceType.KING) { //black player
            return 1.0;
        }else if (! isBlack && y < 7 && piece.getType() != PieceType.KING) { //white player
            return 1.0;
        }
        return 0.0;
    }

    //STATE OF GAME BASED ON WHATS LEFT
    
    /**
     * Determine the game stage based on remaining pieces.
     * @param wPieces The number of white pieces left.
     * @param bPieces The number of black pieces left.
     * @return The game stage: 0 for endgame, 1 for middle game, 2 for opening.
     */
    private static String stageOfGame(int wPieces, int bPieces) {
        // Simple thresholds for game phases
        if (wPieces < 7 || bPieces < 7) {
            return "end"; // Endgame
        } else if (wPieces < 15 || bPieces < 15) {
            return "middle"; // Middle game
        } else {
            return "open"; // Opening
        }
    }

    //KING

    /**
     * Evaluate the king safety for the max player by checking for threats near the king.
     * @param node The current game state.
     * @return King safety heuristic value.
     */
    public static double getKingSafety(DFSTreeNode node) {
        // Leverage pre-existing defensive heuristics
        // promotes king being surrounded
        return DefaultHeuristics.DefensiveHeuristics.getClampedPieceValueTotalSurroundingMaxPlayersKing(node);
    }

    //PAWNs

    /**
     * Checks if pawns are stacked up, and penalizes if it is.
     * @param piece The pawn we are checking if it is blocked by another pawn.
     * @param board The current board set up
     * @return Penalty value for doubled pawns.
     */
    private static double isDoubled(Piece piece, Board board) {
        double doublePawnPenalty = 0.0;
        Coordinate pawnPosition = piece.getCurrentPosition(board);
        int col = pawnPosition.getXPosition();
        int row = pawnPosition.getYPosition();
        boolean isWhite = piece.getPlayer().getAlgebraicSymbol().equals("W");
    
        // Loop through rows ahead (for white) or behind (for black)
        int direction = isWhite ? -1 : 1;
        
        // Check for pawns directly in front or behind in the same file
        for (int i = row + direction; i >= 0 && i < 10; i += direction) {
            Piece pieceInFile = board.getPieceAtPosition(new Coordinate(col, i));
            if (pieceInFile instanceof Pawn && pieceInFile.getPlayer() == piece.getPlayer()) {
                doublePawnPenalty -= 1.0;
                break;
            }
        }
    
        return doublePawnPenalty;
    }
    

    /**
     * Checks if a pawn is isolated (not protecting or protected).
     * @param piece The pawn being checked for isolation.
     * @param board The current board set up
     * @param player The player who owns this pawn
     * @return Penalty value for isolated pawns.
     */
    public static double isIsolated(Piece piece, Board board, Player player) {
        if (!(piece instanceof Pawn)) {
            return 0.0; // Only check isolation for pawns
        }

        int col = piece.getCurrentPosition(board).getXPosition();

        // Check left adjacent file for pawns of the same color
        if (col > 0) {
            for (int r = 0; r < 10; r++) {
                Piece adjacentPiece = board.getPieceAtPosition(new Coordinate(r, col-1));
                if (adjacentPiece instanceof Pawn && adjacentPiece.getPlayer() == player) {
                    return 0.0; // Pawn found in left adjacent file, not isolated
                }
            }
        }

        // Check right adjacent file for pawns of the same color
        if (col <9 ) {
            for (int r = 0; r < 10; r++) {
                Piece adjacentPiece = board.getPieceAtPosition(new Coordinate(r, col+1));
                if (adjacentPiece instanceof Pawn && adjacentPiece.getPlayer() == player) {
                    return 0.0; // Pawn found in right adjacent file, not isolated
                }
            }
        }

        // No pawns found in adjacent files
        return 1.0;
    }


    /**
     * Check if the list of moves contains a pawn promotion move.
     * @param moves The list of moves to check.
     * @return true if a promotion move is found, false otherwise.
     */
    public static boolean hasPromotionMove(List<Move> moves) {
        // Check if any move in the list has type PROMOTEPAWNMOVE
        return moves.stream().anyMatch(move -> move.getType() == MoveType.PROMOTEPAWNMOVE);
    }
    
    /**
     * Determines if a pawn is defending or is defended(increased utility)
     * @param pawn The pawn we are checking
     * @param board The current board
     * @param isBlack Boolean if the current piece is black or white
     * @param checkDefended Boolean if we are checking if it is defended or defending
     * @return
     */
    public static boolean isDefendingOrDefended(Piece pawn, Board board, boolean isBlack, boolean checkDefended) {
        Coordinate pawnPosition = pawn.getCurrentPosition(board);
        int x = pawnPosition.getXPosition();
        int y = pawnPosition.getYPosition();
        int direction = isBlack ? 1 : -1;
    
        // Check the diagonals depending on whether we're checking defending or defended
        for (int dx : new int[]{-1, 1}) {
            int newX = x + dx;
            int newY = y + (checkDefended ? -direction : direction);
    
            if (board.isPositionOccupied(new Coordinate(newX, newY))) {
                Piece otherPiece = board.getPieceAtPosition(new Coordinate(newX, newY));
                if (otherPiece instanceof Pawn && otherPiece.getPlayer() == pawn.getPlayer()) {
                    return true;
                }
            }
        }
        
        return false;
    }
    

    //check if the pawn has passed
    public static boolean isPassed(Piece pawn, Board board, boolean isBlack) {
        int col = pawn.getCurrentPosition(board).getXPosition();
        int row = pawn.getCurrentPosition(board).getYPosition();
        int direction = isBlack ? 1 : -1;
    
        for (int i = row + direction; i >= 0 && i < 10; i += direction) {
            if (board.isPositionOccupied(new Coordinate(col, i))) {
                Piece pieceAhead = board.getPieceAtPosition(new Coordinate(col, i));
                if (pieceAhead.getPlayer() != pawn.getPlayer()) {
                    return false; // Blocked by opponent's piece
                }
            }
        }
    
        return true; // No opponent blocking the pawn's advancement
    }
    

    /**
     * Evaluates the pawn structure and returns the utility based on the pawns.
     * @param node
     * @param piece
     * @param game
     * @param board
     * @param gameState
     * @param isBlack
     * @return
     */
    public static double evalPawnStructure(DFSTreeNode node, Piece piece,Game game, Board board, String gameState, boolean isBlack){
        double pawnStructureUtility = 0.0;
        
        pawnStructureUtility += isIsolated(piece, board, piece.getPlayer()); //penalty for isolated pawns
        pawnStructureUtility += isDoubled(piece, board); //penalty for doubled pawns

        if(hasPromotionMove(piece.getAllMoves(game))){
            pawnStructureUtility += 1.0; //utility for can pawn promote
        }
        
        if(isDefendingOrDefended(piece, board, isBlack, true)){
            pawnStructureUtility += 1.0;
        }
        if(isDefendingOrDefended(piece, board, isBlack, false)){
            pawnStructureUtility += 1.0;
        }
        if(isPassed(piece, board, isBlack)){
            pawnStructureUtility += 2.0;
        }
        pawnStructureUtility += 2 * controlsCenter(node, piece, board);
        return pawnStructureUtility;
    }


    /** Utility for queen movement/position (shouldn't move during opening phase)
     * @param node The current game state.
     * @param piece The queen piece being evaluated.
     * @param board The current board.
     * @param gameState The current phase of the game.
     * @return The utility value for the queen's movement.
     */
    public static double queenMovement(DFSTreeNode node, Piece piece, Board board, String gameState) {
        double queenUtility = 0.0;

        // Define the starting positions for the queens
        Set<Coordinate> queenSpots = new HashSet<>();
        queenSpots.add(new Coordinate(4, 8)); // Black queen starting position
        queenSpots.add(new Coordinate(4, 1)); // White queen starting position

        // Penalize if the queen moves during the opening phase and is not in starting position
        if (gameState.equals("open") && !queenSpots.contains(piece.getCurrentPosition(board))) {
            // Opening phase and queen has moved (heavy penalty)
            queenUtility -= 20.0;
        }
        return queenUtility;
    }

    /** Utility for bishop movement/position (shouldn't move unless 1 Knight has already moved/pawns are developed)
     * @param node The current game state.
     * @param gameState The current phase of the game.
     * @param pawnUtility The utility value for the pawn structure.
     * @param board The current board.
     * @param isBlack Boolean indicating if the current piece is black.
     * @return The utility value for the bishop's movement.
     */
    public static double bishopMovement(DFSTreeNode node, String gameState, double pawnUtility, Board board, boolean isBlack) {
        double bishopUtility = 0.0;

        // Determine if a bishop can move based on knight positions
        boolean canMoveBishopNow = false;
        if (isBlack) {
            // Check if at least one knight's position is empty
            if ((!board.isPositionOccupied(new Coordinate(2, 1)) || !board.isPositionOccupied(new Coordinate(7, 1))) || !gameState.equals("open")) {
                canMoveBishopNow = true; // At least one knight position is empty
            }
        } else {
            if ((!board.isPositionOccupied(new Coordinate(2, 8)) || !board.isPositionOccupied(new Coordinate(7, 8))) || !gameState.equals("open")) {
                canMoveBishopNow = true; // At least one knight position is empty
            }
        }

        // Reward bishop movement if pawns are developed and a knight can move
        if (pawnUtility > 0 && canMoveBishopNow) {
            bishopUtility += 1.0; // Reward bishop for being able to move
        }
        
        return bishopUtility;
    }


    /**
     * Calculates utility for castle position.
     * 
     * @param node The current tree node.
     * @param king The King piece.
     * @param rook1 First Rook piece.
     * @param rook2 Second Rook piece.
     * @param isBlack Boolean indicating if the player is black.
     * @param board The current board configuration.
     * @param gameState The current stage of the game.
     * @return The castling utility.
     */
    public static double castle(DFSTreeNode node, Piece king, Piece rook1, Piece rook2, Board board, boolean isBlack, String gameState) {
        Coordinate kiCoordinate = king.getCurrentPosition(board);
        double castleUtility = 0.0;

        // Castling is more valuable in the opening and middle game.
        if (!gameState.equals("end")) {
            castleUtility += getCastlingUtility(node, kiCoordinate, rook1, board);
            castleUtility += getCastlingUtility(node, kiCoordinate, rook2, board);
        }

        if(rook1!= null){
            castleUtility += evaluateRookPosition(kiCoordinate, rook1, isBlack, board, rook1.getPlayer());
        }
        if(rook2 != null){
            castleUtility += evaluateRookPosition(kiCoordinate, rook2, isBlack, board, rook2.getPlayer());
        }
        

        return castleUtility;
    }

    /**
     * Calculates utility for castling with a rook.
     */
    private static double getCastlingUtility(DFSTreeNode node, Coordinate kiCoordinate, Piece rook, Board board) {
        if (rook == null) return 0.0;
        Coordinate rookCoordinate = rook.getCurrentPosition(board);

        if(hasCastleMove(rook.getAllMoves(node.getGame())));

        // Add utility if the king and rook are in castling positions.
        if (rookCoordinate.getYPosition() == 1 || rookCoordinate.getYPosition() == 8) {
            if (rookCoordinate.getXPosition() == 4 && kiCoordinate.getXPosition() == 3) {
                return 5.0;
            } else if (rookCoordinate.getXPosition() == 6 && kiCoordinate.getXPosition() == 7) {
                return 5.0;
            }
        }
        return 0.0;
    }

    /**
     * Check if the list of moves contains a castle move.
     * @param moves The list of moves to check.
     * @return true if a promotion move is found, false otherwise.
     */
    public static boolean hasCastleMove(List<Move> moves) {
        // Check if any move in the list has type PROMOTEPAWNMOVE
        return moves.stream().anyMatch(move -> move.getType() == MoveType.CASTLEMOVE);
    }

    /**
     * Evaluates the position of a rook and adjusts the utility based on blocking and open lanes.
     * @param kiCoordinate The coordinate of the king.
     * @param rook The rook piece being evaluated.
     * @param isBlack Boolean indicating if the rook is black.
     * @param board The current board state.
     * @param player The player that owns this piece
     * @return The utility value for the rook's position.
     */
    private static double evaluateRookPosition(Coordinate kiCoordinate, Piece rook, boolean isBlack, Board board, Player player) {
        if (rook == null) return 0.0;

        Coordinate rookCoordinate = rook.getCurrentPosition(board);
        double utility = 0.0;

        // Penalize rooks directly behind the king.
        if (rookCoordinate.getXPosition() == kiCoordinate.getXPosition()) {
            if ((isBlack && rookCoordinate.getYPosition() < kiCoordinate.getYPosition()) ||
                (!isBlack && rookCoordinate.getYPosition() > kiCoordinate.getYPosition())) {
                utility -= (kiCoordinate.getYPosition() - rookCoordinate.getYPosition()) * 0.5; // Gradual penalty based on distance
            }
        }

        // Add utility for open lanes (row and column) for the rook.
        boolean isBlocked = isRowOrColBlocked(rookCoordinate, board);
        if (!isBlocked) {
            utility += 2.0; // Higher utility for completely open lanes
        } else {
            utility -= 0.5; // Penalty for being blocked
        }

        return utility;
    }

    /**
     * Checks if there are any pieces blocking the row or column of the rook.
     * @param rookCoordinate The coordinate of the rook.
     * @param board The current board state.
     * @return True if the row or column is blocked, false otherwise.
     */
    private static boolean isRowOrColBlocked(Coordinate rookCoordinate, Board board) {
        // Check the row for blockage
        for (int y = 1; y <= 8; y++) {
            if (y != rookCoordinate.getYPosition() && board.isPositionOccupied(new Coordinate(rookCoordinate.getXPosition(), y))) {
                return true;
            }
        }
        // Check the column for blockage
        for (int x = 1; x <= 8; x++) {
            if (x != rookCoordinate.getXPosition() && board.isPositionOccupied(new Coordinate(x, rookCoordinate.getYPosition()))) {
                return true;
            }
        }
        return false;
    }

    /**
     * Evaluates the position of a knight and adjusts the utility based on mobility and position.
     */
    private static double evaluateKnightPosition(Piece knight, Board board) {
        if (knight == null) return 0.0;

        Coordinate knightCoordinate = knight.getCurrentPosition(board);
        double utility = 0.0;

        // Knight mobility: Count the number of legal moves from the current position
        int mobility = getKnightMobility(knight, knightCoordinate, board);
        utility += mobility; // Higher mobility increases utility

        // Center control: Reward knights in the center of the board
        if (isKnightInCenter(knightCoordinate)) {
            utility += 1.5; // Bonus for being in the center
        }

        // Penalize knights on the edge of the board
        if (isKnightOnEdge(knightCoordinate)) {
            utility -= 1.0; // Penalty for being on the edge
        }

        return utility;
    }

    /**
     * Calculates the number of legal moves for a knight from its current position.
     */
    private static int getKnightMobility(Piece knight, Coordinate knightCoordinate, Board board) {
        int mobilityCount = 0;
        int[][] knightMoves = {
            {2, 1}, {2, -1}, {-2, 1}, {-2, -1},
            {1, 2}, {1, -2}, {-1, 2}, {-1, -2}
        };

        for (int[] move : knightMoves) {
            int newX = knightCoordinate.getXPosition() + move[0];
            int newY = knightCoordinate.getYPosition() + move[1];

            Coordinate newC = new Coordinate(newX, newY);
            if (board.isInbounds(newC)) {
                Piece targetPiece = board.getPieceAtPosition(newC);
                // Count as a valid move if the target position is either empty or occupied by an opponent's piece
                if (targetPiece == null || targetPiece.getPlayer() != knight.getPlayer()) {
                    mobilityCount++;
                }
            }
        }
        return mobilityCount;
    }

    /**
     * Checks if the knight is in the center of the board.
     */
    private static boolean isKnightInCenter(Coordinate knightCoordinate) {
        int x = knightCoordinate.getXPosition();
        int y = knightCoordinate.getYPosition();
        return (x >= 4 && x <= 5) && (y >= 4 && y <= 5); // Assuming a standard 8x8 board
    }

    /**
     * Checks if the knight is on the edge of the board.
     */
    private static boolean isKnightOnEdge(Coordinate knightCoordinate) {
        int x = knightCoordinate.getXPosition();
        int y = knightCoordinate.getYPosition();
        return (x == 1 || x == 8 || y == 1 || y == 8); // Assuming a standard 8x8 board
    }



    // Helper functions to get max and min players from the node
    private static Player getMaxPlayer(DFSTreeNode node) {
        return node.getMaxPlayer();
    }

    private static Player getMinPlayer(DFSTreeNode node) {
        return node.getMaxPlayer().equals(node.getGame().getCurrentPlayer()) 
            ? node.getGame().getOtherPlayer() 
            : node.getGame().getCurrentPlayer();
    }
}
