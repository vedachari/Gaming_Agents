package src.pas.tetris.agents;


import java.awt.Color;
import java.util.ArrayList;
import java.util.Comparator;
// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import edu.bu.tetris.utils.Coordinate;
// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Block;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction() {
        final int featureSize = (Board.NUM_COLS * Board.NUM_ROWS) + 3;
        // final int featureSize = (Board.NUM_COLS * Board.NUM_ROWS) + 5;
        Sequential qFunction = new Sequential();
    
        // Input to First Hidden Layer
        qFunction.add(new Dense(featureSize, 16));
        // qFunction.add(new ReLU());
        qFunction.add(new Tanh());
    
        // Output Layer
        qFunction.add(new Dense(16, 1));
    
        return qFunction;
    }
    

    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
    @Override
    public Matrix getQFunctionInput(final GameView game, final Mino potentialAction) {
        Matrix features = null;
        try {
            // Grayscale image (flattened to row vector)
            Matrix grayscaleImage = game.getGrayscaleImage(potentialAction);
            Matrix flattenedImage = grayscaleImage.flatten();

            Board board = new Board(game.getBoard());
            board.addMino(potentialAction);

            features = Matrix.zeros(1, (Board.NUM_COLS * Board.NUM_ROWS) + 3); // Row vector
            // features = Matrix.zeros(1, (Board.NUM_COLS * Board.NUM_ROWS) + 5); // Row vector
            // double[] columns = columnHeights(game);
            double maxColumn = maxColumnHeights(game);
            // double row = bottomRowControl(board);
            // double bumpiness =  bumpiness(board);
            double holes = 0.0;
            for (int col = 0; col < Board.NUM_COLS; col++) {
                boolean blockFound = false;
                for (int row = 0; row < Board.NUM_ROWS; row++) {
                    if (grayscaleImage.get(row, col) != GameView.UNOCCUPIED_COORDINATE_VALUE) {
                        if (!blockFound) {
                            blockFound = true;
                        }
                    } else if (blockFound) {
                        holes++;
                    }
                }
            }
            double linesCleared = completeLines(potentialAction, game);

            int idx = 0; // Start at the beginning of the feature vector

            // Add flattened grayscale board state to features
            for (int i = 0; i < flattenedImage.numel(); i++) {
                features.set(0, idx++, flattenedImage.get(0, i));
            }

            // // Add column heights
            // for (double columnHeight : columns) {
            //     features.set(0, idx++, (columnHeight /( Board.NUM_ROWS - 2)));
            // }

            features.set(0, idx++, holes); // Normalize holes (assumes max ~200 holes)
            features.set(0, idx++, maxColumn ); // Normalized by max possible height
            // features.set(0, idx++, row / Board.NUM_COLS); // Normalize the Row Control
            // features.set(0, idx++, bumpiness); // Normalize bumpiness similarly
            features.set(0, idx++, linesCleared); // Normalize by max line clears in one move
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return features;
    }

    //calculates the number of completed lines on the board
    //code adapted from Board class
    public double completeLines(Mino potentialAction, GameView game){
        int lines = 0;
        Block[] blocks = potentialAction.getBlocks();
        for (int row = 0; row < Board.NUM_ROWS; row++) {
            boolean isFullRow = true;
            for (int col = 0; col < Board.NUM_COLS; col++) {
                boolean isOccupied = game.getBoard().isCoordinateOccupied(col, row);
                for (Block block : blocks) {
                    Coordinate coord = block.getCoordinate();
                    if (coord.getXCoordinate() == col && coord.getYCoordinate() == row) {
                        isOccupied = true;
                        break;
                    }
                }
                if (!isOccupied) {
                    isFullRow = false;
                    break;
                }
            }
            if (isFullRow) {
                lines++;
            }
        }

        return lines; //square the completeLines (more complete lines the better)
    }

    public double maxColumnHeights(GameView game){
        double maxHeight = 0;
        double[] columnHeights = new double[Board.NUM_COLS];
        for (int col = 0; col < Board.NUM_COLS; col++) {
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                if (game.getBoard().getBlockAt(col, row) != null) {
                    columnHeights[col] = Board.NUM_ROWS - row;
                    maxHeight = Math.max(maxHeight, columnHeights[col]);
                    break;
                }
            }
        }
        return maxHeight;
    }

    public double[] columnHeights(GameView game){
        double[] columnHeights = new double[Board.NUM_COLS];
        for(int col = 0; col<Board.NUM_COLS; col++){
            for(int row = 2; row < Board.NUM_ROWS; row++){
                if (game.getBoard().getBlockAt(col, row) != null) {
                    columnHeights[col] = Board.NUM_ROWS - row;
                    break;
                }
            }
        }
        return columnHeights;
    }


    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {
        double trainingProgress = (double) gameCounter.getCurrentGameIdx() / gameCounter.getNumTrainingGames();
        double explorationProb = Math.max(0.1, 1.0 - trainingProgress);
        return this.getRandom().nextDouble() < explorationProb;
    }
    
    

    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game) {
        List<Mino> minos = game.getFinalMinoPositions(); // Potential moves
        // try {
            // int featureSize = Board.NUM_ROWS * Board.NUM_COLS + 3; // Align with initQFunction
            // // Matrix inputs = Matrix.zeros(minos.size(), featureSize);

            // // for (int i = 0; i < minos.size(); i++) {
            // //     Mino potentialAction = minos.get(i);
            // //     if (potentialAction == null) {
            // //         System.err.println("Error: Potential Action is null.");
            // //         continue; // Skip invalid actions
            // //     }

            // //     Matrix qInput = getQFunctionInput(game, potentialAction);
            // //     inputs.copySlice(i, i + 1, 0, featureSize, qInput);
            // // }

            // Median-based exploration

        // } catch (Exception e) {
        //     e.printStackTrace();
        //     return null;
        // }
        int medianIndex = minos.size() / 2;
        return minos.get(medianIndex);
    }

    

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }


    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game) {
        double score = game.getScoreThisTurn();

        // double reward = 0.0;
        double reward = score * 500.0;
        double maxHeight = maxColumnHeights(game);
        reward -= maxHeight * 2;
    
        // Penalize holes and unevenness
        double totalHoles = holes(game);
        double heightDifferences = bumpiness(game);
        reward -= totalHoles * 5;
        reward -= heightDifferences * 3;
        // reward += 2 * bottomRowControl(game.getBoard());
        return reward;
    }

    /*
     * row control
     */
    public double rowControl(Board board) {
        double control = 0.0;
    
        control += 2.0 * partiallyFilledRows(board)/Board.NUM_ROWS; // Reward nearly full rows
        control += 4.0 * sideControl(board)/ 18.0;          // Penalize gaps
        control += 4.0 * bottomRowControl(board) / Board.NUM_COLS;
        control += 2.0 * rowClearPotential(board);   // Reward clearable rows
        if(numTotalBlocks(board) != 0){
            control -= 1.5 * scatteredBlocksPenalty(board)/ numTotalBlocks(board); // Penalize scattered blocks
        }
        
        return control;
    }

    /*
     * prevent scattered pieces
     */
    public double isolatedBlocks(Board board) {
        double isolatedCount = 0.0;
    
        // Iterate through the board
        for (int row = 2; row < Board.NUM_ROWS; row++) { // Ignore buffer rows
            for (int col = 0; col < 10; col++) {
                Block currentBlock = board.getBlockAt(col, row);
    
                // Only proceed if there's a block at the current position
                if (currentBlock != null) {
                    boolean hasSameColorNeighbor = false;
                    Color currentColor = board.getBlockAt(col, row).getColor();
    
                    // Check neighbors
                    if (row > 2) { // Above
                        Block above = board.getBlockAt(col, row - 1);
                        if (above != null && above.getColor().equals(currentColor)) {
                            hasSameColorNeighbor = true;
                        }
                    }
                    if (row < Board.NUM_ROWS -1) { // Below
                        Block below = board.getBlockAt(col, row + 1);
                        if (below != null && below.getColor().equals(currentColor)) {
                            hasSameColorNeighbor = true;
                        }
                    }
                    if (col > 0) { // Left
                        Block left = board.getBlockAt(col - 1, row);
                        if (left != null && left.getColor().equals(currentColor)) {
                            hasSameColorNeighbor = true;
                        }
                    }
                    if (col < Board.NUM_COLS - 1) { // Right
                        Block right = board.getBlockAt(col + 1, row);
                        if (right != null && right.getColor().equals(currentColor)) {
                            hasSameColorNeighbor = true;
                        }
                    }
    
                    // If no neighbors with the same color, count this block as isolated
                    if (!hasSameColorNeighbor) {
                        isolatedCount += 1.0;
                    }
                }
            }
        }
    
        return isolatedCount;
    }
    
    

    /* 
     * bottom row control
    */
    public double bottomRowControl(Board board) {
        int blocks = 0;
        for (int col = 0; col < 10; col++) {
            if (board.isCoordinateOccupied(col, (Board.NUM_ROWS -1))) {
                blocks ++;
            }
        }
        return blocks; // Reward for a complete bottom row
    }
    

    /*
     * number total blocks
     */
    public double numTotalBlocks(Board board){
        double num = 0.0;

        for(int col = 0; col < 10; ++col) {
            for(int row = Board.NUM_ROWS - 1; row >=2 ; row --) {
                if (board.isCoordinateOccupied(col, row)) {
                    num += 1.0;
                }
            }
        }
        return num;
    }

    //reward partially filled rows
    public double partiallyFilledRows(Board board) {
        double score = 0.0;
        for (int row = 2; row < Board.NUM_ROWS; row++) { // Evaluate rows within the visible play area
            int filledBlocks = 0;
            for (int col = 0; col < 10; col++) {
                if (board.isCoordinateOccupied(col, row)) {
                    filledBlocks++;
                }
            }
            if (filledBlocks >= 8) { // Row is mostly full
                score += (filledBlocks / 10.0); // Reward proportional to row fullness
            }
        }
        return score;
    }
    

    /*
     * Add points for filling up sides
     */
    public double sideControl(Board board){
        double side = 0.0;

        for(int row = Board.NUM_ROWS - 1; row >= Board.NUM_ROWS - 6; row --){
            if(board.isCoordinateOccupied(0, row)){
                side += 3.0;
            }else{
                side -= 1.0;
            }
            if(board.isCoordinateOccupied(9, row)){
                side += 3.0;
            }else{
                side -= 1.0;
            }
        }
        return side;
    }

    /*
     * reward for rows that can be cleared in 1-2 moves
     */
    public double rowClearPotential(Board board) {
        double potential = 0.0;
        for (int row = 2; row <Board.NUM_ROWS; row++) {
            int emptySpaces = 0;
            for (int col = 0; col < 10; col++) {
                if (!board.isCoordinateOccupied(col, row)) {
                    emptySpaces++;
                }
            }
            if (emptySpaces == 1) { // Only one block missing to clear the row
                potential += 5.0;
            } else if (emptySpaces == 2) { // Two blocks missing
                potential += 2.0;
            }
        }
        return potential;
    }

    /*
     * pentaly for scattered blocks
     */
    public double scatteredBlocksPenalty(Board board) {
        double penalty = 0.0;
        for (int row = 2; row < Board.NUM_ROWS; row++) {
            int filledBlocks = 0;
            int firstBlock = -1;
            int lastBlock = -1;
            for (int col = 0; col < Board.NUM_COLS; col++) {
                if (board.isCoordinateOccupied(col, row)) {
                    filledBlocks++;
                    if (firstBlock == -1) firstBlock = col;
                    lastBlock = col;
                }
            }
            if (filledBlocks > 0) {
                penalty += (lastBlock - firstBlock + 1 - filledBlocks); // Penalize gaps within the range
            }
        }
        return penalty;
    }
    
    /*
     * Sum the heights of all the columns
     */
    public double aggregateHeight(Board board){
        double height = 0.0;
        // Loop through each column
        for (int col = 0; col < Board.NUM_COLS; ++col) {
            for (int row = Board.NUM_ROWS -1; row >= 2; row--) {
                if (board.isCoordinateOccupied(col, row)) { //first encountered block
                    height += (22 - row); // Height is the distance from row to bottom (row (Board.Num_ROWS -1))
                    break; // Stop after finding the highest occupied cell
                }
            }
        }
        return height;
    }

    //calculates the number of holes in the board
    public double holes(GameView game){
        double num_holes = 0;
        double[] height = columnHeights(game);
        for (int col = 1; col < Board.NUM_COLS; col++) {
            num_holes += Math.max(0, height[col - 1] - height[col]);
        }
        return num_holes;
    }
    /*
     * Sum the heights of all the columns
     */
    public double bumpiness(GameView game) {
        int sum = 0;
        double[] height = columnHeights(game);
        for (int col = 1; col < Board.NUM_COLS; col++) {
            sum += Math.abs(height[col] - height[col - 1]);
        }
        return sum;
    }

    /*
     * max bump in board
     */
    public double maxBump(Board board) {
        double max = Double.NEGATIVE_INFINITY;
        int previousHeight = 0;

        // Loop through each column
        for (int col = 0; col < 10; ++col) {
            int height = 0;
            for (int row = Board.NUM_ROWS -1; row >=2 ; row --) {
                if (board.isCoordinateOccupied(col, row)) {
                    height = row;
                    break;
                }
            }
            // Calculate bumpiness only after the first column
            if (col > 0) {
                int diff = Math.abs(height - previousHeight);
                if(diff > max){
                    max = diff;
                }
            }
            previousHeight = height;
        }
        return max;
    }

}