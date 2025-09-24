import java.util.Scanner;

public class n_queens {
    private int n;
    private int[] queens;  
    private boolean[] cols;  
    private boolean[] diag1;  
    private boolean[] diag2; 
    private int solutionCount = 0;

    public n_queens(int n) {
        this.n = n;
        queens = new int[n];
        cols = new boolean[n];
        diag1 = new boolean[2 * n];
        diag2 = new boolean[2 * n];
    }

    private void solve() {
        place(0);
        System.out.println("\nTotal solutions: " + solutionCount);
    }

    private void place(int row) {
        if (row == n) {
            solutionCount++;
            printCoordinates();
            return;
        }

        for (int c = 0; c < n; c++) {
            if (isSafe(row, c)) {
                setQueen(row, c, true);
                place(row + 1);
                setQueen(row, c, false); // backtrack
            }
        }
    }

    private boolean isSafe(int r, int c) {
        return !cols[c] && !diag1[r + c] && !diag2[r - c + (n - 1)];
    }

    private void setQueen(int r, int c, boolean place) {
        queens[r] = c;
        cols[c] = place;
        diag1[r + c] = place;
        diag2[r - c + (n - 1)] = place;
    }

    // Print coordinates for the solution
    private void printCoordinates() {
        System.out.println("Solution " + solutionCount + ":");
        for (int r = 0; r < n; r++) {
            int c = queens[r];
            System.out.println("Queen at: (" + r + ", " + c + ")");
        }
        System.out.println();
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter board size n: ");
        int n = sc.nextInt();
        if (n <= 0) {
            System.out.println("n must be >= 1");
            sc.close();
            return;
        }

        n_queens solver = new n_queens(n);
        solver.solve();

        sc.close();
    }
}
