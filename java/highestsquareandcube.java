import java.util.Scanner;

public class highestsquareandcube 
{
    public static void main(String[] args) 
    {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a value: ");
        int value = sc.nextInt();
        sc.close();
        int sqa = findHighestSquare(value);
        int cba = findHighestCube(value);
        System.out.println("Highest square less than or equal to " + value + ": " + sqa);
        System.out.println("Highest cube less than or equal to " + value + ": " + cba);
        System.out.println("Sum of highest square and highest cube: " + (sqa + cba));
    }
    public static int findHighestSquare(int value) 
    {
        int i = (int) Math.sqrt(value);
        return i * i;
    }
    public static int findHighestCube(int value) 
    {
        int i = (int) Math.cbrt(value);
        return i * i * i;
    }
}