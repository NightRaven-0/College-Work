import java.util.Scanner;
public class chocolate 
{
    public static void main(String[] args) 
    {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the number of children: ");
        int child = sc.nextInt();
        sc.close();
        int totalChocolates = calcchoco(child);
        System.out.println("Total chocolates to be distributed: " + totalChocolates);
    }
    public static int calcchoco(int child) 
    {
        int chocolates = 0 ;
        for (int i = 1; i <= child; i++) 
        {
            if (child < 4)
            chocolates += i ;
            else if( child >= 4)
            {
                for (int j = 1; j <= child; j++) 
                {
                    if (j == 1)
                    {
                        chocolates = 1;
                    }
                    else if ( j > 1)
                        if ((j - 1) % 5 == 0 || (j + 1) % 5 == 0)
                        {
                            chocolates += j + 2; 
                        } 
                else
                {
                    chocolates += j ;
                }
            }
        return chocolates;
    }
        }
        return chocolates;
    }
}