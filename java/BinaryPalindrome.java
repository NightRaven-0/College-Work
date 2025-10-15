import java.util.Scanner;

public class BinaryPalindrome {
    public static void main(String[] args) 
    {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter an integer: ");
        int num = sc.nextInt();
        sc.close();
        String binary = Integer.toBinaryString(num);
        System.out.println("Binary representation: " + binary);

        if (isBinaryPalindrome(binary)) 
        {
            System.out.println("The binary representation is a palindrome.");
        } 
        else 
        {
            System.out.println("The binary representation is not a palindrome.");
        }
    }
    public static boolean isBinaryPalindrome(String binary) 
    {
        int start = 0;
        int end = binary.length() - 1;
        while (start < end) 
        {
            if (binary.charAt(start) != binary.charAt(end)) 
            {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }
}