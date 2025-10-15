import java.util.Scanner;
class abc
{    
    public static void main(String[] args)
    {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter the encoded message");
        String s1 = sc.nextLine();
        String t= " ";
        int n = 0;
        char ch = 0;
        String s2 = " ";
        String f =" ";
        int l = s1.length();
          if(l>200)
        {
            System.out.println("Input code exceeded..now exiting");
            System.exit(0);
        }
        sc.close();
        int i = 0;
        for (i=0;i<l;i++)
        {
            ch = s1.charAt(i);
            s2 = ch+s2;
        }
        System.out.println("The decoded message : ");
        for (i=0; i<l-2; i++)
        {
            t=s2.substring(i,i+2);
            n=Integer.parseInt(t);
            if(n==32)
            {
                f = f+" ";
                i++;
            }
        }
    }
}