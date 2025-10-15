import java.util.Scanner;

class abc {    
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter the encoded message:");
        String s1 = sc.nextLine();
        sc.close();
        
        int l = s1.length();
        if (l > 200) {
            System.out.println("Input code exceeded..now exiting");
            System.exit(0);
        }
        
        String f = "";
        for (int i = 0; i < l; i += 2) {
            // Ensure that there are at least two characters remaining in the input string
            if (i + 1 < l) {
                String t = s1.substring(i, i + 2);
                int n = Integer.parseInt(t);
                if (n == 32) {
                    f = f + " ";
                } else {
                    f = f + (char)n;
                }
            }
        }
        
        
        System.out.println("The decoded message: " + f);
    }
}
