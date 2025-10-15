import java.util.Scanner;
public class CRT {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter the number of congruences: ");
        int n = sc.nextInt();

        long[] a = new long[n];
        long[] m = new long[n]; 

        for (int i = 0; i < n; i++) {
            System.out.print("Enter remainder " + (i + 1) + ": ");
            a[i] = sc.nextLong();
            System.out.print("Enter modulus " + (i + 1) + ": ");
            m[i] = sc.nextLong();
        }

        if (isCRTValid(a, m)) {
            System.out.println("The given values satisfy the Chinese Remainder Theorem.");
        } else {
            System.out.println("The given values do not satisfy the Chinese Remainder Theorem.");
        }
    
        sc.close();
    }
    
    public static boolean isCRTValid(long[] a, long[] m) {
        for (int i = 0; i < m.length; i++) {
            for (int j = i + 1; j < m.length; j++) {
                if (gcd(m[i], m[j]) != 1) {
                    return false;
                }
            }
        }

        for (int i = 0; i < a.length; i++) {
            if (a[i] < 0 || a[i] >= m[i]) {
                return false;
            }
        }

        return true;
    }

    public static long gcd(long a, long b) {
        if (b == 0) {
            return a;
        }
        return gcd(b, a % b);
    }
}