import java.util.*;

public class  combinations {

    // Recursive helper to generate combinations
    public static void combine(char[] arr, int k, int start, StringBuilder current, List<String> result) {
        if (current.length() == k) {
            result.add(current.toString());
            return;
        }
        for (int i = start; i < arr.length; i++) {
            current.append(arr[i]);
            combine(arr, k, i + 1, current, result);
            current.deleteCharAt(current.length() - 1); // backtrack
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter string: ");
        String input = sc.nextLine();

        System.out.print("Enter k: ");
        int k = sc.nextInt();

        char[] arr = input.toCharArray();
        List<String> result = new ArrayList<>();

        combine(arr, k, 0, new StringBuilder(), result);

        System.out.println("Combinations:");
        for (String comb : result) {
            System.out.println(comb);
        }

        sc.close();
    }
}
