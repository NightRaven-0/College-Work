import java.util.ArrayList;
import java.util.Iterator;
import java.util.Scanner;

public class CountrySearch {
    public static void main(String[] args) {
        ArrayList<String> countries = new ArrayList<>();
        countries.add("America");
        countries.add("India");
        countries.add("Russia");
        countries.add("Brazil");
        countries.add("Columbia");

        Scanner scanner = new Scanner(System.in);
        System.out.println("Enter a country's name:");
        String searchCountry = scanner.nextLine();

        Iterator<String> iterator = countries.iterator();
        int ranking = 1;
        boolean found = false;

        while (iterator.hasNext()) {
            String country = (String) iterator.next();
            if (country.equalsIgnoreCase(searchCountry)) {
                System.out.println("Ranking: " + ranking);
                found = true;
                break;
            }
            ranking++;
        }

        if (!found) {
            System.out.println("Country not found.");
        }
        scanner.close();
    }
}