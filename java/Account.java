import java.util.Scanner;
public class Account 
{
    private int accId;
    private String name;
    private double balance;
    private static double rateOfInterest;
    public Account(int accId, String name, double balance) 
    {
        this.accId = accId;
        this.name = name;
        this.balance = balance;
    }
    public void displayDetails() 
    {
        System.out.println("Account ID: " + accId);
        System.out.println("Name: " + name);
        System.out.println("Balance: " + balance);
        System.out.println("Rate of Interest: " + rateOfInterest + "%");
    }
    public static void setRateOfInterest(double rate) 
    {
        if (rateOfInterest == 0) {
            rateOfInterest = rate;
            System.out.println("Rate of Interest set to " + rate + "%");
        } else {
            System.out.println("Rate of Interest is already set and cannot be changed.");
        }
    }
    public static void main(String[] args) 
    {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter rate of interest: ");
        double rate = sc.nextDouble();
        Account.setRateOfInterest(rate);

        Account account1 = new Account(1, "John Doe", 1000.0);
        account1.displayDetails();

        Account account2 = new Account(2, "Jane Smith", 2000.0);
        account2.displayDetails();
        Account.setRateOfInterest(8.0);
        sc.close();
    }
}
