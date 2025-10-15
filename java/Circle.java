import java.util.*;
public class Circle implements shape,figure
{
    double r;
    void circle(double r)
    {
        this.r = r;
    }
    public double area()
    {
        return 3.14*r*r;
    }
    public String name()
    {
        return "circle";
    }
    public static void main(String[] args) 
    {
        Scanner sc = new Scanner(System.in);
        Circle c = new Circle();
        System.err.println("Enter the radius: ");
        double r = sc.nextDouble();
        c.circle(r);
        System.out.println("Area of "+c.name()+" is "+c.area());
        sc.close();
    }
}
