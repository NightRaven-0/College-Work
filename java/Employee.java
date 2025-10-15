public class Employee 
{
        private int id;
        private String name;
        private int age;
        public Employee(String name, int age) 
        {
            this.name = name;
            this.age = age;
            this.id = generateNextID();
        }   
        public void show() 
        {
            System.out.println("ID: " + id);
            System.out.println("Name: " + name);
            System.out.println("Age: " + age);
        }   
        public int generateNextID() 
        {
            return ++Employee.lastID;
        }
        private static int lastID = 0;
        public static void showNextID() 
        {
            System.out.println("Next ID: " + (Employee.lastID + 1));
        }
        public static void main(String[] args) 
        {    
        Employee employee1 = new Employee("John Doe", 25);
        employee1.show();
        Employee employee2 = new Employee("Jane Smith", 30);
        employee2.show();
        Employee.showNextID();
        }
    }