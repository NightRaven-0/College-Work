import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
public class CreateTable 
{
    public static void main(String[] args) 
    {
        String dbUrl = "jdbc:mysql://localhost:3306/mydatabase";
        String username = "myuser";
        String password = "mypassword";
        
        try (Connection conn = DriverManager.getConnection(dbUrl, username, password)) 
        {
            try (Statement stmt = conn.createStatement()) 
            { 
                String query = "CREATE TABLE Student (" +
                        "student_name VARCHAR(255), " +
                        "roll_number INT, " +
                        "age INT, " +
                        "gender VARCHAR(10))";
                stmt.executeUpdate(query);
                System.out.println("Table created successfully!");
            }
        }
        
        catch (SQLException e) 
        {
            System.out.println("Error creating table: " + e.getMessage());
        }
    }
}