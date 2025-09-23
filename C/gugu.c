#include<conio.h>
#include<stdio.h>

void main()
{
    int year;
    printf("Enter a year: ");
    scanf_s("%d", &year);

    if ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0))
        printf("It's a me ! a Leap year!\n");
    else
        printf("Not a leap year\n");

    _getch();
}