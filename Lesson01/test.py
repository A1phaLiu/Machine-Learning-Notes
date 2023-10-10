#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
class Employee:
   '所有员工的基类'
   empCount = 0
 
   def __init__(self, name, salary):
      self.name = name
      self.salary = salary
      Employee.empCount += 1
   
   def displayCount(self):
     print(f"Total Employee {Employee.empCount}")

   def displayemp(self):
     print(f"name {self.name}  Salary {self.salary}")
 

 
"创建 Employee 类的第一个对象"
emp1 = Employee("Zara", 2000)
"创建 Employee 类的第二个对象"
emp2 = Employee("Manni", 5000)

print(emp1.name,emp1.salary)
print(emp2.name,emp2.salary)

emp1.displayemp()
emp2.displayemp()
