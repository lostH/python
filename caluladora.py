
def Menu():
    
    print (""" ************
Calculadora
************
        Menu
       1) Suma
       2) Resta
       3) Multiplicacion
       4) Division
       5) para Salir """)
def Calculadora():
    
    Menu()
    o = int(input("Selecione opcion\n"))
    
    while (o != 5):
        x = int(input("Ingrese Numero\n"))
        y = int(input("Ingrese Otro Numero\n"))
        if (o==1):
            print ("La Suma es:", x+y)
            o = int(input("Selecione Opcion\n"))
        elif(o==2):
            print ("La Resta es:",x-y)
            o = int(input("Selecione Opcion\n"))
        elif(o==3):
            print ("La Multiplicacion es:",x*y)
            o = int(input("Selecione Opcion\n"))
        elif(o==4):
            try:
                print ("La Division es:", x/y)
                o = int(input("Selecione Opcion\n"))
            except ZeroDivisionError:
                print ("No se Permite la Division Entre 0")
                o = int(input("Selecione Opcion\n"))
        else:
            print ("No es una opcion")
            o = int(input("Selecione Opcion\n"))
    print ("Programa terminado")
   
        
Calculadora()
