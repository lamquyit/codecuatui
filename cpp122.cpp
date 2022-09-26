#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

long ucln(int a , int b){
	int i ;
	while (a%b !=0){
		i = a % b;
		a = b ;
		b = i ;
	}
	return b ;
}

long bcnn(int a , int b ){
	return a/(ucln(a,b)) * b;
}

long sotm(int a){
	int x =1 ;
	if(a == 0) 
		return 1 ;
		else{
			for(int i = 1; i <= a; i++){
				x = ucln(x,i);
			}
			return x ;
		}
}

int main (){
	int m ;
	cin >> m ; 
	while(m--){
	unsigned long a ; long T;
	do{
		cin >> T;
	}while (T<1 || T>10000);
	for ( int i = 0; i <T;i++){
		while (a<0){
			cin >> a ;
		}
	}
		cout << sotm(a)<< endl;
	}
	return 0;
}
