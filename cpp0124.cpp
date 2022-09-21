#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

void thuaso(long n ){
	for (int i = 2 ; i <= sqrt(n);i ++){
		if (n%i == 0 ){
			int dem = 0 ;
			while (n%i==0){
				++dem ; 
				n/=i;
			}
			cout << i << " "<< dem << endl;
		}
	}
	if ( n!=1){
		cout << n <<" "<< 1 << endl; 
	}
}

int main (){
	long n;
	cin >> n ;
	thuaso(n);
}
