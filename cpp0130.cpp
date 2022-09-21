#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

void uocso(long long n ){
	for ( int i = 2 ; i <= sqrt(n) ; i++){
		while (n%i == 0 ){
			cout << i << " " ;
			n/=i;
		}
	}
	if( n != 1 ){
		cout << n << " " ;
	}
}

int main (){
	int m ;
	cin >> m ;
	while (m --){
		long long  a ;
		cin >> a ; 
		uocso(a);
		cout << endl;
	}
}
