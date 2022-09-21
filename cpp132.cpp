#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

void uocsomax(long long n){
	for(int i = 2 ; i <= sqrt(n ); i ++ ){
		while (n%i==0 ){
			n/=i;
		}
		if (n == 1 ){
			cout << i << endl;
		}
	}
	if (n != 1 ){
		cout << n << endl ;
	}
}

int main (){
	int m ; 
	cin >> m ;
	while (m--){
		long long a ;
		cin >> a ; 
		uocsomax(a);
	}
}
