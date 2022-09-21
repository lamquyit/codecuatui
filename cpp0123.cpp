#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

int	prime (long n ){
	for ( int i = 2 ; i <= sqrt(n);i ++){
		if(n % i == 0 ){
			return 0 ;
		}
	}
	return n > 1;
}

int main (){
	long n;
	cin >> n ;
	if(prime(n)){
		cout << "YES"<<endl;
	}else{
		cout << "NO" << endl;
	}
}
