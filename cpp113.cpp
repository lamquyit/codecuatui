#include <iostream>
#include <iomanip>
#include <math.h>

using namespace std ;

bool somayman(long n){
	int a , b;
	b = n%10 ;
	n/=10;
	a = n %10 ;
	if(a==8 && b == 6 ){
		return 1;
	}
	return 0 ;
}

int main(){
	int m ; 
	cin >> m ;
	while (m--){
		long a ;
		cin >> a ;
		cout << somayman(a) << endl;
	}
}
